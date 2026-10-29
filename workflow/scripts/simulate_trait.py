#!/usr/bin/env python3
"""Simulate a discrete trait history on a phylogeny (Stage 1, WP6).

Produces the three inputs a trait analysis needs, plus the ground truth:

``<prefix>.trees``      NEXUS tree file (consumed by ``empiricalTreeDistributionModel``)
``<prefix>.traits.csv`` ``taxon,state`` observations at the tips
``<prefix>.truth.tsv``  every simulated transition with its height and calendar date
``<prefix>.sim.json``   the simulation settings and a summary

The tree is either generated (a coalescent tree with ``--n-tips`` tips scaled to
``--tree-height``) or read from ``--input-tree``.  A continuous-time Markov chain
is then simulated down the tree: waiting times are exponential with rate
``-q_ii`` and destinations are drawn from the row of the rate matrix.  With
``--n-events N`` the simulation is repeated until exactly ``N`` transitions
occurred, which makes a single-gain (or single-loss, or recurrent) scenario
exactly reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dates import format_date, height_to_decimal_year, parse_date

MODELS = ("equal", "asymmetric")


class SimulationError(Exception):
    pass


# --------------------------------------------------------------------------
# minimal Newick support
# --------------------------------------------------------------------------
@dataclass
class Node:
    name: str | None = None
    length: float = 0.0
    children: list["Node"] = field(default_factory=list)

    @property
    def is_tip(self) -> bool:
        return not self.children


def parse_newick(text: str) -> Node:
    """Parse a Newick string.  BEAST-style ``[&...]`` annotations are dropped."""
    text = re.sub(r"\[&[^\]]*\]", "", text)
    text = text.strip()
    if text.startswith("["):  # leading [&R]
        text = text[text.index("]") + 1:]
    text = text.rstrip().rstrip(";")
    pos = 0

    def parse_node() -> Node:
        nonlocal pos
        node = Node()
        if pos < len(text) and text[pos] == "(":
            pos += 1
            while True:
                node.children.append(parse_node())
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                if pos < len(text) and text[pos] == ")":
                    pos += 1
                    break
                raise SimulationError(f"malformed Newick near position {pos}")
        match = re.match(r"[^,:()\[\]]*", text[pos:])
        label = match.group(0).strip().strip("'\"")
        pos += len(match.group(0))
        if label:
            node.name = label
        if pos < len(text) and text[pos] == ":":
            pos += 1
            match = re.match(r"[0-9eE+\-.]*", text[pos:])
            if not match.group(0):
                raise SimulationError(f"missing branch length near position {pos}")
            node.length = float(match.group(0))
            pos += len(match.group(0))
        return node

    root = parse_node()
    if pos != len(text):
        raise SimulationError(f"trailing characters in Newick: {text[pos:]!r}")
    return root


def to_newick(node: Node, with_lengths: bool = True) -> str:
    if node.children:
        inner = ",".join(to_newick(child, with_lengths) for child in node.children)
        text = f"({inner})"
    else:
        text = node.name or ""
    if with_lengths:
        text += f":{node.length:.12g}"
    elif node.name and node.children:
        text += node.name
    return text


def read_tree(path) -> Node:
    """Read a tree from a Newick file or the first tree of a NEXUS file."""
    text = Path(path).read_text()
    if text.lstrip().lower().startswith("#nexus"):
        match = re.search(r"\btree\s+\S+\s*=\s*(.+?);", text, re.S | re.I)
        if not match:
            raise SimulationError(f"{path}: no tree statement found")
        return parse_newick(match.group(1))
    return parse_newick(text)


# --------------------------------------------------------------------------
# trees
# --------------------------------------------------------------------------
def coalescent_tree(n_tips: int, height: float, rng: random.Random) -> Node:
    """Random ultrametric coalescent tree, rescaled to `height`.

    All tips are sampled at the present.  Each coalescent event advances every
    active lineage by the waiting time, so the resulting tree is ultrametric.
    """
    tips = [Node(name=f"t{i + 1}") for i in range(n_tips)]
    nodes = list(tips)
    while len(nodes) > 1:
        k = len(nodes)
        wait = rng.expovariate(k * (k - 1) / 2.0)
        for node in nodes:
            node.length += wait
        i, j = rng.sample(range(k), 2)
        a, b = nodes[i], nodes[j]
        parent = Node(children=[a, b])
        for idx in sorted((i, j), reverse=True):
            nodes.pop(idx)
        nodes.append(parent)
    root = nodes[0]
    root.length = 0.0
    total = tree_height(root)
    for node in iter_nodes(root):
        if node is not root:
            node.length *= height / total
    return root


def iter_nodes(root: Node):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def node_distances(root: Node) -> dict[int, float]:
    """Distance from the root to every node."""
    distances: dict[int, float] = {}
    stack = [(root, 0.0)]
    while stack:
        node, dist = stack.pop()
        distances[id(node)] = dist
        for child in node.children:
            stack.append((child, dist + child.length))
    return distances


def tree_height(root: Node) -> float:
    distances = node_distances(root)
    tips = [n for n in iter_nodes(root) if n.is_tip]
    return max(distances[id(tip)] for tip in tips)


def tip_nodes(root: Node) -> list[Node]:
    return [n for n in iter_nodes(root) if n.is_tip]


# --------------------------------------------------------------------------
# CTMC simulation
# --------------------------------------------------------------------------
def build_rate_matrix(states: list[str], model: str, rates: list[float]) -> list[list[float]]:
    n = len(states)
    q = [[0.0] * n for _ in range(n)]
    if model == "equal":
        if len(rates) != 1:
            raise SimulationError("model equal needs exactly one rate")
        for i in range(n):
            for j in range(n):
                if i != j:
                    q[i][j] = rates[0]
    else:
        if len(rates) != n * (n - 1):
            raise SimulationError(
                f"model asymmetric needs {n * (n - 1)} rates in row-major order, got {len(rates)}"
            )
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    q[i][j] = rates[k]
                    k += 1
    for i in range(n):
        q[i][i] = -sum(q[i][j] for j in range(n) if j != i)
    return q


def simulate_history(root: Node, states: list[str], q: list[list[float]],
                     root_state: int, rng: random.Random):
    """Simulate the CTMC down the tree.

    Returns ``(events, tip_states)`` where an event is
    ``(height_above_present, source_index, dest_index)``.
    """
    n = len(states)
    distances = node_distances(root)
    height = tree_height(root)
    events: list[tuple[float, int, int]] = []
    tip_states: dict[str, int] = {}

    def simulate_branch(parent: Node, child: Node, start: int) -> int:
        duration = child.length
        current = start
        elapsed = 0.0
        while True:
            rate = -q[current][current]
            if rate <= 0.0:
                break
            elapsed += rng.expovariate(rate)
            if elapsed >= duration:
                break
            weights = [q[current][j] if j != current else 0.0 for j in range(n)]
            total = sum(weights)
            if total <= 0.0:
                break
            draw = rng.random() * total
            cumulative = 0.0
            dest = current
            for j, weight in enumerate(weights):
                cumulative += weight
                if draw <= cumulative:
                    dest = j
                    break
            events.append((height - (distances[id(parent)] + elapsed), current, dest))
            current = dest
        return current

    def walk(node: Node, state: int) -> None:
        if node.is_tip:
            tip_states[node.name] = state
            return
        for child in node.children:
            walk(child, simulate_branch(node, child, state))

    walk(root, root_state)
    events.sort(key=lambda e: -e[0])
    return events, tip_states


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_nexus(root: Node, path, n_trees: int = 1, name: str = "SIMULATED") -> None:
    tips = [n.name for n in tip_nodes(root)]
    newick = to_newick(root)
    with open(path, "w") as fh:
        fh.write("#NEXUS\n")
        fh.write("Begin taxa;\n")
        fh.write(f"    Dimensions ntax={len(tips)};\n")
        fh.write("    Taxlabels\n")
        for tip in tips:
            fh.write(f"        {tip}\n")
        fh.write(";\n")
        fh.write("End;\n")
        fh.write("Begin trees;\n")
        for i in range(n_trees):
            fh.write(f"tree {name}_{i} = [&R] {newick};\n")
        fh.write("End;\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-prefix", required=True)
    tree = parser.add_mutually_exclusive_group()
    tree.add_argument("--input-tree", help="Newick or NEXUS tree to simulate on")
    tree.add_argument("--n-tips", type=int, default=20, help="tips of the generated coalescent tree")
    parser.add_argument("--tree-height", type=float, default=3.0, help="height of the generated tree")
    parser.add_argument("--n-trees", type=int, default=1, help="copies of the tree in the NEXUS file")
    parser.add_argument("--states", default="0,1", help="comma separated state codes")
    parser.add_argument("--model", choices=MODELS, default="asymmetric")
    parser.add_argument(
        "--rates",
        default="0.5,0.2",
        help="transition rates, row-major over ordered state pairs (or a single rate for --model equal)",
    )
    parser.add_argument("--root-state", help="state code of the root (default: drawn uniformly)")
    parser.add_argument(
        "--n-events",
        type=int,
        help="resample until exactly this many transitions occurred",
    )
    parser.add_argument(
        "--min-derived-fraction",
        type=float,
        default=0.0,
        help="resample until at least this fraction of tips differ from the root state",
    )
    parser.add_argument(
        "--max-derived-fraction",
        type=float,
        default=1.0,
        help="resample until at most this fraction of tips differ from the root state",
    )
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--mrsd", default="2020-01-01", help="most recent sampling date")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    states = [s.strip() for s in args.states.split(",")]
    if len(states) < 2:
        raise SimulationError("need at least two states")
    rates = [float(r) for r in args.rates.replace(",", " ").split()]
    q = build_rate_matrix(states, args.model, rates)
    mrsd = parse_date(args.mrsd)

    rng = random.Random(args.seed)
    if args.input_tree:
        root = read_tree(args.input_tree)
    else:
        root = coalescent_tree(args.n_tips, args.tree_height, rng)
    height = tree_height(root)
    tips = tip_nodes(root)
    distances = node_distances(root)
    if max(distances[id(t)] for t in tips) - min(distances[id(t)] for t in tips) > 1e-6:
        print("warning: the tree is not ultrametric", file=sys.stderr)

    root_state = states.index(args.root_state) if args.root_state else rng.randrange(len(states))

    events, tip_states, derived_fraction = [], {}, 0.0
    for attempt in range(args.max_attempts):
        events, tip_states = simulate_history(root, states, q, root_state, rng)
        if args.n_events is not None and len(events) != args.n_events:
            continue
        derived_fraction = sum(
            1 for tip in tips if tip_states[tip.name] != root_state
        ) / len(tips)
        if args.min_derived_fraction <= derived_fraction <= args.max_derived_fraction:
            break
    else:
        raise SimulationError(
            f"could not simulate a history with {args.n_events} events and a derived "
            f"fraction in [{args.min_derived_fraction:.2f}, {args.max_derived_fraction:.2f}] "
            f"in {args.max_attempts} attempts"
        )

    prefix = str(args.output_prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    write_nexus(root, f"{prefix}.trees", n_trees=args.n_trees)

    with open(f"{prefix}.traits.csv", "w") as fh:
        fh.write("taxon,state\n")
        for tip in tips:
            fh.write(f"{tip.name},{states[tip_states[tip.name]]}\n")

    with open(f"{prefix}.truth.tsv", "w") as fh:
        fh.write("event\theight\tyears_before_mrsd\tdate\tsource\tdest\n")
        for i, (event_height, source, dest) in enumerate(events, start=1):
            date = height_to_decimal_year(event_height, mrsd)
            fh.write(
                f"{i}\t{event_height:.6f}\t{event_height:.6f}\t{format_date(date)}\t"
                f"{states[source]}\t{states[dest]}\n"
            )

    summary = {
        "seed": args.seed,
        "states": states,
        "model": args.model,
        "rates": rates,
        "rate_matrix": q,
        "root_state": states[root_state],
        "tree_height": height,
        "n_tips": len(tips),
        "mrsd": mrsd.isoformat(),
        "n_events": len(events),
        "derived_fraction": derived_fraction,
        "n_events_by_direction": {
            f"{states[a]}->{states[b]}": sum(1 for _, s, d in events if s == a and d == b)
            for a in range(len(states))
            for b in range(len(states))
            if a != b and any(s == a and d == b for _, s, d in events)
        },
        "tip_state_counts": {
            s: sum(1 for tip in tips if states[tip_states[tip.name]] == s) for s in states
        },
    }
    with open(f"{prefix}.sim.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        f"simulated {len(events)} transition(s) on a {len(tips)}-tip tree of height {height:.4g}: "
        f"{summary['n_events_by_direction']}; tip states {summary['tip_state_counts']}"
    )
    for i, (event_height, source, dest) in enumerate(events, start=1):
        print(
            f"  event {i}: {states[source]}->{states[dest]} at height {event_height:.4f} "
            f"({format_date(height_to_decimal_year(event_height, mrsd))})"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SimulationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
