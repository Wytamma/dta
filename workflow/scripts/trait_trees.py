#!/usr/bin/env python3
"""BEAST ``logTree`` parsing and through-time trait summaries.

This module provides the machinery behind the prevalence-over-time and TMRCA
figures:

* an annotated-Newick parser that keeps BEAST's ``[&key=value]`` annotations and
  resolves the ``translate`` table of a ``logTree`` file;
* node heights measured from the most recent tip, matching the convention used
  by ``completeHistoryLogger`` and by ``summarize_jumps.py``;
* per-branch state trajectories reconstructed from the sampled transition
  history (the ``branch.history`` annotation written when complete-history
  logging is on);
* :func:`prevalence_counts`: how many reconstructed lineages carry each state
  at each point in time;
* :func:`tmrca_by_state`: the age of the most recent common ancestor of the
  tips in each state.

Everything is expressed in the time unit of the phylogeny (years), measured
from the most recent tip, so ``date = mrsd - height`` recovers calendar time.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: annotation written by the template when complete-history logging is enabled
HISTORY_TAG = "branch.history"

_EVENT_GROUP = re.compile(r"\{([^{}]*)\}")
_TOKEN = re.compile(
    r"(?P<open>\()|(?P<close>\))|(?P<comma>,)|(?P<colon>:)"
    r"|(?P<annot>\[[^\]]*\])"
    r"|(?P<quoted>'(?:[^']*)'|\"(?:[^\"]*)\")"
    r"|(?P<word>[^,():;\[\]\s]+)"
)


class TreeParseError(Exception):
    pass


@dataclass
class Node:
    name: str | None = None
    length: float = 0.0
    annotations: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = field(default=None, repr=False, compare=False)

    @property
    def is_tip(self) -> bool:
        return not self.children


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _add_annotation(annotations: dict[str, str], item: str) -> None:
    item = item.strip()
    if not item or "=" not in item:
        return
    key, value = item.split("=", 1)
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    annotations[key.strip()] = value


def parse_annotations(body: str) -> dict[str, str]:
    """Parse the inside of a ``[&...]`` group into a dict.

    Values may contain commas nested inside braces (``branch.history``), so the
    brace depth is tracked while splitting; everything else is handled by C
    string operations to keep this off the hot parsing path.
    """
    annotations: dict[str, str] = {}
    parts = body.split(",")
    depth = 0
    start = 0
    for index, part in enumerate(parts):
        depth += part.count("{") - part.count("}")
        if depth > 0:
            continue
        _add_annotation(annotations, ",".join(parts[start:index + 1]))
        start = index + 1
        depth = 0
    if start < len(parts):
        _add_annotation(annotations, ",".join(parts[start:]))
    return annotations


def parse_newick(text: str, translate: dict | None = None) -> Node:
    """Parse a Newick string, keeping BEAST ``[&...]`` annotations.

    Node annotations appear before the ``:``, branch annotations (including the
    sampled ``branch.history``) after it; both are merged into
    ``Node.annotations``.  A single regular expression tokenises the string, so
    the cost is dominated by ``re`` rather than by a Python character loop.
    """
    text = text.strip().rstrip(";")
    root: Node | None = None
    current: Node | None = None
    stack: list[Node] = []
    expect_length = False
    #: a label now starts a new child of the innermost group (after ``(``/``,``)
    expect_child = False
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup
        token = match.group()
        if kind == "open":
            node = Node()
            if stack:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            elif root is None:
                root = node
            else:
                raise TreeParseError("trailing characters in Newick")
            stack.append(node)
            current = node
            expect_length = False
            expect_child = True
        elif kind == "close":
            if not stack:
                raise TreeParseError("unbalanced parentheses in Newick")
            current = stack.pop()
            expect_length = False
            expect_child = False
        elif kind == "comma":
            if not stack:
                raise TreeParseError("comma outside a group in Newick")
            current = stack[-1]
            expect_length = False
            expect_child = True
        elif kind == "colon":
            expect_length = True
        elif kind == "word" or kind == "quoted":
            value = token[1:-1] if kind == "quoted" else token
            if expect_length:
                try:
                    current.length = float(value)  # type: ignore[union-attr]
                except ValueError as error:
                    raise TreeParseError(f"bad branch length {value!r}") from error
                expect_length = False
            else:
                if expect_child:
                    # a label after ``(``/``,`` names a leaf, not the group
                    node = Node()
                    if stack:
                        node.parent = stack[-1]
                        stack[-1].children.append(node)
                    elif root is None:
                        root = node
                    else:
                        raise TreeParseError("trailing characters in Newick")
                    current = node
                    expect_child = False
                if current is None:
                    if root is not None:
                        raise TreeParseError("trailing characters in Newick")
                    root = current = Node()
                current.name = translate.get(value, value) if translate else value
        elif kind == "annot":
            if current is not None:
                body = token[1:-1]
                if body.startswith("&"):
                    current.annotations.update(parse_annotations(body[1:]))
    if stack:
        raise TreeParseError("unbalanced parentheses in Newick")
    if root is None:
        raise TreeParseError("empty tree")
    return root


def parse_translate(text: str) -> dict[str, str]:
    """Parse the ``translate`` block of a NEXUS file, if present."""
    match = re.search(r"\btranslate\b(.*?);", text, re.S | re.I)
    if not match:
        return {}
    translate = {}
    for entry in match.group(1).split(","):
        entry = entry.strip()
        if not entry:
            continue
        bits = entry.split(None, 1)
        if len(bits) != 2:
            continue
        key, value = bits[0].strip(), bits[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        translate[key] = value
    return translate


_TREE_LINE = re.compile(r"^\s*tree\s+(\S+)\s*(?:\[&[^\]]*\])?\s*=\s*(.*?)\s*;\s*$")
_TREE_START = re.compile(r"^\s*tree\s")


def tree_count(path) -> int:
    """Number of ``tree`` statements in a ``logTree`` file.

    A plain line scan; the trees are not tokenised, so asking a multi-gigabyte
    log how many samples it holds costs a second rather than a parse per tree.
    """
    with Path(path).open() as handle:
        return sum(1 for line in handle if _TREE_START.match(line))


def read_trees(path, tag: str | None = None, indices=None):
    """Yield ``(name, root)`` for every tree in a BEAST ``logTree`` file.

    The file is streamed line by line, so a multi-gigabyte tree log never has to
    be held in memory; the ``translate`` table is read from the header before
    the first tree statement.  ``indices`` limits parsing to the given 0-based
    tree positions (the rest are recognised and skipped), which is how thinning
    avoids paying for samples that are discarded.
    """
    path = Path(path)
    translate: dict[str, str] = {}
    header: list[str] = []
    found = 0
    with path.open() as handle:
        for line in handle:
            # a 100 kB tree line is only worth the full regex when it is kept
            if _TREE_START.match(line) is None:
                if not found:
                    header.append(line)
                continue
            if not found:
                translate = parse_translate("".join(header))
            index = found
            found += 1
            if indices is not None and index not in indices:
                continue
            match = _TREE_LINE.match(line)
            if match is None:
                raise TreeParseError(f"{path}: malformed tree statement {index}")
            name, newick = match.groups()
            yield name, parse_newick(newick, translate=translate)
    if not found:
        raise TreeParseError(f"{path}: no tree statements found")


def tree_state_number(name: str) -> int | None:
    match = re.search(r"STATE_(\d+)", name)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# basic tree quantities
# --------------------------------------------------------------------------
def iter_nodes(root: Node):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def postorder(root: Node):
    """Yield every node after its children."""
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded or node.is_tip:
            yield node
        else:
            stack.append((node, True))
            for child in node.children:
                stack.append((child, False))


def node_heights(root: Node) -> dict[int, float]:
    """Height above the most recent tip for every node (tips are at 0)."""
    distances: dict[int, float] = {}
    stack = [(root, 0.0)]
    while stack:
        node, distance = stack.pop()
        distances[id(node)] = distance
        for child in node.children:
            stack.append((child, distance + child.length))
    tallest = max(distances[id(node)] for node in iter_nodes(root) if node.is_tip)
    return {key: tallest - value for key, value in distances.items()}


def root_height(root: Node) -> float:
    heights = node_heights(root)
    return heights[id(root)]


def tip_states(root: Node, tag: str) -> dict[str, str]:
    return {
        node.name: node.annotations.get(tag)
        for node in iter_nodes(root)
        if node.is_tip and node.annotations.get(tag) is not None
    }


def parse_history_events(text: str | None) -> list[tuple[float, str, str]]:
    """Parse a ``branch.history`` annotation into ``(height, source, dest)``.

    Events are returned oldest first (largest height first).  Both the
    3-element ``{height,source,dest}`` and the 4-element
    ``{site,height,source,dest}`` forms are accepted.
    """
    events = []
    for group in _EVENT_GROUP.findall(text or ""):
        parts = [part.strip() for part in group.split(",")]
        if len(parts) == 3:
            _, height, source, dest = "?", *parts
        elif len(parts) == 4:
            _, height, source, dest = parts
        else:
            continue
        try:
            events.append((float(height), source, dest))
        except ValueError:
            continue
    events.sort(key=lambda event: -event[0])
    return events


# --------------------------------------------------------------------------
# TMRCA
# --------------------------------------------------------------------------
def tmrca_by_state(root: Node, tag: str, states: list[str],
                    heights: dict[int, float] | None = None) -> dict[str, tuple[int, float | None]]:
    """Map every state to ``(n_tips, mrca_height)``.

    ``mrca_height`` is ``None`` when no tip carries the state.  All states are
    counted in one post-order traversal instead of one traversal per state.
    """
    heights = node_heights(root) if heights is None else heights
    index = {state: position for position, state in enumerate(states)}
    n_states = len(states)
    totals = [0] * n_states
    for node in iter_nodes(root):
        if node.is_tip:
            position = index.get(node.annotations.get(tag))
            if position is not None:
                totals[position] += 1
    best: list[float | None] = [None] * n_states
    rows: dict[int, list[int]] = {}
    for node in postorder(root):
        row = [0] * n_states
        if node.is_tip:
            position = index.get(node.annotations.get(tag))
            if position is not None:
                row[position] = 1
        else:
            for child in node.children:
                child_row = rows.pop(id(child))
                for position in range(n_states):
                    row[position] += child_row[position]
        height = heights[id(node)]
        for position, total in enumerate(totals):
            if total and row[position] == total:
                if best[position] is None or height < best[position]:
                    best[position] = height
        rows[id(node)] = row
    return {state: (totals[position], best[position]) for position, state in enumerate(states)}


def overall_tmrca(root: Node) -> float:
    """TMRCA of every tip in the tree (the root height)."""
    return root_height(root)


# --------------------------------------------------------------------------
# prevalence
# --------------------------------------------------------------------------
def history_annotation(node: Node, history_tag: str = HISTORY_TAG) -> str | None:
    """Return the sampled history annotation of a branch, tolerating a rename."""
    if history_tag in node.annotations:
        return node.annotations[history_tag]
    for key, value in node.annotations.items():
        if key.endswith(".history") or key == "history":
            return value
    return None


def branch_states(node: Node, states: list[str], tag: str, grid: np.ndarray,
                  heights: dict[int, float], history_tag: str = HISTORY_TAG):
    """State index of one branch (``node`` to its parent) at every grid height.

    Returns ``None`` when the branch carries no state annotation at its child
    end.  Below the youngest sampled transition the branch is in that
    transition's destination, between two transitions it is in the younger
    one's destination, and above the oldest transition it is in that
    transition's source (which is the parent's state).
    """
    index = {state: position for position, state in enumerate(states)}
    child_state = node.annotations.get(tag)
    if child_state not in index:
        return None
    events = parse_history_events(history_annotation(node, history_tag))
    if events:
        # ascending heights: the state below the youngest event is its
        # destination, above the oldest event it is that event's source
        ordered = sorted(events, key=lambda event: event[0])
        event_heights = np.array([event[0] for event in ordered])
        destinations = [index.get(event[2], index[child_state]) for event in ordered]
        oldest_source = index.get(ordered[-1][1], index[child_state])
        lookup = np.array(destinations + [oldest_source], dtype=int)
    else:
        event_heights = np.empty(0)
        lookup = np.array([index[child_state]], dtype=int)
    # index of the youngest event strictly older than each grid height
    positions = np.searchsorted(event_heights, grid, side="right")
    return lookup[positions]


def prevalence_counts(root: Node, states: list[str], tag: str, grid: np.ndarray,
                      history_tag: str = HISTORY_TAG,
                      heights: dict[int, float] | None = None) -> np.ndarray:
    """``(n_states, n_grid)`` numbers of reconstructed lineages per state.

    A lineage is a branch crossing the given height; the grid must be heights
    above the most recent tip.  Branches are half open, ``[child, parent)``.

    Each branch is a short sequence of constant-state segments; every segment
    contributes a unit step to a difference array, so the work per branch is a
    couple of ``bisect`` lookups instead of a full grid comparison.
    """
    heights = node_heights(root) if heights is None else heights
    grid = np.asarray(grid, dtype=float)
    n_grid = len(grid)
    order = np.argsort(grid, kind="stable")
    ascending = grid[order].tolist()
    index = {state: position for position, state in enumerate(states)}
    differences = [[0] * (n_grid + 1) for _ in states]
    for node in iter_nodes(root):
        if node.parent is None:
            continue
        child_state = node.annotations.get(tag)
        state_index = index.get(child_state)
        if state_index is None:
            continue
        lower = heights[id(node)]
        upper = heights[id(node.parent)]
        if upper <= lower:
            continue
        events = parse_history_events(history_annotation(node, history_tag))
        segments = []
        if events:
            ordered = sorted(events, key=lambda event: event[0])
            start = lower
            for height, _, destination in ordered:
                segments.append((start, height, index.get(destination, state_index)))
                start = height
            segments.append((start, upper, index.get(ordered[-1][1], state_index)))
        else:
            segments.append((lower, upper, state_index))
        for segment_start, segment_end, segment_state in segments:
            start = bisect_left(ascending, max(segment_start, lower))
            end = bisect_left(ascending, min(segment_end, upper))
            if start < end:
                row = differences[segment_state]
                row[start] += 1
                row[end] -= 1
    counts = np.cumsum(np.asarray(differences, dtype=float), axis=1)[:, :n_grid]
    unsorted = np.empty_like(counts)
    unsorted[:, order] = counts
    return unsorted


def natural_sort_key(value: str):
    return [(0, int(token)) if token.isdigit() else (1, token) for token in re.split(r"(\d+)", value)]
