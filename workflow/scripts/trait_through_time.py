#!/usr/bin/env python3
"""Prevalence over time and TMRCA for a discrete trait (Stage 1, WP7).

Reads the ``logTree`` file of one or more BEAST chains and writes

``<prefix>.prevalence.tsv``
    share of reconstructed lineages carrying each state through time
    (``date, state, proportion_mean, proportion_lo95, proportion_hi95,
    n_lineages_mean, n_samples``)
``<prefix>.tmrca.tsv``
    per state: how often the state appears in the reconstruction, how many tips
    carry it, and the posterior of the age of the most recent common ancestor of
    those tips (the *emergence date* of the state)
``<prefix>.tmrca_samples.tsv``
    the per-sample values behind the TMRCA table, for density plots

Definitions
-----------
*Lineage prevalence at time t* is the fraction of branches crossing ``t`` whose
sampled state at ``t`` is ``s``, i.e. the share of circulating lineages carrying
the trait.  The sampled state of a branch comes from the per-branch transition
history that BEAST writes when complete-history logging is on
(``branch.history``); without it the state of the child node is used for the
whole branch and a warning is printed.

The *emergence date* of a state is the age of the most recent common ancestor of
the tips carrying it, ``date = mrsd - height``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dates import decimal_year, format_date, parse_date
from trait_trees import (
    HISTORY_TAG,
    TreeParseError,
    history_annotation,
    iter_nodes,
    node_heights,
    parse_history_events,
    prevalence_counts,
    read_trees,
    tip_states,
    tmrca_by_state,
    tree_count,
    tree_state_number,
)
from trait_data import natural_sort_key


def state_appears(root, tag: str, states: list[str]) -> set[str]:
    """States that occur anywhere in the sampled reconstruction."""
    present = set(tip_states(root, tag).values())
    for node in iter_nodes(root):
        for _, source, dest in parse_history_events(history_annotation(node)):
            present.add(source)
            present.add(dest)
    return present & set(states)


def collect_trees(path, burnin_fraction: float, max_trees: int):
    """Read the trees, drop burn-in and thin evenly to at most ``max_trees``.

    The log is scanned once to count its samples and only the kept positions are
    parsed, so a 10k-sample log costs the same as a 200-sample one instead of
    holding every tree in memory and throwing 98% away.
    """
    total = tree_count(path)
    if not total:
        raise TreeParseError(f"{path}: no trees")
    burnin = int(total * burnin_fraction)
    kept = max(total - burnin, 1)
    burnin = total - kept
    if max_trees and kept > max_trees:
        positions = burnin + np.linspace(0, kept - 1, max_trees).round().astype(int)
        wanted = set(dict.fromkeys(positions.tolist()))
    else:
        wanted = set(range(burnin, total))
    return [
        (name, root, tree_state_number(name)) for name, root in read_trees(path, indices=wanted)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trees", nargs="+", required=True, help="BEAST logTree file(s)")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tag", required=True, help="state annotation name")
    parser.add_argument(
        "--trait-json",
        help="trait report from trait_data.py; used to report the user's state labels",
    )
    parser.add_argument("--history-tag", default=HISTORY_TAG)
    parser.add_argument("--states", help="comma separated state codes (default: from the tips)")
    parser.add_argument("--mrsd", required=True, help="most recent sampling date, YYYY-MM-DD")
    parser.add_argument("--burnin-fraction", type=float, default=0.1)
    parser.add_argument("--max-trees", type=int, default=200, help="thin to at most this many trees")
    parser.add_argument("--n-grid", type=int, default=100, help="time points for the prevalence curve")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # BEAST-safe state codes (s0, s1, ...) are mapped back to the user's labels
    labels: dict[str, str] = {}
    if args.trait_json and Path(args.trait_json).exists():
        report = json.loads(Path(args.trait_json).read_text())
        labels = {beast: user for user, beast in report.get("state_code_map", {}).items()}

    def label(state: str) -> str:
        return labels.get(state, state)

    mrsd = parse_date(args.mrsd)
    mrsd_year = decimal_year(mrsd)

    samples = []
    for path in args.trees:
        samples.extend(collect_trees(path, args.burnin_fraction, args.max_trees))
    if not samples:
        raise TreeParseError("no trees left after burn-in")

    if args.states:
        states = [state.strip() for state in args.states.split(",") if state.strip()]
    else:
        observed = set().union(*(set(tip_states(root, args.tag).values()) for _, root, _ in samples))
        states = sorted(observed, key=natural_sort_key)

    has_history = any(
        history_annotation(node, args.history_tag) is not None
        for _, root, _ in samples
        for node in iter_nodes(root)
    )
    if not has_history:
        print(
            "warning: the tree log carries no per-branch histories; prevalence is "
            "approximated from the reconstructed node states. Re-run BEAST with "
            "complete-history logging enabled for exact curves.",
            file=sys.stderr,
        )

    heights = [node_heights(root) for _, root, _ in samples]
    root_heights = [heights[i][id(root)] for i, (_, root, _) in enumerate(samples)]
    tallest = max(root_heights)
    # Below the oldest sampled tip the lineage pool shrinks towards the last few
    # genomes, which reflects the sampling schedule rather than the trait, so the
    # curve stops at the earliest sampling date.  The small epsilon keeps the grid
    # strictly inside the tree when all tips are sampled at the same time (tips
    # sit at height 0 in exact arithmetic but at ~1e-11 in the written tree).
    oldest_tip = max(
        heights[i][id(tip)]
        for i, (_, root, _) in enumerate(samples)
        for tip in iter_nodes(root)
        if tip.is_tip
    )
    bottom = max(oldest_tip, tallest * 1e-6)
    top = max(tallest * (1.0 - 1e-6), bottom)
    grid = np.linspace(top, bottom, args.n_grid)
    dates = mrsd_year - grid
    tree_tmrca = float(np.median(root_heights))

    proportions = np.full((len(samples), len(states), args.n_grid), np.nan)
    lineages = np.zeros((len(samples), args.n_grid))
    for index, (_, root, _) in enumerate(samples):
        counts = prevalence_counts(
            root, states, args.tag, grid, args.history_tag, heights=heights[index]
        )
        totals = counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            proportions[index] = np.where(totals > 0, counts / totals, np.nan)
        lineages[index] = totals

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(proportions, axis=0)
        lo = np.nanpercentile(proportions, 2.5, axis=0)
        hi = np.nanpercentile(proportions, 97.5, axis=0)
    contributing = np.sum(~np.isnan(proportions[:, 0, :]), axis=0)
    mean_lineages = np.nanmean(np.where(lineages > 0, lineages, np.nan), axis=0)

    tmrca_rows = []
    appears_counts = {state: 0 for state in states}
    at_root_counts = {state: 0 for state in states}
    tip_counts = {state: [] for state in states}
    for index, (_, root, state_number) in enumerate(samples):
        present = state_appears(root, args.tag, states)
        for state in states:
            appears_counts[state] += int(state in present)
        for state, (n_tips, height) in tmrca_by_state(
            root, args.tag, states, heights=heights[index]
        ).items():
            tip_counts[state].append(n_tips)
            if height is None:
                continue
            # a scattered state's MRCA is the tree root, i.e. not an emergence
            if abs(height - root_heights[index]) < 1e-9:
                at_root_counts[state] += 1
            tmrca_rows.append(
                {
                    "sample": state_number if state_number is not None else index,
                    "state": state,
                    "n_tips": n_tips,
                    "height": height,
                    "date": mrsd_year - height,
                }
            )

    prefix = str(args.output_prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}.prevalence.tsv", "w") as handle:
        handle.write("date\tstate\tproportion_mean\tproportion_lo95\tproportion_hi95\tn_lineages_mean\tn_samples\n")
        for point in range(args.n_grid):
            for state_index, state in enumerate(states):
                handle.write(
                    f"{dates[point]:.6f}\t{label(state)}\t{mean[state_index, point]:.6f}\t"
                    f"{lo[state_index, point]:.6f}\t{hi[state_index, point]:.6f}\t"
                    f"{mean_lineages[point]:.3f}\t{int(contributing[point])}\n"
                )

    with open(f"{prefix}.tmrca_samples.tsv", "w") as handle:
        handle.write("sample\tstate\tn_tips\theight\tdate\n")
        for row in tmrca_rows:
            handle.write(
                f"{row['sample']}\t{label(row['state'])}\t{row['n_tips']}\t"
                f"{row['height']:.6f}\t{row['date']:.6f}\n"
            )

    heights_by_state = {state: [] for state in states}
    for row in tmrca_rows:
        heights_by_state[row["state"]].append(row["height"])

    with open(f"{prefix}.tmrca.tsv", "w") as handle:
        handle.write(
            "state\tn_tips\tappears_in\tat_root\ttmrca_mean\ttmrca_median\ttmrca_lo95\t"
            "tmrca_hi95\tdate_median\tdate_lo95\tdate_hi95\ttree_tmrca\n"
        )
        for state in states:
            values = np.asarray(heights_by_state[state], dtype=float)
            n_tips = np.asarray(tip_counts[state], dtype=float)
            appears = appears_counts[state] / len(samples)
            at_root = at_root_counts[state] / len(samples)
            if values.size == 0:
                handle.write(
                    f"{label(state)}\t0\t{appears:.4f}\t{at_root:.4f}\t\t\t\t\t\t\t\t"
                    f"{tree_tmrca:.6f}\n"
                )
                continue
            lo95, median, hi95 = np.percentile(values, [2.5, 50, 97.5])
            handle.write(
                f"{label(state)}\t{n_tips.mean():.3f}\t{appears:.4f}\t{at_root:.4f}\t"
                f"{values.mean():.6f}\t{median:.6f}\t{lo95:.6f}\t{hi95:.6f}\t"
                f"{format_date(mrsd_year - float(median))}\t"
                f"{format_date(mrsd_year - float(lo95))}\t"
                f"{format_date(mrsd_year - float(hi95))}\t{tree_tmrca:.6f}\n"
            )

    print(
        f"{len(samples)} trees, {len(states)} states, {args.n_grid} time points "
        f"({dates[-1]:.3f} to {dates[0]:.3f}), tree TMRCA {tallest:.3f}"
    )
    for state in states:
        values = np.asarray(heights_by_state[state], dtype=float)
        median = float(np.median(values)) if values.size else float("nan")
        at_root = at_root_counts[state] / len(samples)
        print(
            f"  state {label(state)}: appears in {appears_counts[state] / len(samples):.1%} of trees, "
            f"{np.mean(tip_counts[state]):.1f} tips, MRCA "
            f"{format_date(mrsd_year - median) if values.size else 'n/a'}"
            + (f" (= tree root in {at_root:.0%} of trees: carriers are not a clade)"
               if at_root >= 0.5 else "")
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (TreeParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
