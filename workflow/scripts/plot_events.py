#!/usr/bin/env python3
"""Figures for reconstructed trait transitions (Stage 1, WP7).

Four figures, each driven by tables written by ``summarize_jumps.py`` or
``trait_through_time.py``:

``counts``        posterior number of transitions per direction (mean with 95% CI)
``dates``         posterior distribution of transition dates per direction
``root``          posterior root state probabilities
``through_time``  prevalence of each state over time plus one emergence-date
                  density per state (the tables come from
                  ``trait_through_time.py``)
``arrivals``      the same prevalence panel plus, per state, every sampled
                  transition *into* it — when the state appeared, which stays
                  informative when the carriers are paraphyletic and their MRCA
                  is the tree root in every sample

Run ``plot_events.py --help`` or ``plot_events.py <kind> --help``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT = "#2c7fb8"
MUTED = "#7f8c8d"


#: one colour per state, used consistently across both panels
PALETTE = [
    "#1b9e77",
    "#7570b3",
    "#d95f02",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#e7298a",
    "#666666",
]


def gaussian_kde(values, grid, bandwidth: float | None = None) -> np.ndarray:
    """Simple Gaussian KDE (Silverman's rule) so scipy is not required."""
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if values.size == 0:
        return np.zeros_like(grid)
    if bandwidth is None:
        span = (grid[-1] - grid[0]) / 40.0 if grid.size > 1 else 1.0
        if values.size > 1 and values.std(ddof=1) > 0:
            iqr = np.subtract(*np.percentile(values, [75, 25]))
            scale = iqr / 1.34 if iqr > 0 else values.std(ddof=1)
            bandwidth = 0.9 * scale * values.size ** (-0.2)
        else:
            bandwidth = span
    # a posterior with almost no spread would otherwise render as an invisible
    # spike; never go below 1% of the plotted range
    bandwidth = max(float(bandwidth), 0.01 * (grid[-1] - grid[0]), 1e-9)
    diff = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * diff**2).sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))


def states_by_mrca(summary: pd.DataFrame) -> list[str]:
    """Row order for the density panel: oldest median MRCA first.

    Ties (states whose MRCAs are the same node, or which were never sampled)
    keep the table's own order, so the figure is stable across runs.
    """
    order = list(summary["state"])
    if "date_median" not in summary.columns:
        return order

    def key(state: str):
        entry = summary.loc[summary["state"] == state, "date_median"]
        raw = None if entry.empty else entry.iloc[0]
        missing = raw is None or pd.isna(raw) or raw == ""
        value = "" if missing else str(raw)
        return (missing, value, order.index(state))

    return sorted(order, key=key)


def first_acquisition(acquisitions: pd.DataFrame | None, state: str):
    """``(median date, P(acquired), lo95, hi95)`` for ``state``.

    All four are ``None`` when the table is missing or the state was never
    entered in any sample.
    """
    if acquisitions is None or acquisitions.empty or "state" not in acquisitions.columns:
        return None, None, None, None
    entry = acquisitions[acquisitions["state"] == state]
    if entry.empty or pd.isna(entry["first_median"].iloc[0]):
        return None, None, None, None
    row = entry.iloc[0]
    probability = float(row["p_acquired"]) if "p_acquired" in entry.columns else None
    return (
        float(row["first_median"]),
        probability,
        float(row["first_lo95"]) if "first_lo95" in entry.columns else None,
        float(row["first_hi95"]) if "first_hi95" in entry.columns else None,
    )


def states_by_acquisition(acquisitions: pd.DataFrame | None, fallback: list[str]) -> list[str]:
    """Row order for the arrival panels: earliest median first acquisition first.

    States that are never entered (the root state of every sample) sort last;
    ties keep ``fallback``'s order so the figure is stable across runs.
    """
    order = list(fallback)

    def key(state: str):
        median, _, _, _ = first_acquisition(acquisitions, state)
        return (median is None, median if median is not None else 0.0, order.index(state))

    return sorted(order, key=key)


def plot_through_time(
    prevalence: pd.DataFrame,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    output,
    title: str | None = None,
    states: list[str] | None = None,
    acquisitions: pd.DataFrame | None = None,
) -> None:
    """Prevalence over time plus one emergence-date density per state.

    ``acquisitions`` is the ``acquisitions.tsv`` table: per state, the posterior
    of the first transition *into* it from any source, plus the probability that
    the state was entered at all.  That is the quantity to quote when dating an
    acquisition (the emergence density is the age of the crown of the carrier
    clade, which is younger).  Taking the minimum of the per-direction earliest
    transitions instead would be later than the truth and would ignore sources
    that were never sampled.
    """
    from dates import format_date

    # rows follow the timing (oldest first) unless the caller fixed the order
    state_list = states or states_by_mrca(summary)
    if not state_list:
        raise ValueError("no states to plot")
    figure = plt.figure(figsize=(9.5, 3.0 + 1.35 * len(state_list)))
    gridspec = figure.add_gridspec(
        1 + len(state_list), 1, height_ratios=[2.0] + [1.0] * len(state_list), hspace=0.45
    )
    colours = {state: PALETTE[i % len(PALETTE)] for i, state in enumerate(state_list)}

    axis = figure.add_subplot(gridspec[0])
    for state in state_list:
        block = prevalence[prevalence["state"] == state].sort_values("date")
        if block.empty:
            continue
        colour = colours[state]
        axis.plot(block["date"], 100 * block["proportion_mean"], color=colour, lw=2, label=state)
        axis.fill_between(
            block["date"],
            100 * block["proportion_lo95"],
            100 * block["proportion_hi95"],
            color=colour,
            alpha=0.18,
            linewidth=0,
        )
    for state in state_list:
        earliest, _, _, _ = first_acquisition(acquisitions, state)
        if earliest is None:
            continue
        axis.axvline(earliest, color=colours[state], lw=1.0, ls=":", alpha=0.8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("% of lineages")
    axis.set_title(title or "Trait prevalence over time")
    axis.legend(title="state", frameon=False, loc="upper left")
    axis.tick_params(labelbottom=False)
    axis.spines[["top", "right"]].set_visible(False)

    mrsd_year = (
        float((samples["date"] + samples["height"]).median()) if not samples.empty else None
    )
    low = float(prevalence["date"].min())
    high = float(prevalence["date"].max())
    if not samples.empty:
        low = min(low, float(samples["date"].min()))
        high = max(high, float(samples["date"].max()))
    if acquisitions is not None and not acquisitions.empty and "state" in acquisitions.columns:
        # the acquisition markers are annotations on the rows, so the shared
        # axis must reach them even when every MRCA sits at the tree root
        firsts = acquisitions.loc[acquisitions["state"].isin(state_list), "first_median"].dropna()
        if not firsts.empty:
            low = min(low, float(firsts.min()))
            high = max(high, float(firsts.max()))
    pad = 0.03 * (high - low)
    grid = np.linspace(low - pad, high + pad, 256)

    for row, state in enumerate(state_list):
        axis_row = figure.add_subplot(gridspec[1 + row], sharex=axis)
        values = samples.loc[samples["state"] == state, "date"].to_numpy(dtype=float)
        entry = summary[summary["state"] == state]
        appears = float(entry["appears_in"].iloc[0]) if not entry.empty else float("nan")
        at_root = float(entry["at_root"].iloc[0]) if "at_root" in entry.columns and not entry.empty else 0.0
        tree_height = (
            float(entry["tree_tmrca"].iloc[0])
            if "tree_tmrca" in entry.columns and not entry.empty
            else None
        )
        if values.size:
            density = gaussian_kde(values, grid)
            axis_row.fill_between(grid, 0, density, color=colours[state], alpha=0.28, linewidth=0)
            axis_row.plot(grid, density, color=colours[state], lw=1.2)
            median = float(np.median(values))
            axis_row.axvline(median, color="#37474f", lw=1.0, ls="--")
            earliest, probability, _, _ = first_acquisition(acquisitions, state)
            if earliest is not None:
                # the state's own colour; the row is labelled with the state and
                # the annotation names the acquisition, so no legend entry is
                # needed for a colour that changes from row to row
                axis_row.axvline(earliest, color=colours[state], lw=1.2, ls=":")
                label = f" acquisition {format_date(earliest)}"
                if probability is not None and probability < 0.995:
                    label += f" (P={probability:.2f})"
                axis_row.text(
                    # the label shares the line's colour, so keep it clear of it
                    earliest + 0.01 * (grid[-1] - grid[0]),
                    0.05 * axis_row.get_ylim()[1],
                    label,
                    color=colours[state],
                    fontsize=7.5,
                    va="bottom",
                )
        else:
            axis_row.text(
                0.5, 0.5, "state never sampled", ha="center", va="center",
                transform=axis_row.transAxes, fontsize=8, color=MUTED,
            )
        if tree_height is not None and mrsd_year is not None:
            tree_date = mrsd_year - tree_height
            axis_row.axvline(tree_date, color="#9e9e9e", lw=0.9, ls=(0, (2, 2)))
            if row == 0:
                from matplotlib.lines import Line2D

                axis_row.legend(
                    handles=[
                        Line2D([], [], color="#9e9e9e", lw=0.9, ls=(0, (2, 2)),
                               label="tree TMRCA"),
                        Line2D([], [], color="#37474f", lw=1.0, ls="--",
                               label="median MRCA of the carriers"),
                    ],
                    loc="upper right",
                    frameon=False,
                    fontsize=7,
                    handlelength=1.6,
                )
        note = f"appears in {appears:.0%} of trees"
        if values.size:
            note += f"  ·  median MRCA {format_date(float(np.median(values)))}"
        if at_root >= 0.5:
            note += "  ·  MRCA is the tree root: the carriers are not a clade"
        axis_row.set_title(note, loc="right", fontsize=7.5, color=MUTED, pad=2)
        axis_row.set_yticks([])
        axis_row.set_ylabel(state, rotation=0, ha="right", va="center", fontsize=9)
        axis_row.spines[["top", "right", "left"]].set_visible(False)
        axis_row.tick_params(labelbottom=row == len(state_list) - 1)

    axis.set_xlim(low - pad, high + pad)
    axis_row.set_xlabel("date")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_arrivals(
    prevalence: pd.DataFrame,
    events: pd.DataFrame,
    acquisitions: pd.DataFrame | None,
    output,
    title: str | None = None,
    states: list[str] | None = None,
    summary: pd.DataFrame | None = None,
) -> None:
    """Prevalence plus, per state, *when it was entered*.

    The lower panels histogram every sampled transition into the state (all
    samples, every source) instead of the age of the MRCA of the tips carrying
    it.  The two answer different questions, and for a reversible trait that has
    been gained and lost many times the MRCA panel cannot answer its own: the
    carrier tips are then paraphyletic, their MRCA is the tree root in every
    sample, and its density collapses to a spike at the root.  Arrival times
    stay informative however mixed the carriers are.

    The orange-free coding is the same as ``plot_through_time``: the state's own
    colour, a dotted line at the posterior median of the *first* arrival with
    its 95% interval shaded, annotated with ``P(acquired)`` and the number of
    sampled arrivals.
    """
    from dates import format_date

    if "date" not in events.columns or events.empty:
        raise ValueError("arrival dates need --mrsd when summarising; re-run summarize_jumps.py")

    if states:
        state_list = states
    else:
        fallback = (
            states_by_mrca(summary)
            if summary is not None and not summary.empty
            else sorted(prevalence["state"].unique())
        )
        state_list = states_by_acquisition(acquisitions, fallback)
    if not state_list:
        raise ValueError("no states to plot")

    arrival_dates = {
        state: events.loc[events["dest"] == state, "date"].to_numpy(dtype=float)
        for state in state_list
    }

    # one calendar window for every panel, so a date lines up down the figure
    spans = [
        float(prevalence["date"].min()),
        float(prevalence["date"].max()),
        float(events["date"].min()),
        float(events["date"].max()),
    ]
    if acquisitions is not None and not acquisitions.empty:
        for column in ("first_lo95", "first_hi95"):
            if column in acquisitions.columns:
                values = acquisitions[column].dropna()
                if not values.empty:
                    spans += [float(values.min()), float(values.max())]
    spans = [value for value in spans if np.isfinite(value)]
    if not spans:
        raise ValueError("no dated prevalence or arrival rows to plot")
    low, high = min(spans), max(spans)
    if high <= low:
        high = low + 1.0
    pad = 0.03 * (high - low)
    edges = np.linspace(low - pad, high + pad, 41)

    figure = plt.figure(figsize=(9.5, 3.0 + 1.25 * len(state_list)))
    gridspec = figure.add_gridspec(
        1 + len(state_list), 1, height_ratios=[2.0] + [1.0] * len(state_list), hspace=0.45
    )
    colours = {state: PALETTE[i % len(PALETTE)] for i, state in enumerate(state_list)}

    axis = figure.add_subplot(gridspec[0])
    for state in state_list:
        block = prevalence[prevalence["state"] == state].sort_values("date")
        if block.empty:
            continue
        axis.plot(block["date"], 100 * block["proportion_mean"], color=colours[state], lw=2, label=state)
        axis.fill_between(
            block["date"],
            100 * block["proportion_lo95"],
            100 * block["proportion_hi95"],
            color=colours[state],
            alpha=0.18,
            linewidth=0,
        )
    for state in state_list:
        median, _, _, _ = first_acquisition(acquisitions, state)
        if median is not None:
            axis.axvline(median, color=colours[state], lw=1.0, ls=":", alpha=0.8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("% of lineages")
    axis.set_title(title or "Trait prevalence over time")
    axis.legend(title="state", frameon=False, loc="upper left")
    axis.tick_params(labelbottom=False)
    axis.spines[["top", "right"]].set_visible(False)

    for row, state in enumerate(state_list):
        axis_row = figure.add_subplot(gridspec[1 + row], sharex=axis)
        values = arrival_dates[state]
        axis_row.spines[["top", "right", "left"]].set_visible(False)
        if values.size:
            axis_row.hist(
                values, bins=edges, color=colours[state], alpha=0.75, linewidth=0,
                histtype="stepfilled",
            )
            median, probability, lo95, hi95 = first_acquisition(acquisitions, state)
            if median is not None:
                axis_row.axvline(median, color=colours[state], lw=1.2, ls=":")
                upper = axis_row.get_ylim()[1]
                if lo95 is not None and hi95 is not None:
                    # a whisker at the foot of the panel: a full-height band
                    # would compete with the histogram it annotates
                    level = 0.10 * upper
                    axis_row.plot(
                        [lo95, hi95], [level, level], color=colours[state], lw=2.0,
                        solid_capstyle="butt", alpha=0.85,
                    )
                    for edge in (lo95, hi95):
                        axis_row.plot(
                            [edge, edge], [level - 0.05 * upper, level + 0.05 * upper],
                            color=colours[state], lw=1.0, alpha=0.85,
                        )
                label = f"first {format_date(median)}"
                if probability is not None and probability < 0.995:
                    label += f" (P={probability:.2f})"
                axis_row.text(
                    median + 0.025 * (high - low),
                    0.88 * upper,
                    label,
                    color=colours[state],
                    fontsize=7.5,
                    va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                )
            note = f"{int(values.size)} sampled arrivals"
            axis_row.set_title(note, loc="right", fontsize=7.5, color=MUTED, pad=2)
        else:
            axis_row.text(
                0.5, 0.5, "state never entered", ha="center", va="center",
                transform=axis_row.transAxes, fontsize=8, color=MUTED,
            )
        axis_row.set_yticks([])
        axis_row.set_ylabel(state, rotation=0, ha="right", va="center", fontsize=9)
        axis_row.tick_params(labelbottom=row == len(state_list) - 1)

    axis.set_xlim(low - pad, high + pad)
    axis_row.set_xlabel("date")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def read_table(path) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"table not found: {path}")
    return pd.read_csv(path, sep="\t")


def finish(fig, output) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_counts(table: pd.DataFrame, output, title: str | None = None) -> None:
    if table.empty:
        raise ValueError("no transition directions to plot")
    table = table.sort_values("mean")
    figure, axes = plt.subplots(figsize=(6.5, 0.55 * len(table) + 1.8))
    y = np.arange(len(table))
    means = table["mean"].to_numpy()
    lower = means - table["lo95"].to_numpy()
    upper = table["hi95"].to_numpy() - means
    axes.barh(y, means, color=ACCENT, alpha=0.85, height=0.6)
    axes.errorbar(means, y, xerr=[lower, upper], fmt="none", ecolor="#22313f", capsize=3, lw=1.2)
    for position, row in enumerate(table.itertuples()):
        axes.text(
            row.hi95 + 0.05 + 0.02 * max(1.0, table["hi95"].max()),
            position,
            f"P(>0)={row.p_positive:.2f}",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    axes.set_yticks(y)
    axes.set_yticklabels(table["direction"])
    axes.set_xlabel("number of transitions (posterior mean, 95% CI)")
    axes.set_title(title or "Reconstructed transitions")
    axes.set_xlim(left=0)
    axes.spines[["top", "right"]].set_visible(False)
    finish(figure, output)


def plot_dates(dates: pd.DataFrame, events: pd.DataFrame, output, title: str | None = None) -> None:
    if dates.empty:
        raise ValueError("no transition directions to plot")
    if "date" not in events.columns:
        raise ValueError("event dates need --mrsd when summarising; re-run summarize_jumps.py")
    directions = list(dates["direction"])
    figure, axes = plt.subplots(
        len(directions), 1, figsize=(7, 2.1 * len(directions) + 0.8), squeeze=False, sharex=True
    )
    for index, (axis, direction) in enumerate(zip(axes[:, 0], directions)):
        row = dates[dates["direction"] == direction].iloc[0]
        values = events.loc[events["direction"] == direction, "date"].to_numpy()
        if values.size:
            bins = max(8, min(30, values.size // 4 + 4))
            axis.hist(values, bins=bins, color=ACCENT, alpha=0.75, label="transition times")
        else:
            axis.text(0.5, 0.5, "no transition sampled", ha="center", va="center", transform=axis.transAxes)
        if not np.isnan(row["first_median"]):
            axis.axvspan(row["first_lo95"], row["first_hi95"], color="#f39c12", alpha=0.18)
            axis.axvline(row["first_median"], color="#d35400", lw=1.6)
            axis.text(
                row["first_median"],
                axis.get_ylim()[1] * 0.92,
                f" earliest {row['first_date']}",
                color="#d35400",
                fontsize=8,
                va="top",
            )
        axis.set_ylabel(direction, rotation=0, ha="right", va="center", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelbottom=index == len(directions) - 1)
    axes[-1, 0].set_xlabel("calendar year")
    # one calendar window for the whole figure, so the panels line up vertically
    plotted = events.loc[events["direction"].isin(directions), "date"].to_numpy(dtype=float)
    edges = np.concatenate(
        [plotted, dates[["first_lo95", "first_hi95"]].to_numpy(dtype=float).ravel()]
    )
    edges = edges[np.isfinite(edges)]
    if edges.size:
        low, high = float(edges.min()), float(edges.max())
        pad = 0.02 * (high - low) if high > low else 1.0
        axes[0, 0].set_xlim(low - pad, high + pad)
    axes[0, 0].set_title(title or "Posterior transition dates (earliest event in orange)")
    finish(figure, output)


def plot_root(table: pd.DataFrame, output, title: str | None = None) -> None:
    if table.empty:
        raise ValueError("no root states to plot")
    label_column = "label" if "label" in table.columns else "state"
    figure, axis = plt.subplots(figsize=(0.9 * len(table) + 2.2, 3.4))
    axis.bar(
        [str(v) for v in table[label_column]],
        table["probability"],
        color=ACCENT,
        alpha=0.85,
    )
    for position, row in enumerate(table.itertuples()):
        axis.text(position, row.probability + 0.02, f"{row.probability:.2f}", ha="center", fontsize=8)
    axis.set_ylabel("posterior probability")
    axis.set_xlabel("root state")
    axis.set_ylim(0, 1.05)
    axis.set_title(title or "Root state")
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="kind", required=True)

    counts = subparsers.add_parser("counts", help="transition counts per direction")
    counts.add_argument("--counts", required=True)
    counts.add_argument("--output", required=True)
    counts.add_argument("--title")

    dates = subparsers.add_parser("dates", help="transition date distributions")
    dates.add_argument("--dates", required=True)
    dates.add_argument("--events", required=True)
    dates.add_argument("--output", required=True)
    dates.add_argument("--title")

    root = subparsers.add_parser("root", help="root state probabilities")
    root.add_argument("--root-states", required=True)
    root.add_argument("--output", required=True)
    root.add_argument("--title")

    through = subparsers.add_parser(
        "through_time", help="prevalence over time plus emergence-date densities"
    )
    through.add_argument("--prevalence", required=True)
    through.add_argument("--tmrca-samples", required=True)
    through.add_argument("--tmrca", required=True)
    through.add_argument(
        "--acquisitions",
        help="acquisitions.tsv (first transition into each state), marked on the rows",
    )
    through.add_argument("--output", required=True)
    through.add_argument("--title")
    through.add_argument("--states", help="comma separated subset of states to plot")

    arrivals = subparsers.add_parser(
        "arrivals",
        help="prevalence plus when each state was entered (every sampled transition into it)",
    )
    arrivals.add_argument("--prevalence", required=True)
    arrivals.add_argument("--events", required=True, help="events.tsv (needs --mrsd)")
    arrivals.add_argument(
        "--acquisitions",
        help="acquisitions.tsv (first transition into each state, with its 95% interval)",
    )
    arrivals.add_argument("--tmrca", help="tmrca.tsv, only used to order the rows")
    arrivals.add_argument("--output", required=True)
    arrivals.add_argument("--title")
    arrivals.add_argument("--states", help="comma separated subset of states to plot")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    states = (
        [state.strip() for state in args.states.split(",") if state.strip()]
        if getattr(args, "states", None)
        else None
    )
    if args.kind == "counts":
        plot_counts(read_table(args.counts), args.output, args.title)
    elif args.kind == "dates":
        plot_dates(read_table(args.dates), read_table(args.events), args.output, args.title)
    elif args.kind == "root":
        plot_root(read_table(args.root_states), args.output, args.title)
    elif args.kind == "arrivals":
        plot_arrivals(
            read_table(args.prevalence),
            read_table(args.events),
            read_table(args.acquisitions) if args.acquisitions else None,
            args.output,
            args.title,
            states,
            read_table(args.tmrca) if args.tmrca else None,
        )
    else:
        plot_through_time(
            read_table(args.prevalence),
            read_table(args.tmrca_samples),
            read_table(args.tmrca),
            args.output,
            args.title,
            states,
            read_table(args.acquisitions) if args.acquisitions else None,
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
