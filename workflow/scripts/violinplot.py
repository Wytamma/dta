#!/usr/bin/env python3
"""Violin plots of the posterior distributions in a BEAST log.

Groups the log columns into families and writes one figure per family next to
the log file:

``<log>_<name>.rates.png``    transition rate parameters
``<log>_<name>.root.png``     root state frequencies
``<log>_In.Out.png``          transitions into and out of each state
``<log>_<pair>.png``          ordered pair transition counts

``--name`` is the BEAST id prefix (``location`` for the legacy geographic
analysis, the trait name otherwise) and ``--pair-prefix`` is the prefix of the
ordered pair registers (``Location`` for the legacy analysis).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _orientation() -> dict:
    """matplotlib renamed ``vert`` to ``orientation`` in 3.10."""
    parts = matplotlib.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return {"orientation": "horizontal"}
    if (major, minor) >= (3, 10):
        return {"orientation": "horizontal"}
    return {"vert": False}


def group_columns(columns, name: str, pair_prefix: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        if column.startswith(f"{name}.root"):
            groups[f"{name}.root"].append(column)
        elif column.startswith(f"{name}.rates"):
            groups[f"{name}.rates"].append(column)
        elif column.startswith("c_Into") or column.startswith("c_OutOf"):
            groups["In.Out"].append(column)
        elif column.startswith(f"c_{pair_prefix}.") and not column.startswith(
            f"c_{pair_prefix}.total"
        ):
            groups[pair_prefix].append(column)
    return groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_file")
    parser.add_argument("--name", default="location", help="BEAST id prefix in the log")
    parser.add_argument(
        "--pair-prefix", default="Location", help="prefix of the ordered pair registers"
    )
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.log_file, sep="\t", comment="#")
    groups = group_columns(list(frame.columns), args.name, args.pair_prefix)

    for group, columns in groups.items():
        melted = frame.melt(id_vars=["state"], value_vars=columns, var_name="parameter")
        figure, axis = plt.subplots(figsize=(6.5, max(2.0, 0.35 * len(columns) + 1.2)))
        order = melted.groupby("parameter")["value"].mean().sort_values(ascending=False).index
        data = [melted.loc[melted["parameter"] == parameter, "value"] for parameter in order]
        axis.violinplot(data, showextrema=False, **_orientation())
        axis.set_yticks(range(1, len(order) + 1))
        axis.set_yticklabels(order, fontsize=7)
        axis.set_xlabel("posterior value")
        axis.set_title(group)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(f"{args.log_file}_{group}.png", dpi=150)
        plt.close(figure)
        print(f"{args.log_file}_{group}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
