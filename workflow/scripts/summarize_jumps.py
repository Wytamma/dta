#!/usr/bin/env python3
"""Summarise posterior transition histories into events, counts and dates.

Stage 1, WP4.  Consumes the outputs of one or more BEAST chains:

``<prefix>.jumpHistory.log``  complete transition histories (``completeHistoryLogger``),
                              one file per chain (``--jump-history`` is repeatable)
``<prefix>.trees``            ancestral states per node (``logTree``), for root states
``<prefix>.log``              Markov jump count registers, used as a cross-check
``<prefix>.traits.json``      trait report from ``trait_data.py`` (state code map)

and writes

``<prefix>.events.tsv``       every transition of every retained sample
``<prefix>.event_counts.tsv`` posterior counts per transition direction
``<prefix>.event_dates.tsv``  posterior transition dates per direction
``<prefix>.root_states.tsv``  posterior root state probabilities
``<prefix>.count_check.tsv``  complete histories vs Markov jump registers
``<prefix>.summary.json``     machine readable digest of all of the above
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from dates import parse_date
from jump_history import (
    JumpHistoryError,
    acquisition_summary,
    date_summary,
    direction_counts,
    events_frame,
    expected_directions,
    parse_jump_history,
    read_count_registers,
    read_root_states,
)


def load_trait_report(path) -> dict:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise JumpHistoryError(f"trait report not found: {path}")
    return json.loads(path.read_text())


def cross_check_registers(log_paths, prefix, counts, code_map, burnins):
    """Compare history-derived counts with BEAST's Markov jump count registers.

    Both are estimates of the same quantity from the same samples, so their
    posterior means must agree to within Monte Carlo error.  ``burnins`` is one
    burn-in per log (``None`` skips the filter), because each chain numbers its
    states from zero and must lose its own prefix.
    """
    if not log_paths:
        return None, None
    if isinstance(burnins, (int, float, type(None))):
        burnins = [burnins] * len(log_paths)
    frames = [
        read_count_registers(path, prefix, burnin)
        for path, burnin in zip(log_paths, burnins)
    ]
    frame = pd.concat(frames, ignore_index=True)
    rows = []
    for _, row in counts.iterrows():
        beast_source = code_map.get(row["source"], row["source"])
        beast_dest = code_map.get(row["dest"], row["dest"])
        column = f"c_{prefix}.{beast_source}.{beast_dest}[1]"
        if column not in frame.columns:
            continue
        rows.append(
            {
                "direction": row["direction"],
                "history_mean": float(row["mean"]),
                "register_mean": float(frame[column].mean()),
                "difference": float(row["mean"]) - float(frame[column].mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return None, None
    return table, float(table["difference"].abs().max())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--jump-history", nargs="+", required=True,
        help="completeHistoryLogger log(s), one per chain",
    )
    parser.add_argument("--trees", nargs="+", help="BEAST logTree file(s) (for root states)")
    parser.add_argument("--log", nargs="+", help="BEAST log file(s) (jump count registers, cross-check)")
    parser.add_argument("--trait-json", help="trait report from trait_data.py")
    parser.add_argument("--tag", help="tree annotation tag (default: the trait name)")
    parser.add_argument("--prefix-id", help="BEAST id prefix used for the c_ registers")
    parser.add_argument("--mrsd", help="most recent sampling date, YYYY-MM-DD")
    parser.add_argument("--burnin-fraction", type=float, default=0.1)
    parser.add_argument("--burnin-states", type=int, help="absolute burn-in in MCMC states")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = load_trait_report(args.trait_json)
    code_map = report.get("state_code_map", {})
    trait = report.get("trait", "trait")
    tag = args.tag or report.get("trait", args.prefix_id or "location")
    id_prefix = args.prefix_id or tag
    mrsd = parse_date(args.mrsd) if args.mrsd else None

    samples: list = []
    infos: list[dict] = []
    for path in args.jump_history:
        chain_samples, chain_info = parse_jump_history(
            path,
            burnin_states=args.burnin_states,
            burnin_fraction=args.burnin_fraction,
        )
        chain = len(infos) + 1
        for sample in chain_samples:
            sample.chain = chain
        samples.extend(chain_samples)
        infos.append(chain_info)
    info = {
        "files": [entry["file"] for entry in infos],
        "n_chains": len(infos),
        "n_samples_total": sum(entry["n_samples_total"] for entry in infos),
        "n_samples_kept": sum(entry["n_samples_kept"] for entry in infos),
        "burnin_states": min(entry["burnin_states"] for entry in infos),
        "n_events_total": sum(entry["n_events_total"] for entry in infos),
        "samples_with_count_mismatch": [
            state for entry in infos for state in entry["samples_with_count_mismatch"]
        ][:10],
    }
    states = [str(state) for state in report.get("states", [])]
    # every ordered pair, so a direction that never occurred is reported as
    # P(>0) = 0 instead of being missing from the table
    directions = expected_directions(states) if states else None

    events = events_frame(samples, mrsd=mrsd, code_map=code_map)
    counts = direction_counts(samples, directions=directions, code_map=code_map)
    dates = (
        date_summary(samples, mrsd=mrsd, code_map=code_map, directions=directions)
        if mrsd
        else None
    )
    acquisitions = acquisition_summary(
        samples, mrsd=mrsd, code_map=code_map, states=states or None
    )

    prefix = str(args.output_prefix)
    events.to_csv(f"{prefix}.events.tsv", sep="\t", index=False)
    counts.to_csv(f"{prefix}.event_counts.tsv", sep="\t", index=False)
    acquisitions.to_csv(f"{prefix}.acquisitions.tsv", sep="\t", index=False)

    digest = {
        "trait": trait,
        "states": report.get("states", []),
        "mrsd": mrsd.isoformat() if mrsd else None,
        "jump_history": info,
        "directions": counts.to_dict(orient="records"),
        "acquisitions": acquisitions.to_dict(orient="records"),
    }

    if dates is not None:
        dates.to_csv(f"{prefix}.event_dates.tsv", sep="\t", index=False)
        digest["dates"] = dates.to_dict(orient="records")

    if args.trees:
        root_states, tree_info = read_root_states(
            args.trees,
            tag=tag,
            burnin_fraction=args.burnin_fraction,
        )
        # state_code_map is user code -> BEAST code; report the user's labels
        inverse_map = {v: k for k, v in code_map.items()}
        root_states["label"] = [inverse_map.get(s, s) for s in root_states["state"]]
        root_states.to_csv(f"{prefix}.root_states.tsv", sep="\t", index=False)
        digest["root_states"] = root_states.to_dict(orient="records")
        digest["root_states_info"] = tree_info

    table, worst = cross_check_registers(
        args.log,
        id_prefix,
        counts,
        code_map,
        [entry["burnin_states"] for entry in infos],
    )
    if table is not None:
        table.to_csv(f"{prefix}.count_check.tsv", sep="\t", index=False)
        digest["count_check"] = {
            "max_abs_difference": worst,
            "table": table.to_dict(orient="records"),
        }

    with open(f"{prefix}.summary.json", "w") as handle:
        json.dump(digest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # ---- console report ---------------------------------------------------
    print(
        f"{trait}: {info['n_samples_kept']}/{info['n_samples_total']} samples kept "
        f"(burn-in {info['burnin_states']} states), {info['n_events_total']} transitions"
    )
    if info["samples_with_count_mismatch"]:
        print(
            f"warning: {len(info['samples_with_count_mismatch'])} samples where the "
            "complete history length differs from totalChangeCount",
            file=sys.stderr,
        )
    for _, row in counts.iterrows():
        print(
            f"  {row['direction']:<10} n={row['n_events']:<5} mean={row['mean']:.2f} "
            f"[{row['lo95']:.2f}, {row['hi95']:.2f}] P(>0)={row['p_positive']:.3f}"
        )
    if dates is not None:
        for _, row in dates.iterrows():
            if row["n_events"] == 0:
                print(f"  {row['direction']:<10} never observed")
                continue
            print(
                f"  {row['direction']:<10} first event "
                f"{row['first_median']:.3f} [{row['first_lo95']:.3f}, {row['first_hi95']:.3f}]"
                f"  pooled median {row['pooled_median']:.3f}"
            )
    for _, row in acquisitions.iterrows():
        if not row["p_acquired"]:
            print(f"  state {row['state']:<8} never acquired")
            continue
        when = row.get("first_date", "")
        detail = f", first {when}" if isinstance(when, str) and when else ""
        print(
            f"  state {row['state']:<8} acquired in {row['p_acquired']:.1%} of samples"
            f"{detail}"
        )
    if "root_states" in digest:
        top = digest["root_states"][0]
        print(
            f"  root state: {top['label']} with P={top['probability']:.3f} "
            f"({len(digest['root_states'])} states observed)"
        )
    if table is not None:
        print(f"  count cross-check: max |history - register| = {worst:.3g}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (JumpHistoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
