#!/usr/bin/env python3
"""Simulation-based *recovery* checks of the trait pipeline (Stage 1, WP6).

For every scenario the whole pipeline is run on simulated data with a *known*
history:

    simulate_trait.py -> trait_data.py -> generate_beast_xml.py -> BEAST ->
    summarize_jumps.py

and the posterior is compared with the truth:

* does the true acquisition date fall inside the posterior 95% interval of the
  earliest transition in that direction (and of the pooled transition times)?
* does the true number of transitions fall inside the 95% interval of the
  posterior count?
* what posterior probability does the true root state get?

These are recovery checks, not a calibration test.  Each scenario *conditions*
the simulation on the number of events (and often on the derived fraction), so
the simulated histories are drawn from the prior conditional on that constraint
rather than from the prior the analysis uses; the data therefore contain less
information than an unconditioned draw would, and the reported coverage of the
95% intervals cannot be read as a frequentist 95%.  A genuine calibration check
would draw the rates from the prior, simulate without constraints, and repeat
over many replicates; the ``conditional_on`` field of every record states what
was fixed so that the results are never quoted as coverage.

Scenarios cover single gains, repeated gains, losses, near fixation and sparse
(missing) trait observations.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dates import decimal_year
from generate_beast_xml import main as generate_xml
from jump_history import JumpHistoryError
from simulate_trait import SimulationError
from simulate_trait import main as simulate
from summarize_jumps import main as summarize
from trait_data import TraitError
from trait_data import main as preprocess

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR.parent / "templates" / "beast_template.xml.j2"

#: scenario -> simulate_trait options (plus optional ``mask_fraction``)
SCENARIOS: dict[str, dict] = {
    "single_gain": {
        "root_state": "0",
        "n_events": 1,
        "rates": "0.08,0.000001",
        "min_derived_fraction": 0.3,
    },
    "recurrent_gain": {
        "root_state": "0",
        "n_events": 3,
        "rates": "0.4,0.05",
        "min_derived_fraction": 0.3,
    },
    "single_loss": {
        "root_state": "1",
        "n_events": 1,
        "rates": "0.000001,0.08",
        "min_derived_fraction": 0.3,
    },
    "near_fixation": {
        "root_state": "0",
        "n_events": 1,
        "rates": "0.08,0.000001",
        "min_derived_fraction": 0.75,
        "max_derived_fraction": 0.95,
    },
    "sparse": {
        "root_state": "0",
        "n_events": 1,
        "rates": "0.08,0.000001",
        "min_derived_fraction": 0.3,
        "mask_fraction": 0.5,
    },
}


def mask_observations(source, target, fraction: float, seed: int) -> int:
    """Replace `fraction` of the tip states with a missing value."""
    lines = Path(source).read_text().splitlines()
    header, rows = lines[0], lines[1:]
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), max(1, int(len(rows) * fraction)))
    for index in indices:
        taxon = rows[index].split(",")[0]
        rows[index] = f"{taxon},NA"
    Path(target).write_text("\n".join([header] + rows) + "\n")
    return len(indices)


def read_tsv(path) -> list[dict]:
    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def simulate_argv(args, options: dict, prefix, seed: int) -> list[str]:
    argv = [
        "--output-prefix", str(prefix),
        "--n-tips", str(args.n_tips),
        "--tree-height", str(args.tree_height),
        "--states", "0,1",
        "--model", "asymmetric",
        "--rates", options["rates"],
        "--root-state", options["root_state"],
        "--mrsd", args.mrsd,
        "--seed", str(seed),
    ]
    if options.get("n_events"):
        argv += ["--n-events", str(options["n_events"])]
    if options.get("min_derived_fraction"):
        argv += ["--min-derived-fraction", str(options["min_derived_fraction"])]
    if options.get("max_derived_fraction"):
        argv += ["--max-derived-fraction", str(options["max_derived_fraction"])]
    return argv


def simulate_with_retry(args, options: dict, prefix, index: int, record: dict) -> int:
    """Simulate a history, retrying with a different seed when the scenario
    constraints are unreachable for the tree grown from the first seed."""
    base = args.seed + index
    for attempt in range(6):
        seed = base + attempt * 7919
        try:
            simulate(simulate_argv(args, options, prefix, seed))
        except SimulationError:
            continue
        record["seed"] = seed
        return seed
    raise SimulationError(f"no history for {options} in 6 attempts")


def run_replicate(args, scenario: str, index: int) -> dict:
    options = SCENARIOS[scenario]
    directory = Path(args.work_dir) / f"{scenario}_{index}"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "run"
    record: dict = {"scenario": scenario, "replicate": index}

    seed = simulate_with_retry(args, options, prefix, index, record)

    traits_csv = f"{prefix}.traits.csv"
    if options.get("mask_fraction"):
        masked = f"{prefix}.masked.traits.csv"
        record["n_masked"] = mask_observations(
            traits_csv, masked, options["mask_fraction"], seed
        )
        traits_csv = masked

    preprocess([
        "--input-trees", f"{prefix}.trees",
        "--trait-data", traits_csv,
        "--trait", "state",
        "--match-mode", "exact",
        "--output-prefix", str(prefix),
    ])
    generate_xml([
        "--input_template", str(TEMPLATE),
        "--input_trees", f"{prefix}.trees",
        "--output_xml", f"{prefix}.xml",
        "--traits", f"{prefix}.traits.tsv",
        "--id-prefix", "trait",
        "--trait-model", "asymmetric",
        "--log-complete-history",
        "--chain_length", str(args.chain_length),
        "--sample_every", str(args.sample_every),
        "--name", "run",
    ])

    completed = subprocess.run(
        [args.beast_bin, "-seed", str(args.beast_seed + index), "-overwrite", "run.xml"],
        cwd=directory, capture_output=True, text=True,
    )
    if completed.returncode != 0 or "Exception" in completed.stdout:
        raise RuntimeError(f"BEAST failed: {completed.stdout[-1500:]}")

    summarize([
        "--jump-history", f"{prefix}.jumpHistory.log",
        "--trees", f"{prefix}.trees",
        "--log", f"{prefix}.log",
        "--trait-json", f"{prefix}.traits.json",
        "--tag", "trait",
        "--prefix-id", "trait",
        "--mrsd", args.mrsd,
        "--burnin-fraction", str(args.burnin_fraction),
        "--output-prefix", str(prefix),
    ])

    truth = read_tsv(f"{prefix}.truth.tsv")
    counts = {row["direction"]: row for row in read_tsv(f"{prefix}.event_counts.tsv")}
    dates = {row["direction"]: row for row in read_tsv(f"{prefix}.event_dates.tsv")}
    roots = read_tsv(f"{prefix}.root_states.tsv")
    sim = json.loads(Path(f"{prefix}.sim.json").read_text())

    # the direction the simulation actually moved in
    direction = max(
        sim["n_events_by_direction"], key=lambda key: sim["n_events_by_direction"][key]
    )
    source, dest = direction.split("->")
    true_events = [
        row for row in truth if row["source"] == source and row["dest"] == dest
    ]
    true_earliest = decimal_year(args.mrsd) - max(float(row["height"]) for row in true_events)
    true_count = len(true_events)

    count_row = counts[direction]
    date_row = dates[direction]
    # the simulation was conditioned on these, so the intervals below are not a
    # calibration check; keep the constraint with the numbers
    record["conditional_on"] = {
        key: options[key]
        for key in ("n_events", "min_derived_fraction", "max_derived_fraction")
        if options.get(key) is not None
    }
    record.update(
        {
            "direction": direction,
            "true_count": true_count,
            "true_earliest": true_earliest,
            "true_root": sim["root_state"],
            "derived_fraction": sim["derived_fraction"],
            "posterior_count_mean": float(count_row["mean"]),
            "posterior_count_lo95": float(count_row["lo95"]),
            "posterior_count_hi95": float(count_row["hi95"]),
            "p_positive": float(count_row["p_positive"]),
            "first_lo95": float(date_row["first_lo95"]),
            "first_hi95": float(date_row["first_hi95"]),
            "pooled_lo95": float(date_row["pooled_lo95"]),
            "pooled_hi95": float(date_row["pooled_hi95"]),
            "count_covered": float(count_row["lo95"]) <= true_count <= float(count_row["hi95"]),
            "first_covered": float(date_row["first_lo95"]) <= true_earliest <= float(date_row["first_hi95"]),
            "pooled_covered": float(date_row["pooled_lo95"]) <= true_earliest <= float(date_row["pooled_hi95"]),
            "true_root_probability": next(
                (float(row["probability"]) for row in roots if row["label"] == sim["root_state"]),
                0.0,
            ),
            "count_check": json.loads(Path(f"{prefix}.summary.json").read_text())["count_check"][
                "max_abs_difference"
            ],
        }
    )
    if not args.keep_runs:
        shutil.rmtree(directory, ignore_errors=True)
    return record


def summarise(records: list[dict]) -> dict:
    summary = {}
    for scenario in sorted({record["scenario"] for record in records}):
        rows = [record for record in records if record["scenario"] == scenario]
        ok = [row for row in rows if "error" not in row]
        if not ok:
            summary[scenario] = {"n": len(rows), "n_failed": len(rows)}
            continue
        summary[scenario] = {
            "n": len(ok),
            "n_failed": len(rows) - len(ok),
            "coverage_first_date": sum(row["first_covered"] for row in ok) / len(ok),
            "coverage_pooled_date": sum(row["pooled_covered"] for row in ok) / len(ok),
            "coverage_count": sum(row["count_covered"] for row in ok) / len(ok),
            "mean_true_root_probability": sum(row["true_root_probability"] for row in ok) / len(ok),
            "mean_posterior_count": sum(row["posterior_count_mean"] for row in ok) / len(ok),
            "mean_true_count": sum(row["true_count"] for row in ok) / len(ok),
            "mean_p_positive": sum(row["p_positive"] for row in ok) / len(ok),
            "max_count_check_difference": max(row["count_check"] for row in ok),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--n-tips", type=int, default=24)
    parser.add_argument("--tree-height", type=float, default=3.0)
    parser.add_argument("--chain-length", type=int, default=300000)
    parser.add_argument("--sample-every", type=int, default=3000)
    parser.add_argument("--burnin-fraction", type=float, default=0.1)
    parser.add_argument("--mrsd", default="2020-01-01")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--beast-seed", type=int, default=1000)
    parser.add_argument("--beast-bin", default="beast")
    parser.add_argument("--work-dir", default="calibration_runs")
    parser.add_argument("--keep-runs", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; known: {sorted(SCENARIOS)}")

    records = []
    for scenario in scenarios:
        for index in range(args.replicates):
            try:
                record = run_replicate(args, scenario, index)
            except (RuntimeError, TraitError, JumpHistoryError, SimulationError, ValueError) as error:
                record = {"scenario": scenario, "replicate": index, "error": str(error)}
            records.append(record)
            status = record.get("error") or (
                f"{record['direction']} count {record['posterior_count_mean']:.2f} "
                f"(true {record['true_count']}), first date "
                f"{'covered' if record['first_covered'] else 'MISSED'}"
            )
            print(f"[{scenario} {index + 1}/{args.replicates}] {status}", flush=True)

    summary = summarise(records)
    prefix = args.output_prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for record in records for key in record})
    with open(f"{prefix}.calibration.tsv", "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for record in records:
            handle.write("\t".join(str(record.get(column, "")) for column in columns) + "\n")
    with open(f"{prefix}.calibration.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print()
    print(f"{'scenario':<16}{'n':>4}{'first':>8}{'pooled':>8}{'count':>8}{'root P':>8}{'count chk':>11}")
    for scenario, row in summary.items():
        if "coverage_first_date" not in row:
            print(f"{scenario:<16}{row['n']:>4}{'  (all failed)':>27}")
            continue
        print(
            f"{scenario:<16}{row['n']:>4}"
            f"{row['coverage_first_date']:>8.2f}{row['coverage_pooled_date']:>8.2f}"
            f"{row['coverage_count']:>8.2f}{row['mean_true_root_probability']:>8.2f}"
            f"{row['max_count_check_difference']:>11.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
