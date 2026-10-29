#!/usr/bin/env python3
"""MCMC diagnostics for BEAST logs (Stage 1, WP5/WP6).

Computes the effective sample size of every numeric column of a BEAST log using
Geyer's initial positive sequence estimator on the autocorrelation function, plus
the usual posterior summaries.  Columns that are not numeric (the current tree,
the complete history string) are ignored.

This deliberately avoids depending on ``loganalyser`` so the check runs in the
same Python environment as the rest of the pipeline and is unit-testable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ESS_THRESHOLD = 200.0


def autocorrelation(values: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function of a 1-D array."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator == 0.0:
        return np.zeros(n)
    return np.correlate(centred, centred, mode="full")[n - 1:] / denominator


def effective_sample_size(values) -> float:
    """Geyer initial positive sequence ESS.

    ``ESS = n / (-1 + 2 * sum(Gamma_k))``, with ``Gamma_k = rho_2k + rho_2k+1``
    truncated at the first non-positive pair (Geyer 1992).  Pairing the lags
    from an odd offset instead would cut the sum at a different place, because
    only Geyer's pairs are guaranteed to be positive for a reversible chain.

    A constant series has no autocorrelation to estimate; its ESS is undefined
    and ``nan`` is returned rather than ``n``, which would read as a passing
    parameter.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return float(n)
    if np.all(values == values[0]):
        return float("nan")
    acf = autocorrelation(values)
    total = 0.0
    lag = 0
    while 2 * lag + 1 < n:
        pair = acf[2 * lag] + acf[2 * lag + 1]
        if pair < 0:
            break
        total += pair
        lag += 1
    tau = -1.0 + 2.0 * total
    return float(n / tau) if tau > 0 else float(n)


def log_diagnostics(log_path, burnin_fraction: float = 0.1, ess_threshold: float = DEFAULT_ESS_THRESHOLD):
    """ESS and posterior summaries for every numeric column of a BEAST log."""
    frame = pd.read_csv(log_path, sep="\t", comment="#")
    if frame.empty:
        raise ValueError(f"{log_path}: no samples")
    if "state" in frame.columns:
        threshold = frame["state"].max() * burnin_fraction
        frame = frame[frame["state"] >= threshold]
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["state"], errors="ignore")
    rows = []
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        ess = effective_sample_size(values)
        rows.append(
            {
                "column": column,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "lo95": float(np.percentile(values, 2.5)),
                "hi95": float(np.percentile(values, 97.5)),
                "ess": ess,
                "n_samples": len(values),
                "passes": bool(ess >= ess_threshold),
            }
        )
    table = pd.DataFrame(rows).sort_values("ess")
    return table, {"burnin_fraction": burnin_fraction, "ess_threshold": ess_threshold}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", nargs="+", help="BEAST log file(s), one per chain")
    parser.add_argument("--output", required=True, help="ESS table to write")
    parser.add_argument("--burnin-fraction", type=float, default=0.1)
    parser.add_argument("--ess-threshold", type=float, default=DEFAULT_ESS_THRESHOLD)
    parser.add_argument(
        "--fail-on-low-ess",
        action="store_true",
        help="exit non-zero when a parameter is below the threshold",
    )
    args = parser.parse_args(argv)

    tables = []
    for path in args.log:
        table, _ = log_diagnostics(path, args.burnin_fraction, args.ess_threshold)
        table.insert(0, "chain", Path(path).name)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(args.output, sep="\t", index=False)

    undefined = combined[combined["ess"].isna()]
    failing = combined[combined["ess"].notna() & ~combined["passes"].fillna(False)]
    finite = combined["ess"].dropna()
    smallest = (
        f"{finite.min():.1f} ({combined.loc[finite.idxmin(), 'column']})"
        if not finite.empty
        else "undefined (no varying parameter)"
    )
    print(f"{len(combined)} parameters, ESS threshold {args.ess_threshold:g}; min ESS {smallest}")
    if not undefined.empty:
        print(
            f"warning: {len(undefined)} parameter(s) are constant, so their ESS is undefined: "
            + ", ".join(f"{row.chain}:{row.column}" for row in undefined.itertuples()),
            file=sys.stderr,
        )
    if not failing.empty:
        print(
            f"warning: {len(failing)} parameter(s) below the ESS threshold "
            f"of {args.ess_threshold:g} -- the chain has not converged, so the "
            "posterior summaries are not trustworthy: "
            + ", ".join(
                f"{row.chain}:{row.column}={row.ess:.1f}" for row in failing.itertuples()
            ),
            file=sys.stderr,
        )
        if args.fail_on_low_ess:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
