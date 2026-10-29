#!/usr/bin/env python3
"""Parse and summarise BEAST complete jump histories (Stage 1, WP4).

BEAST's ``completeHistoryLogger`` writes one line per posterior sample::

    state<TAB>totalChangeCount<TAB>{...} N

where the history column contains every individual transition, nested and
comma separated::

    {{1,2.954822623502875,s0,s1},{1,1.2041,s1,s0}} 2

Each event is ``{site,height,source,destination}``; ``height`` is measured from
the most recent tip, so a transition at height ``h`` happened ``h`` time units
(years) before the most recent sampling date.

This module turns that log into tidy tables:

* every event of every retained sample (``events_frame``)
* posterior counts per transition direction (``direction_counts``)
* posterior distributions of transition dates (``date_summary``)
* posterior root-state probabilities from the ``logTree`` file
  (``read_root_states``)

Samples without a transition of a given direction are *kept* in the counts (they
carry the information that the direction may not have occurred at all), and the
date summaries are computed over the samples that do have at least one event,
which is the quantity of interest when a transition may be absent.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dates import decimal_year, format_date

#: ``{site,height,source,destination}``
EVENT_RE = re.compile(r"\{(\d+),([^,{}]+),([^{},]+),([^{},]+)\}")
TREE_RE = re.compile(r"^\s*tree\s+(\S+)\s*(?:\[&[^\]]*\])?\s*=\s*(.*);\s*$")
STATE_NAME_RE = re.compile(r"STATE_(\d+)")

EVENT_COLUMNS = [
    "chain", "state", "event", "site", "height", "date", "date_iso", "source", "dest", "direction",
]


@dataclass
class JumpSample:
    """One posterior sample of the complete transition history."""

    state: int
    total: int
    events: list[tuple[int, float, str, str]] = field(default_factory=list)
    #: 1-based chain index, so samples from different chains stay distinguishable
    chain: int = 1


class JumpHistoryError(Exception):
    pass


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_jump_history(path, burnin_states: int | None = None, burnin_fraction: float = 0.1):
    """Read a ``completeHistoryLogger`` log.

    Returns ``(samples, info)`` where ``samples`` is the list of
    :class:`JumpSample` after burn-in and ``info`` records the burn-in decision.
    """
    path = Path(path)
    if not path.exists():
        raise JumpHistoryError(f"jump history log not found: {path}")

    all_samples: list[JumpSample] = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[0].strip().lower() == "state":
                continue
            if len(fields) < 3:
                raise JumpHistoryError(f"{path}: malformed line: {line[:120]!r}")
            state = int(fields[0])
            total = int(round(float(fields[1])))
            events: list[tuple[int, float, str, str]] = []
            for column in fields[2:]:
                for site, height, source, dest in EVENT_RE.findall(column):
                    events.append((int(site), float(height), source.strip(), dest.strip()))
            all_samples.append(JumpSample(state=state, total=total, events=events))

    if not all_samples:
        raise JumpHistoryError(f"{path}: no samples found")

    all_samples.sort(key=lambda s: s.state)
    if burnin_states is None:
        burnin_states = int(all_samples[-1].state * burnin_fraction)
    kept = [s for s in all_samples if s.state >= burnin_states]
    if not kept:
        kept = all_samples[-1:]

    mismatched = [s.state for s in kept if len(s.events) != s.total]
    info = {
        "file": str(path),
        "n_samples_total": len(all_samples),
        "n_samples_kept": len(kept),
        "burnin_states": burnin_states,
        "n_events_total": int(sum(s.total for s in kept)),
        "samples_with_count_mismatch": mismatched[:10],
    }
    return kept, info


def events_frame(samples: list[JumpSample], mrsd=None, code_map: dict | None = None) -> pd.DataFrame:
    """Tidy table with one row per transition of every retained sample."""
    inverse = {v: k for k, v in (code_map or {}).items()}
    records = []
    for sample in samples:
        for index, (site, height, source, dest) in enumerate(sample.events, start=1):
            source = inverse.get(source, source)
            dest = inverse.get(dest, dest)
            record = {
                "chain": sample.chain,
                "state": sample.state,
                "event": index,
                "site": site,
                "height": height,
                "source": source,
                "dest": dest,
                "direction": f"{source}->{dest}",
            }
            if mrsd is not None:
                date = decimal_year(mrsd) - height
                record["date"] = date
                record["date_iso"] = format_date(date)
            records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=EVENT_COLUMNS)
    return frame


def observed_directions(samples: list[JumpSample], code_map: dict | None = None) -> list[str]:
    """Every transition direction seen in the sample, in natural order."""
    inverse = {v: k for k, v in (code_map or {}).items()}
    seen = {
        f"{inverse.get(a, a)}->{inverse.get(b, b)}" for s in samples for _, _, a, b in s.events
    }
    return sorted(seen, key=lambda d: (d.split("->")[1], d.split("->")[0]))


def expected_directions(states: list[str]) -> list[str]:
    """Every ordered pair of distinct states.

    Summaries are reported for all of them, not only the directions that were
    sampled: a direction with no event is evidence too, and it must show as
    ``P(>0) = 0`` rather than be missing from the table.
    """
    return sorted(
        (f"{source}->{dest}" for source in states for dest in states if source != dest),
        key=lambda d: (d.split("->")[1], d.split("->")[0]),
    )


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def direction_counts(
    samples: list[JumpSample],
    directions: list[str] | None = None,
    code_map: dict | None = None,
) -> pd.DataFrame:
    """Posterior of the number of transitions per direction.

    Every retained sample contributes a count (zero when the direction does not
    appear), so ``p_positive`` is the posterior probability that the direction
    occurred at least once.
    """
    inverse = {v: k for k, v in (code_map or {}).items()}
    if directions is None:
        directions = observed_directions(samples, code_map)

    counts = {d: [] for d in directions}
    for sample in samples:
        seen = Counter(
            f"{inverse.get(a, a)}->{inverse.get(b, b)}" for _, _, a, b in sample.events
        )
        for direction in directions:
            counts[direction].append(seen.get(direction, 0))

    rows = []
    for direction in directions:
        values = np.asarray(counts[direction], dtype=float)
        source, dest = direction.split("->")
        rows.append(
            {
                "source": source,
                "dest": dest,
                "direction": direction,
                "n_events": int(values.sum()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "lo95": float(np.percentile(values, 2.5)),
                "hi95": float(np.percentile(values, 97.5)),
                "p_positive": float((values > 0).mean()),
                "max": int(values.max()),
            }
        )
    return pd.DataFrame(rows)


def date_summary(
    samples: list[JumpSample],
    mrsd=None,
    code_map: dict | None = None,
    directions: list[str] | None = None,
) -> pd.DataFrame:
    """Posterior distribution of transition dates per direction.

    Two summaries are reported per direction:

    ``first_*``
        the distribution of the *earliest* (oldest) transition of that
        direction within each sample that has at least one -- the natural
        summary when a direction is expected to occur once.
    ``pooled_*``
        the distribution of all transitions pooled over samples (the density of
        transition times).
    """
    if mrsd is None:
        raise JumpHistoryError("date summaries need --mrsd")
    inverse = {v: k for k, v in (code_map or {}).items()}
    if directions is None:
        directions = observed_directions(samples, code_map)

    per_direction: dict[str, dict[str, list[float]]] = {
        d: {"first": [], "last": [], "pooled": []} for d in directions
    }
    for sample in samples:
        grouped: dict[str, list[float]] = {}
        for _, height, source, dest in sample.events:
            direction = f"{inverse.get(source, source)}->{inverse.get(dest, dest)}"
            if direction not in per_direction:
                continue
            grouped.setdefault(direction, []).append(height)
        for direction, heights in grouped.items():
            dates = [decimal_year(mrsd) - h for h in heights]
            per_direction[direction]["first"].append(min(dates))
            per_direction[direction]["last"].append(max(dates))
            per_direction[direction]["pooled"].extend(dates)

    rows = []
    for direction in directions:
        source, dest = direction.split("->")
        entry = per_direction[direction]
        row = {
            "source": source,
            "dest": dest,
            "direction": direction,
            "n_events": len(entry["pooled"]),
            "n_samples_with_event": len(entry["first"]),
        }
        for label in ("first", "last", "pooled"):
            values = np.asarray(entry[label], dtype=float)
            if values.size == 0:
                row.update(
                    {
                        f"{label}_mean": np.nan,
                        f"{label}_median": np.nan,
                        f"{label}_lo95": np.nan,
                        f"{label}_hi95": np.nan,
                    }
                )
                continue
            lo, median, hi = np.percentile(values, [2.5, 50, 97.5])
            row.update(
                {
                    f"{label}_mean": float(values.mean()),
                    f"{label}_median": float(median),
                    f"{label}_lo95": float(lo),
                    f"{label}_hi95": float(hi),
                }
            )
            if label != "pooled":
                row[f"{label}_date"] = format_date(float(median))
        rows.append(row)
    return pd.DataFrame(rows)


def acquisition_summary(
    samples: list[JumpSample],
    mrsd=None,
    code_map: dict | None = None,
    states: list[str] | None = None,
) -> pd.DataFrame:
    """Posterior of the first transition *into* each state, from any source.

    For every sample, the earliest event whose destination is the state is
    computed first and then summarised across samples.  Taking the minimum of
    the per-direction "earliest" medians instead would be later than the truth
    (the minimum of per-sample minima is <= the minimum of the medians) and
    would ignore every sample in which that source was never used.

    ``p_acquired`` is the posterior probability that the state was entered at
    least once; it is 0 for a state that is always the root state.
    """
    inverse = {v: k for k, v in (code_map or {}).items()}
    if states is None:
        states = sorted({inverse.get(b, b) for s in samples for _, _, _, b in s.events})

    per_state: dict[str, list[float]] = {state: [] for state in states}
    for sample in samples:
        earliest: dict[str, float] = {}
        for _, height, _, dest in sample.events:
            label = inverse.get(dest, dest)
            if label in per_state and (label not in earliest or height > earliest[label]):
                earliest[label] = height
        for state in states:
            if state in earliest:
                # heights grow towards the past; dates shrink
                per_state[state].append(
                    decimal_year(mrsd) - earliest[state] if mrsd is not None else earliest[state]
                )

    rows = []
    for state in states:
        values = np.asarray(per_state[state], dtype=float)
        row = {
            "state": state,
            "n_samples": len(samples),
            "n_samples_with_event": int(values.size),
            "p_acquired": float(values.size / len(samples)) if samples else 0.0,
        }
        if values.size:
            lo, median, hi = np.percentile(values, [2.5, 50, 97.5])
            row.update(
                {
                    "first_mean": float(values.mean()),
                    "first_median": float(median),
                    "first_lo95": float(lo),
                    "first_hi95": float(hi),
                }
            )
            if mrsd is not None:
                row["first_date"] = format_date(float(median))
        else:
            row.update(
                {
                    "first_mean": np.nan,
                    "first_median": np.nan,
                    "first_lo95": np.nan,
                    "first_hi95": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# ancestral states
# --------------------------------------------------------------------------
def read_root_states(
    trees_paths, tag: str, burnin_trees: int | None = None, burnin_fraction: float = 0.1
):
    """Posterior distribution of the root state from BEAST ``logTree`` file(s).

    The state of every node is annotated as ``tag="state"``; the root annotation
    is the last bracket group of each tree.  ``trees_paths`` is one path or a
    list of them (one per chain), and the burn-in is taken from the start of
    *each* file: concatenating the chains first and dropping one prefix would
    keep the burn-in of every chain but the first.
    """
    if isinstance(trees_paths, (str, Path)):
        trees_paths = [trees_paths]

    pattern = re.compile(rf'{re.escape(tag)}="([^"]*)"')
    per_file: list[list[tuple[int, int, str]]] = []
    for trees_path in trees_paths:
        trees_path = Path(trees_path)
        if not trees_path.exists():
            raise JumpHistoryError(f"tree log not found: {trees_path}")
        entries: list[tuple[int, int, str]] = []
        with open(trees_path) as handle:
            for index, line in enumerate(handle):
                match = TREE_RE.match(line)
                if not match:
                    continue
                name, newick = match.groups()
                root_annotation = newick[newick.rfind(")"):]
                state_match = pattern.search(root_annotation)
                if not state_match:
                    continue
                state_number = STATE_NAME_RE.search(name)
                entries.append(
                    (
                        int(state_number.group(1)) if state_number else index,
                        index,
                        state_match.group(1),
                    )
                )
        per_file.append(entries)

    if not any(per_file):
        raise JumpHistoryError(f"{trees_paths}: no root state annotated with tag {tag!r}")

    kept: list[tuple[int, int, str]] = []
    dropped = 0
    for entries in per_file:
        drop = burnin_trees if burnin_trees is not None else int(len(entries) * burnin_fraction)
        dropped += min(drop, max(len(entries) - 1, 0))
        kept.extend(entries[drop:] or entries[-1:])

    counts = Counter(state for _, _, state in kept)
    total = sum(counts.values())
    return (
        pd.DataFrame(
            {
                "state": list(counts.keys()),
                "n": list(counts.values()),
                "probability": [c / total for c in counts.values()],
            }
        ).sort_values("probability", ascending=False, ignore_index=True),
        {
            "n_trees_total": sum(len(entries) for entries in per_file),
            "n_trees_kept": len(kept),
            "burnin_trees": dropped,
            "n_files": len(per_file),
        },
    )


# --------------------------------------------------------------------------
# cross-checks
# --------------------------------------------------------------------------
def read_count_registers(log_path, prefix: str, burnin_states: int | None = None) -> pd.DataFrame:
    """Read the ``c_<...>`` Markov jump count registers from a BEAST log file.

    ``prefix`` is the BEAST id prefix (``location`` or the trait name); the
    returned frame keeps the ``state`` column so that callers can align it with
    the jump histories.
    """
    frame = pd.read_csv(log_path, sep="\t", comment="#")
    columns = [c for c in frame.columns if c.startswith("c_")]
    if not columns:
        raise JumpHistoryError(f"{log_path}: no c_* jump count registers")
    if burnin_states is not None and "state" in frame.columns:
        frame = frame[frame["state"] >= burnin_states]
    keep = [c for c in ("state",) + tuple(columns) if c in frame.columns]
    return frame[keep].reset_index(drop=True)
