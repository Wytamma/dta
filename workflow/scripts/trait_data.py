#!/usr/bin/env python3
"""Trait preprocessing for the DTA pipeline (Stage 1, WP1).

Generalises the original phylogeographic input -- where the discrete state was
parsed out of the taxon label -- into arbitrary categorical genomic traits
supplied as external metadata.

A trait is any categorical genome feature; only its *source* differs:

``table``
    an external CSV/TSV with one row per taxon (``--trait-data``), e.g. an AMR
    annotation sheet.
``label``
    the state is encoded in the taxon label itself, e.g.
    ``EPI_ISL_2546697|India|B.1.617.2|2021-05-05`` (``--source label``), which is
    how the geographic analyses have always worked.

Taxon matching between the metadata and the phylogeny can be ``exact``,
``field`` (split the label on a delimiter and take one field) or ``regex``.

Outputs (written next to ``--output-prefix``):

``<prefix>.traits.tsv``
    ``taxon<TAB>state``, one row per taxon **in tree order**, ``?`` for
    unobserved taxa.  This is the file consumed by the XML generator.
``<prefix>.traits.json``
    machine readable report: state codes, counts, missing taxa, match mode.

The module is importable (``from trait_data import ...``) as well as usable as
a command line tool.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# BEAST codes unobserved attributes with "?" (see dr.evolution.datatype.GeneralDataType).
MISSING = "?"

#: Tokens in the metadata that mean "not observed".
DEFAULT_MISSING_TOKENS = ("", "?", "na", "n/a", "nan", "none", "null", "-", ".")

#: Characters that would corrupt the BEAST XML, the jump-history log or the
#: pairwise register names.
FORBIDDEN_STATE_CHARS = set(',{}"\t\n\r')
WARN_STATE_CHARS = set(". ")

MATCH_MODES = ("exact", "field", "regex")


class TraitError(Exception):
    """Raised for any user-facing problem with the trait input."""


# --------------------------------------------------------------------------
# taxa
# --------------------------------------------------------------------------
def _split_taxon_tokens(text: str) -> list[str]:
    """Split a NEXUS taxlabels line into individual (possibly quoted) labels."""
    text = text.strip()
    if not text or text == ";":
        return []
    if text.endswith(";"):
        text = text[:-1]
    tokens = re.findall(r"'([^']*)'|\"([^\"]*)\"|(\S+)", text)
    return [a or b or c for a, b, c in tokens]


def extract_taxa_from_nexus(nexus_file) -> list[str]:
    """Return the taxon labels of a NEXUS file, in file order.

    Reads the ``taxlabels`` block (the labels used by the ``translate`` table of
    the tree block) and tolerates both one-label-per-line and all-on-one-line
    formatting.
    """
    taxa: list[str] = []
    in_block = False
    with open(nexus_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if not in_block:
                if low.startswith("begin taxa"):
                    in_block = True
                continue
            if low.startswith("end"):
                break
            if low.startswith("dimensions"):
                continue
            if low.startswith("taxlabels"):
                taxa.extend(_split_taxon_tokens(line[len("taxlabels"):]))
                continue
            taxa.extend(_split_taxon_tokens(line))
    if not taxa:
        raise TraitError(f"no taxlabels block found in {nexus_file}")
    return taxa


# --------------------------------------------------------------------------
# observation tables
# --------------------------------------------------------------------------
@dataclass
class TraitTable:
    """A tidy taxon -> raw state mapping read from an external table."""

    trait: str
    observations: dict[str, str]
    source: str
    columns: list[str]

    def __len__(self) -> int:
        return len(self.observations)


def _sniff_delimiter(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".tab"):
        return "\t"
    if suffix in (".csv",):
        return ","
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def read_trait_table(
    path,
    trait: str | None = None,
    taxon_column: str | None = None,
    state_column: str | None = None,
    delimiter: str | None = None,
    header: bool = True,
) -> TraitTable:
    """Read an external metadata table into a :class:`TraitTable`.

    ``trait`` selects the column holding the state; when omitted the table must
    contain exactly one candidate column besides the taxon column.
    """
    path = Path(path)
    if not path.exists():
        raise TraitError(f"trait data file not found: {path}")
    text = path.read_text()
    delim = delimiter or _sniff_delimiter(path, text)
    rows = [
        [c.strip() for c in row]
        for row in csv.reader(text.splitlines(), delimiter=delim)
        if any(c.strip() for c in row)
    ]
    if not rows:
        raise TraitError(f"{path}: no data rows")

    if header:
        columns = rows[0]
        body = rows[1:]
    else:
        columns = [f"column{i + 1}" for i in range(len(rows[0]))]
        body = rows

    if len(columns) < 2:
        raise TraitError(
            f"{path}: need at least a taxon column and one trait column, found {columns}"
        )

    # ---- taxon column -----------------------------------------------------
    if taxon_column:
        if taxon_column not in columns:
            raise TraitError(f"{path}: no column {taxon_column!r} (have {columns})")
        taxon_idx = columns.index(taxon_column)
    else:
        taxon_idx = 0
        for i, col in enumerate(columns):
            if re.fullmatch(r"(taxon|taxa|strain|isolate|name|id|accession|seqid)", col, re.I):
                taxon_idx = i
                break

    # ---- state column -----------------------------------------------------
    candidates = [c for i, c in enumerate(columns) if i != taxon_idx]
    if state_column:
        if state_column not in candidates:
            raise TraitError(f"{path}: no column {state_column!r} (have {candidates})")
        state_col = state_column
    elif trait:
        if not header:
            raise TraitError("--trait requires a header row")
        if trait not in candidates:
            raise TraitError(f"{path}: no column {trait!r} (have {candidates})")
        state_col = trait
    else:
        if len(candidates) != 1:
            raise TraitError(
                f"{path}: cannot infer the trait column, candidates are {candidates}; "
                "pass --trait or --state-column"
            )
        state_col = candidates[0]
    state_idx = columns.index(state_col)

    observations: dict[str, str] = {}
    for row in body:
        if len(row) <= max(taxon_idx, state_idx):
            raise TraitError(f"{path}: short row {row!r}")
        key = row[taxon_idx]
        if not key:
            raise TraitError(f"{path}: empty taxon identifier in row {row!r}")
        if key in observations:
            raise TraitError(f"{path}: duplicate taxon {key!r}")
        observations[key] = row[state_idx]

    return TraitTable(trait=state_col, observations=observations, source=str(path), columns=columns)


# --------------------------------------------------------------------------
# taxon matching
# --------------------------------------------------------------------------
def extract_label_value(
    label: str,
    mode: str = "exact",
    delimiter: str = "|",
    index: int = 0,
    regex: str | None = None,
) -> str | None:
    """Reduce a tree label to the key used to look up the metadata table."""
    if mode == "exact":
        return label
    if mode == "field":
        parts = label.split(delimiter)
        i = index if index >= 0 else len(parts) + index
        if not 0 <= i < len(parts):
            return None
        return parts[i]
    if mode == "regex":
        if not regex:
            raise TraitError("--match-mode regex requires --taxon-regex")
        match = re.search(regex, label)
        if not match:
            return None
        return match.group(1) if match.groups() else match.group(0)
    raise TraitError(f"unknown match mode {mode!r} (expected one of {MATCH_MODES})")


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------
@dataclass
class TraitAssignment:
    """Validated trait values for every taxon of a phylogeny."""

    trait: str
    states: list[str]
    taxa: list[str]
    values: list[str]
    missing: list[str]
    unmatched_observations: list[str]
    source: str = ""

    @property
    def observed(self) -> list[tuple[str, str]]:
        return [(t, v) for t, v in zip(self.taxa, self.values) if v != MISSING]

    def counts(self) -> dict[str, int]:
        counts = {s: 0 for s in self.states}
        for value in self.values:
            if value != MISSING:
                counts[value] = counts.get(value, 0) + 1
        return counts

    def frequencies(self) -> list[float]:
        counts = self.counts()
        total = sum(counts.values())
        if total == 0:
            return [1.0 / len(self.states)] * len(self.states)
        return [counts[s] / total for s in self.states]

    def state_code_map(self) -> dict[str, str]:
        """User state code -> BEAST safe state code (identity for most traits)."""
        return {code: to_beast_code(code) for code in self.states}

    def to_dict(self) -> dict:
        return {
            "trait": self.trait,
            "states": self.states,
            "state_code_map": self.state_code_map(),
            "state_counts": self.counts(),
            "n_taxa": len(self.taxa),
            "n_observed": len(self.observed),
            "n_missing": len(self.missing),
            "missing_taxa": self.missing,
            "unmatched_observations": self.unmatched_observations,
            "source": self.source,
        }


def natural_sort_key(value: str):
    """Sort state codes so that ``0,1,2,10`` beats ``0,1,10,2``."""
    return [(0, int(tok)) if tok.isdigit() else (1, tok) for tok in re.split(r"(\d+)", value)]


#: Anything a Java number parser accepts is a hazard, see :func:`to_beast_code`.
_NUMERIC_CODE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def to_beast_code(code: str) -> str:
    """Return a BEAST-safe state code (idempotent).

    BEAST 1.10's ``CompleteHistoryLogger`` parses its own history strings with a
    generic value parser that turns anything number-like into ``Integer`` or
    ``Double``, and then casts the result back to ``String``.  Numeric state
    codes (``0``, ``1``, ``0.5``, ...) therefore abort the run with a
    ``ClassCastException`` as soon as the complete-history logger fires.  Such
    codes are prefixed with ``s`` so that ``0`` becomes ``s0``.
    """
    if not _NUMERIC_CODE.match(code):
        return code
    return "s" + re.sub(r"[^A-Za-z0-9_]", "_", code)


def validate_state_code(code: str) -> None:
    bad = FORBIDDEN_STATE_CHARS & set(code)
    if bad:
        raise TraitError(
            f"state code {code!r} contains characters that break the BEAST XML "
            f"or the jump-history log: {sorted(bad)}"
        )
    if code == MISSING:
        raise TraitError(f"{MISSING!r} is reserved for missing observations")


def assign_traits(
    labels: list[str],
    table: TraitTable,
    mode: str = "exact",
    delimiter: str = "|",
    index: int = 0,
    regex: str | None = None,
    states: list[str] | None = None,
    missing_tokens=DEFAULT_MISSING_TOKENS,
    on_missing: str = "unknown",
    require_variation: bool = False,
) -> TraitAssignment:
    """Match ``table`` onto ``labels`` and validate the result.

    ``on_missing`` is ``unknown`` (code the taxon as ``?``, BEAST treats it as
    missing data) or ``error``.
    """
    missing_lower = {t.lower() for t in missing_tokens}
    values: list[str] = []
    missing: list[str] = []
    unmatched = set(table.observations)

    for label in labels:
        key = extract_label_value(label, mode, delimiter, index, regex)
        key = key.strip() if key is not None else None
        raw = table.observations.get(key) if key is not None else None
        if raw is None:
            if on_missing == "error":
                raise TraitError(f"no trait observation for taxon {label!r} (looked up {key!r})")
            values.append(MISSING)
            missing.append(label)
            continue
        unmatched.discard(key)
        raw = raw.strip()
        if raw.lower() in missing_lower:
            if on_missing == "error":
                raise TraitError(f"taxon {label!r} has a missing trait value ({raw!r})")
            values.append(MISSING)
            missing.append(label)
            continue
        validate_state_code(raw)
        values.append(raw)

    observed = sorted(set(values) - {MISSING})
    if not observed:
        raise TraitError("no taxon has an observed trait value")

    if states:
        state_list = list(states)
        unknown = sorted(set(observed) - set(state_list))
        if unknown:
            raise TraitError(
                f"observed states {unknown} are not part of the declared state set {state_list}"
            )
    else:
        state_list = sorted(observed, key=natural_sort_key)

    if len(state_list) < 2:
        raise TraitError(
            f"a discrete trait needs at least two states, got {state_list!r}"
        )

    for code in state_list:
        validate_state_code(code)

    unobserved = [s for s in state_list if s not in observed]
    if unobserved:
        print(
            f"warning: declared state(s) {unobserved} are never observed; "
            "their rates are not identifiable from the data",
            file=sys.stderr,
        )
    if require_variation and len(observed) < 2:
        raise TraitError(
            f"the trait is constant ({observed[0]!r} in every taxon); "
            "there is nothing to reconstruct"
        )

    return TraitAssignment(
        trait=table.trait,
        states=state_list,
        taxa=list(labels),
        values=values,
        missing=missing,
        unmatched_observations=sorted(unmatched),
        source=table.source,
    )


def assign_from_labels(
    labels: list[str],
    delimiter: str = "|",
    index: int = 0,
    regex: str | None = None,
    mode: str = "field",
    trait: str = "location",
    states: list[str] | None = None,
) -> TraitAssignment:
    """Build an assignment from states encoded in the labels (legacy mode)."""
    observations = {}
    for label in labels:
        value = extract_label_value(label, mode, delimiter, index, regex)
        if value is None:
            raise TraitError(f"could not extract a state from taxon label {label!r}")
        observations[label] = value.strip()
    table = TraitTable(trait=trait, observations=observations, source="<taxon labels>", columns=[])
    return assign_traits(
        labels,
        table,
        mode="exact",
        states=states,
        missing_tokens=(),
        on_missing="error",
    )


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_assignment(assignment: TraitAssignment, tsv_path, json_path=None) -> None:
    tsv_path = Path(tsv_path)
    with open(tsv_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["taxon", "state"])
        for taxon, value in zip(assignment.taxa, assignment.values):
            writer.writerow([taxon, value])
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(assignment.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")


def write_state_labels(assignment: TraitAssignment, path) -> None:
    """Write the BEAST code -> user label mapping used by the tree figures."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["beast_code", "label"])
        for code, beast_code in assignment.state_code_map().items():
            writer.writerow([beast_code, code])


def read_assignment(tsv_path) -> tuple[list[str], list[str]]:
    """Read back ``<prefix>.traits.tsv`` -> (taxa, states)."""
    taxa, values = [], []
    with open(tsv_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if header is None or [h.strip() for h in header[:2]] != ["taxon", "state"]:
            raise TraitError(f"{tsv_path}: expected a 'taxon\\tstate' header, got {header}")
        for row in reader:
            if not row:
                continue
            taxa.append(row[0])
            values.append(row[1] if len(row) > 1 else MISSING)
    return taxa, values


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-trees", required=True, help="NEXUS file supplying the taxa")
    parser.add_argument(
        "--source",
        choices=("label", "table"),
        help="where the states come from (default: table if --trait-data is given, else label)",
    )
    parser.add_argument("--trait-data", help="external metadata table (source table)")
    parser.add_argument(
        "--trait",
        help="name of the trait: the metadata column (source table) or a label "
        "for the trait (source label, default 'location')",
    )
    parser.add_argument("--taxon-column", help="name of the taxon column in the metadata")
    parser.add_argument("--state-column", help="name of the state column in the metadata")
    parser.add_argument("--delimiter", help="field delimiter of the metadata table")
    parser.add_argument(
        "--match-mode",
        choices=MATCH_MODES,
        default="exact",
        help="how a tree label is reduced to a metadata key (default: exact)",
    )
    parser.add_argument(
        "--label-delimiter", default="|", help="delimiter for --match-mode field"
    )
    parser.add_argument("--label-index", type=int, default=0, help="field index for --match-mode field")
    parser.add_argument("--taxon-regex", help="regex for --match-mode regex")
    parser.add_argument(
        "--states",
        help="comma separated state codes fixing the state order, e.g. '0,1'",
    )
    parser.add_argument(
        "--on-missing",
        choices=("unknown", "error"),
        default="unknown",
        help="what to do with taxa that have no observation (default: unknown -> '?')",
    )
    parser.add_argument("--output-prefix", required=True, help="output file prefix")
    parser.add_argument(
        "--require-variation", action="store_true", help="fail if the trait is constant"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    labels = extract_taxa_from_nexus(args.input_trees)
    states = [s.strip() for s in args.states.split(",")] if args.states else None

    source = args.source or ("table" if args.trait_data else "label")
    if source == "table" and not args.trait_data:
        raise TraitError("--source table needs --trait-data")

    if source == "label":
        assignment = assign_from_labels(
            labels,
            delimiter=args.label_delimiter,
            index=args.label_index,
            regex=args.taxon_regex,
            mode="regex" if args.taxon_regex else "field",
            trait=args.trait or "location",
            states=states,
        )
    else:
        table = read_trait_table(
            args.trait_data,
            trait=args.trait,
            taxon_column=args.taxon_column,
            state_column=args.state_column,
            delimiter=args.delimiter,
        )
        assignment = assign_traits(
            labels,
            table,
            mode=args.match_mode,
            delimiter=args.label_delimiter,
            index=args.label_index,
            regex=args.taxon_regex,
            states=states,
            on_missing=args.on_missing,
            require_variation=args.require_variation,
        )

    prefix = str(args.output_prefix)
    write_assignment(assignment, f"{prefix}.traits.tsv", f"{prefix}.traits.json")
    write_state_labels(assignment, f"{prefix}.state_labels.tsv")

    report = assignment.to_dict()
    remapped = {k: v for k, v in report["state_code_map"].items() if k != v}
    if remapped:
        print(
            f"note: numeric state codes are prefixed for BEAST {remapped} "
            "(BEAST 1.10's completeHistoryLogger cannot log numeric state codes)"
        )
    print(f"trait:  {report['trait']}")
    print(f"states: {report['states']}")
    print(f"taxa:   {report['n_taxa']} ({report['n_observed']} observed, {report['n_missing']} missing)")
    print(f"counts: {report['state_counts']}")
    if report["unmatched_observations"]:
        print(
            f"warning: {len(report['unmatched_observations'])} metadata rows matched no taxon, "
            f"e.g. {report['unmatched_observations'][:5]}",
            file=sys.stderr,
        )
    if assignment.missing:
        print(
            f"warning: {len(assignment.missing)} taxa coded as missing, "
            f"e.g. {assignment.missing[:5]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TraitError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
