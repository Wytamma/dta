#!/usr/bin/env python3
"""Render a BEAST 1.10 XML for discrete trait analysis (Stage 1, WP2 + WP3).

A trait is any categorical genome feature.  The states always arrive as a
normalised table written by ``trait_data.py`` (``--traits``), whatever their
source: an external annotation sheet or the taxon labels themselves.  The trait
name becomes the BEAST id prefix (``--id-prefix``), so a label-sourced trait
called ``location`` reproduces the historical ``location.*`` / ``Location.*``
identifiers.

Model options
-------------
``equal``       symmetric CTMC: relative rates fixed at 1, the overall trait
                rate is carried by the clock rate (this is the ER/symmetric
                model).
``asymmetric``  every ordered transition gets its own rate.
``bssvs``       asymmetric + Bayesian stochastic search variable selection
                (rate indicators with a Poisson prior on the number of
                non-zero rates).  This is the legacy geographic default.

Markov jump registers are always emitted: ``Into.<state>``, ``OutOf.<state>``,
``total`` and one register per ordered pair.  For a binary trait ``Into.1`` and
``OutOf.1`` are exactly the directional acquisition and loss counts.

With ``--log-complete-history`` the XML additionally enables BEAST's
uniformization-based stochastic mapping and a ``completeHistoryLogger``, which
writes every individual transition (origin, destination and height along the
tree) to ``<prefix>.jumpHistory.log``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from jinja2 import Template

from trait_data import (
    MISSING,
    TraitError,
    extract_taxa_from_nexus,
    natural_sort_key,
    read_assignment,
    to_beast_code,
)

MODELS = ("equal", "asymmetric", "bssvs")
ROOT_PRIORS = ("uniform", "empirical")


# --------------------------------------------------------------------------
# model building
# --------------------------------------------------------------------------
def build_registers(states: list[str], pair_prefix: str):
    """Markov jump count registers and Markov reward registers.

    The flat vectors follow the state order of ``states`` row-major, which is
    the ordering BEAST expects for a ``generalSubstitutionModel`` rate matrix.
    """
    n = len(states)
    matrices: dict[str, str] = {}
    rewards: dict[str, str] = {}

    for i, source in enumerate(states):
        into = [0] * (n * n)
        outof = [0] * (n * n)
        for j, dest in enumerate(states):
            if i == j:
                continue
            into[j * n + i] = 1
            outof[i * n + j] = 1
        matrices[f"Into.{source}"] = " ".join(str(v) for v in into)
        matrices[f"OutOf.{source}"] = " ".join(str(v) for v in outof)

        for j, dest in enumerate(states):
            if i == j:
                continue
            single = [0] * (n * n)
            single[i * n + j] = 1
            matrices[f"{pair_prefix}.{source}.{dest}"] = " ".join(str(v) for v in single)

        reward = [0] * n
        reward[i] = 1
        rewards[f"rewards_{source}"] = " ".join(str(v) for v in reward)

    matrices["total"] = " ".join("0" if k // n == k % n else "1" for k in range(n * n))
    return matrices, rewards


def bssvs_prior_mean(rate_count: int, configured: float | None) -> float:
    """Mean of the Poisson prior on the number of non-zero rates.

    With ``rate_count = k(k - 1)`` possible ordered routes, the default puts the
    expected number of routes at half of them (``k(k - 1) / 2``), which is the
    diffuse prior recommended by Gao et al. (2023) and what PrioriTree
    specifies.  A fixed value would be informative in a way that depends on the
    number of states: BEAST's own asymmetric default is ``k - 1``, which is the
    sparse prior those authors warn against.
    """
    return float(configured) if configured is not None else rate_count / 2.0


def build_model_config(model: str, args, rate_count: int) -> dict:
    mean = bssvs_prior_mean(rate_count, args.bssvs_prior_mean)
    if model == "equal":
        return dict(
            estimate_rates=False,
            use_indicators=False,
            rate_prior_shape=args.rate_prior_shape,
            rate_prior_scale=args.rate_prior_scale,
            bssvs_mean=mean,
        )
    if model == "asymmetric":
        return dict(
            estimate_rates=True,
            use_indicators=False,
            rate_prior_shape=args.rate_prior_shape,
            rate_prior_scale=args.rate_prior_scale,
            bssvs_mean=mean,
        )
    if model == "bssvs":
        return dict(
            estimate_rates=True,
            use_indicators=True,
            rate_prior_shape=args.rate_prior_shape,
            rate_prior_scale=args.rate_prior_scale,
            bssvs_mean=mean,
        )
    raise TraitError(f"unknown trait model {model!r} (expected one of {MODELS})")


def build_root_prior(spec: str, states: list[str], values: list[str]) -> dict:
    """Root state prior: ``uniform``, ``empirical`` or ``fixed:<state>``."""
    if spec == "uniform":
        return dict(
            type="uniform",
            estimated=True,
            state=None,
            vector=" ".join("1.0" for _ in states),
        )
    if spec == "empirical":
        counts = [0] * len(states)
        for value in values:
            if value != MISSING:
                counts[states.index(value)] += 1
        total = sum(counts)
        frequencies = [c / total for c in counts] if total else [1.0] * len(states)
        return dict(
            type="empirical",
            estimated=True,
            state=None,
            vector=" ".join(f"{f:g}" for f in frequencies),
        )
    if spec.startswith("fixed:"):
        state = spec.split(":", 1)[1]
        if state not in states:
            raise TraitError(f"fixed root state {state!r} is not one of {states}")
        values_out = ["1.0" if s == state else "0.0" for s in states]
        return dict(type="fixed", estimated=False, state=state, vector=" ".join(values_out))
    raise TraitError(
        f"unknown root prior {spec!r} (expected one of {ROOT_PRIORS} or fixed:<state>)"
    )


def sanitise_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"trait_{cleaned}"
    return cleaned


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------
def load_states_and_values(args, labels: list[str]):
    """Return (taxa, values, states) in tree order, read from the trait table."""
    table_taxa, values = read_assignment(args.traits)
    if len(set(table_taxa)) != len(table_taxa):
        raise TraitError(f"{args.traits}: duplicate taxon rows")
    unknown = sorted(set(table_taxa) - set(labels))
    if unknown:
        raise TraitError(
            f"{args.traits}: {len(unknown)} taxa are not in the tree, e.g. {unknown[:5]}"
        )
    absent = sorted(set(labels) - set(table_taxa))
    if absent:
        raise TraitError(
            f"{args.traits}: {len(absent)} tree taxa have no row, e.g. {absent[:5]}; "
            "re-run trait_data.py"
        )
    if table_taxa != labels:
        print(
            "warning: trait table order differs from the tree order; using the tree order",
            file=sys.stderr,
        )
        lookup = dict(zip(table_taxa, values))
        values = [lookup[taxon] for taxon in labels]
    taxa = list(labels)

    states = (
        [state.strip() for state in args.state_order.split(",")]
        if args.state_order
        else sorted(set(values) - {MISSING}, key=natural_sort_key)
    )
    if len(states) < 2:
        raise TraitError(f"need at least two observed states, got {states!r}")
    unknown = sorted((set(values) - {MISSING}) - set(states))
    if unknown:
        raise TraitError(f"observed states {unknown} are not in the declared order {states}")

    if args.state_code_style == "safe":
        remapped = {code: to_beast_code(code) for code in states if to_beast_code(code) != code}
        if remapped:
            print(
                f"note: using BEAST-safe state codes {remapped} "
                "(BEAST 1.10's completeHistoryLogger cannot log numeric state codes)",
                file=sys.stderr,
            )
        values = [to_beast_code(value) if value != MISSING else value for value in values]
        states = [to_beast_code(state) for state in states]
    return taxa, values, states


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def resolve_id_prefix(args) -> str:
    """BEAST id prefix (attribute name, tree tag, parameter ids).

    Kept separate from the file prefix, which the Snakefile sets to the run
    name; the id prefix is the trait name.
    """
    return sanitise_name(args.id_prefix or Path(args.name).stem)


def render_xml(args, taxa, values, states) -> str:
    name = resolve_id_prefix(args)
    pair_prefix = args.pair_prefix or name
    matrices, rewards = build_registers(states, pair_prefix)
    rate_count = len(states) * (len(states) - 1)
    model = build_model_config(args.trait_model, args, rate_count)
    root_prior = build_root_prior(args.root_state, states, values)

    template_path = Path(args.input_template)
    if not template_path.exists():
        raise TraitError(f"template not found: {template_path}")
    template = Template(template_path.read_text())

    return template.render(
        name=name,
        taxa=[{"id": t, "state": v, "location": v} for t, v in zip(taxa, values)],
        states=states,
        location_codes=states,
        state_count=len(states),
        rate_count=rate_count,
        rate_values=" ".join("1.0" for _ in range(rate_count)),
        matrices=matrices,
        rewards=rewards,
        model=model,
        root_prior=root_prior,
        clock_rate_init=args.clock_rate,
        clock_prior_shape=args.clock_prior_shape,
        clock_prior_scale=args.clock_prior_scale,
        log_complete_history=bool(args.log_complete_history),
        history_tag=args.history_tag,
        # BEAST resolves file names relative to the XML file, so the bare file
        # name is correct even for a nested output prefix.
        tree_file=Path(args.input_trees).name,
        log_filename=f"{args.name}.log",
        tree_log_filename=f"{args.name}.trees",
        jump_history_filename=f"{args.name}.jumpHistory.log",
        chain_length=args.chain_length,
        sample_every=args.sample_every,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_template", required=True, help="Jinja2 BEAST template")
    parser.add_argument("--input_trees", required=True, help="NEXUS trees file")
    parser.add_argument("--output_xml", required=True, help="XML file to write")
    parser.add_argument("--name", default="location", help="output file prefix")
    parser.add_argument(
        "--id-prefix",
        help="BEAST id prefix (attribute name, tree tag, parameter ids); "
        "defaults to the trait name",
    )
    parser.add_argument("--chain_length", type=int, default=50000000)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--traits", required=True, help="normalised trait table")
    parser.add_argument(
        "--state-order", help="comma separated state codes fixing the CTMC ordering"
    )
    parser.add_argument(
        "--trait-model",
        choices=MODELS,
        default="asymmetric",
        help="equal | asymmetric | bssvs (default: asymmetric, as in config/config.yaml)",
    )
    parser.add_argument(
        "--state-code-style",
        choices=("safe", "as-is"),
        default="safe",
        help="rewrite numeric state codes so BEAST's completeHistoryLogger can log them",
    )
    parser.add_argument(
        "--root-state",
        default="uniform",
        help="uniform | empirical | fixed:<state> (default: uniform)",
    )
    parser.add_argument("--pair-prefix", help="prefix for the ordered pair jump registers")
    parser.add_argument("--clock-rate", type=float, default=0.01, help="initial transition rate")
    parser.add_argument("--clock-prior-shape", type=float, default=0.5)
    parser.add_argument("--clock-prior-scale", type=float, default=2.0)
    parser.add_argument("--rate-prior-shape", type=float, default=1.0)
    parser.add_argument("--rate-prior-scale", type=float, default=1.0)
    parser.add_argument(
        "--bssvs-prior-mean",
        type=float,
        default=None,
        help="mean of the Poisson prior on the number of non-zero rates "
        "(default: half of the k(k-1) ordered routes, i.e. the diffuse prior of "
        "Gao et al. 2023; BEAST's own asymmetric default is k-1)",
    )
    parser.add_argument(
        "--log-complete-history",
        action="store_true",
        help="enable the completeHistoryLogger (individual transition events)",
    )
    parser.add_argument(
        "--history-tag",
        default="branch.history",
        help="tree annotation name for the sampled per-branch histories "
        "(must match trait_trees.HISTORY_TAG)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    labels = extract_taxa_from_nexus(args.input_trees)
    taxa, values, states = load_states_and_values(args, labels)
    xml = render_xml(args, taxa, values, states)
    with open(args.output_xml, "w") as fh:
        fh.write(xml)

    counts = {s: values.count(s) for s in states}
    print(
        f"wrote {args.output_xml}: id_prefix={resolve_id_prefix(args)} "
        f"model={args.trait_model} root={args.root_state} states={counts} "
        f"missing={values.count(MISSING)} complete_history={bool(args.log_complete_history)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TraitError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
