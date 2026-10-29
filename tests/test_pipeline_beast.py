"""WP6: end-to-end recovery of a simulated single gain.

Simulates a trait history with exactly one acquisition, runs the real pipeline
(trait preprocessing -> XML -> BEAST -> summaries) and checks that the
acquisition is recovered: its date must fall inside the posterior 95% interval
and the complete-history counts must agree with BEAST's Markov jump registers.

Needs BEAST on PATH (or ``BEAST_BIN``); run with ``pytest -m slow``.
"""
import json
import subprocess
from pathlib import Path

import pytest

from generate_beast_xml import main as xml_main
from simulate_trait import main as simulate_main
from summarize_jumps import main as summarize_main
from trait_data import main as traits_main

pytestmark = pytest.mark.slow

CHAIN_LENGTH = 200000
SAMPLE_EVERY = 2000


@pytest.fixture(scope="module")
def analysis(tmp_path_factory, beast_bin, template_path):
    """Run the whole pipeline once and return the artefacts."""
    directory = tmp_path_factory.mktemp("beast_run")
    prefix = directory / "run"

    assert simulate_main([
        "--output-prefix", str(prefix),
        "--n-tips", "24", "--tree-height", "3.0",
        "--states", "0,1", "--model", "asymmetric", "--rates", "0.08,0.000001",
        "--root-state", "0", "--n-events", "1",
        "--mrsd", "2020-01-01", "--seed", "7",
    ]) == 0

    assert traits_main([
        "--input-trees", f"{prefix}.trees",
        "--trait-data", f"{prefix}.traits.csv",
        "--trait", "state", "--match-mode", "exact",
        "--output-prefix", str(prefix),
    ]) == 0

    assert xml_main([
        "--input_template", str(template_path),
        "--input_trees", f"{prefix}.trees",
        "--output_xml", f"{prefix}.xml",
        "--traits", f"{prefix}.traits.tsv",
        "--id-prefix", "cfrD", "--trait-model", "asymmetric",
        "--log-complete-history",
        "--chain_length", str(CHAIN_LENGTH), "--sample_every", str(SAMPLE_EVERY),
        "--name", "run",
    ]) == 0

    completed = subprocess.run(
        [beast_bin, "-seed", "42", "-overwrite", "run.xml"],
        cwd=directory, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout[-2000:]
    assert "Exception" not in completed.stdout

    assert summarize_main([
        "--jump-history", f"{prefix}.jumpHistory.log",
        "--trees", f"{prefix}.trees",
        "--log", f"{prefix}.log",
        "--trait-json", f"{prefix}.traits.json",
        "--tag", "cfrD", "--prefix-id", "cfrD",
        "--mrsd", "2020-01-01", "--burnin-fraction", "0.1",
        "--output-prefix", str(prefix),
    ]) == 0

    return {
        "directory": directory,
        "prefix": prefix,
        "truth": [
            line.split("\t")
            for line in Path(f"{prefix}.truth.tsv").read_text().splitlines()[1:]
        ],
        "summary": json.loads(Path(f"{prefix}.summary.json").read_text()),
        "counts": _read_tsv(f"{prefix}.event_counts.tsv"),
        "dates": _read_tsv(f"{prefix}.event_dates.tsv"),
        "events": _read_tsv(f"{prefix}.events.tsv"),
        "root": _read_tsv(f"{prefix}.root_states.tsv"),
    }


def _read_tsv(path):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_single_gain_is_recovered(analysis):
    counts = {row["direction"]: row for row in analysis["counts"]}
    assert "0->1" in counts
    assert float(counts["0->1"]["p_positive"]) > 0.5
    # the simulated history has exactly one transition
    assert float(counts["0->1"]["mean"]) == pytest.approx(1.0, abs=1.0)


def test_true_gain_date_is_inside_the_credible_interval(analysis):
    truth = analysis["truth"][0]
    assert truth[4] == "0" and truth[5] == "1"
    true_date = float(truth[1])  # height in years before the MRSD
    true_year = 2020.0 - true_date

    dates = {row["direction"]: row for row in analysis["dates"]}
    row = dates["0->1"]
    assert float(row["first_lo95"]) <= true_year <= float(row["first_hi95"])
    assert float(row["pooled_lo95"]) <= true_year <= float(row["pooled_hi95"])


def test_events_table_matches_the_summary(analysis):
    events = analysis["events"]
    gains = [row for row in events if row["direction"] == "0->1"]
    counts = {row["direction"]: row for row in analysis["counts"]}
    assert len(gains) == int(counts["0->1"]["n_events"])
    for row in gains:
        assert row["source"] == "0" and row["dest"] == "1"
        assert 0.0 <= float(row["height"]) <= 3.0
        assert row["date_iso"].startswith("20")


def test_complete_history_agrees_with_jump_registers(analysis):
    check = analysis["summary"]["count_check"]
    assert check["max_abs_difference"] < 0.25
    assert len(check["table"]) == 2


def test_root_states_are_reported(analysis):
    root = analysis["root"]
    assert len(root) == 2
    assert sum(float(row["probability"]) for row in root) == pytest.approx(1.0)
    assert {row["label"] for row in root} == {"0", "1"}


def test_no_count_mismatches(analysis):
    assert analysis["summary"]["jump_history"]["samples_with_count_mismatch"] == []
