"""WP4: parsing and summarising complete transition histories."""
import numpy as np
import pytest

from dates import decimal_year
from jump_history import (
    JumpHistoryError,
    JumpSample,
    acquisition_summary,
    date_summary,
    direction_counts,
    events_frame,
    expected_directions,
    parse_jump_history,
    read_count_registers,
    read_root_states,
)

HEADER = """# BEAST v1.10.4 Prerelease #bc6cbd9
# Generated today
# -overwrite run.xml
state\ttotalChangeCount\tcompleteHistory_1
"""


def write_log(tmp_path, rows, name="run.jumpHistory.log"):
    path = tmp_path / name
    path.write_text(HEADER + "".join(rows))
    return path


ROWS = [
    "0\t0\t{} 0\n",
    "1000\t1\t{{1,2.0,s0,s1}} 1\n",
    "2000\t2\t{{1,2.5,s0,s1},{1,0.5,s1,s0}} 2\n",
    "3000\t0\t{} 0\n",
]


def test_parse_basic(tmp_path):
    samples, info = parse_jump_history(write_log(tmp_path, ROWS), burnin_states=0)
    assert [s.state for s in samples] == [0, 1000, 2000, 3000]
    assert samples[0].events == []
    assert samples[1].events == [(1, 2.0, "s0", "s1")]
    assert samples[2].events == [(1, 2.5, "s0", "s1"), (1, 0.5, "s1", "s0")]
    assert info["n_events_total"] == 3
    assert info["samples_with_count_mismatch"] == []


def test_burnin_fraction_keeps_the_tail(tmp_path):
    samples, info = parse_jump_history(write_log(tmp_path, ROWS), burnin_fraction=0.5)
    assert info["burnin_states"] == 1500
    assert [s.state for s in samples] == [2000, 3000]


def test_parse_reports_count_mismatch(tmp_path):
    path = write_log(tmp_path, ["1000\t5\t{{1,2.0,s0,s1}} 1\n"])
    samples, info = parse_jump_history(path, burnin_states=0)
    assert info["samples_with_count_mismatch"] == [1000]
    assert len(samples[0].events) == 1


def test_parse_missing_file(tmp_path):
    with pytest.raises(JumpHistoryError, match="not found"):
        parse_jump_history(tmp_path / "absent.log")


def test_parse_empty_file(tmp_path):
    path = tmp_path / "empty.log"
    path.write_text(HEADER)
    with pytest.raises(JumpHistoryError, match="no samples"):
        parse_jump_history(path)


def test_parse_malformed_line(tmp_path):
    path = write_log(tmp_path, ["1000\t1\n"])
    with pytest.raises(JumpHistoryError, match="malformed"):
        parse_jump_history(path, burnin_states=0)


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
@pytest.fixture
def samples(tmp_path):
    return parse_jump_history(write_log(tmp_path, ROWS), burnin_states=0)[0]


def test_direction_counts_includes_zero_samples(samples):
    counts = direction_counts(samples)
    by_direction = counts.set_index("direction")
    assert by_direction.loc["s0->s1", "n_events"] == 2
    assert by_direction.loc["s0->s1", "mean"] == pytest.approx(0.5)
    assert by_direction.loc["s0->s1", "p_positive"] == pytest.approx(0.5)
    assert by_direction.loc["s1->s0", "n_events"] == 1
    assert by_direction.loc["s1->s0", "max"] == 1


def test_direction_counts_maps_state_codes_back(samples):
    counts = direction_counts(samples, code_map={"0": "s0", "1": "s1"})
    assert set(counts["direction"]) == {"0->1", "1->0"}
    assert set(counts["source"]) == {"0", "1"}


def test_direction_counts_with_no_events(tmp_path):
    samples = parse_jump_history(write_log(tmp_path, ["0\t0\t{} 0\n"]), burnin_states=0)[0]
    counts = direction_counts(samples)
    assert counts.empty


def test_events_frame_dates_and_mapping(samples):
    frame = events_frame(samples, mrsd="2020-01-01", code_map={"0": "s0", "1": "s1"})
    assert list(frame["direction"]) == ["0->1", "0->1", "1->0"]
    # heights are years before the MRSD
    assert list(frame["height"]) == [2.0, 2.5, 0.5]
    assert list(frame["source"]) == ["0", "0", "1"]
    assert frame.iloc[0]["date"] == pytest.approx(2018.0, abs=0.01)
    assert frame.iloc[0]["date_iso"].startswith("2018-")
    assert frame.iloc[1]["date_iso"].startswith("2017-")


def test_events_frame_without_mrsd(samples):
    frame = events_frame(samples)
    assert "date" not in frame.columns
    assert len(frame) == 3


def test_date_summary_first_and_last(samples):
    dates = date_summary(samples, mrsd="2020-01-01").set_index("direction")
    row = dates.loc["s0->s1"]
    assert row["n_events"] == 2
    assert row["n_samples_with_event"] == 2
    # gains at heights 2.0 (sample 1000) and 2.5 (sample 2000) -> dates 2018.0 / 2017.5
    assert row["first_median"] == pytest.approx(2017.75, abs=0.01)
    assert row["last_median"] == pytest.approx(2017.75, abs=0.01)
    assert row["pooled_median"] == pytest.approx(2017.75, abs=0.01)
    assert row["first_lo95"] == pytest.approx(2017.5, abs=0.05)
    assert row["first_hi95"] == pytest.approx(2018.0, abs=0.05)


def test_date_summary_never_observed_direction(samples):
    dates = date_summary(samples, mrsd="2020-01-01", directions=["s0->s1", "s1->s0", "s0->s0"])
    never = dates.set_index("direction").loc["s0->s0"]
    assert never["n_events"] == 0
    assert np.isnan(never["first_median"])


def test_date_summary_needs_mrsd(samples):
    with pytest.raises(JumpHistoryError, match="--mrsd"):
        date_summary(samples)


# --------------------------------------------------------------------------
# root states
# --------------------------------------------------------------------------
TREES = """#NEXUS

Begin taxa;
\tDimensions ntax=3;
\tTaxlabels
\t\ta
\t\tb
\t\tc
\t;
End;
Begin trees;
tree STATE_0 = [&R] ((a[&loc="x"]:1.0,b[&loc="x"]:1.0)[&loc="x"]:1.0,c[&loc="y"]:2.0)[&loc="x"];
tree STATE_100 = [&R] ((a[&loc="y"]:1.0,b[&loc="y"]:1.0)[&loc="y"]:1.0,c[&loc="y"]:2.0)[&loc="y"];
tree STATE_200 = [&R] ((a[&loc="x"]:1.0,b[&loc="x"]:1.0)[&loc="x"]:1.0,c[&loc="y"]:2.0)[&loc="x"];
End;
"""


def test_read_root_states(tmp_path):
    path = tmp_path / "run.trees"
    path.write_text(TREES)
    table, info = read_root_states(path, tag="loc", burnin_fraction=0.0)
    by_state = table.set_index("state")
    assert by_state.loc["x", "n"] == 2
    assert by_state.loc["x", "probability"] == pytest.approx(2 / 3)
    assert info["n_trees_kept"] == 3


def test_read_root_states_burnin(tmp_path):
    path = tmp_path / "run.trees"
    path.write_text(TREES)
    table, info = read_root_states(path, tag="loc", burnin_trees=1)
    assert info["n_trees_kept"] == 2
    by_state = table.set_index("state")
    assert by_state.loc["y", "n"] == 1
    assert by_state.loc["x", "n"] == 1


def test_read_root_states_unknown_tag(tmp_path):
    path = tmp_path / "run.trees"
    path.write_text(TREES)
    with pytest.raises(JumpHistoryError, match="no root state"):
        read_root_states(path, tag="nope")


# --------------------------------------------------------------------------
# cross-check
# --------------------------------------------------------------------------
def test_read_count_registers(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(
        "state\tposterior\tc_Into.s1[1]\tc_cfrD.s0.s1[1]\n"
        "0\t-1\t0\t0\n"
        "100\t-2\t1\t1\n"
    )
    frame = read_count_registers(path, "cfrD")
    assert list(frame.columns) == ["state", "c_Into.s1[1]", "c_cfrD.s0.s1[1]"]
    assert frame["c_cfrD.s0.s1[1]"].mean() == 0.5
    assert len(read_count_registers(path, "cfrD", burnin_states=50)) == 1


def test_summarize_cli_merges_chains(tmp_path):
    """Two chains must be summarised together (one --jump-history per chain)."""
    import json

    from summarize_jumps import main

    write_log(tmp_path, ROWS, name="chain1.jumpHistory.log")
    write_log(tmp_path, ROWS, name="chain2.jumpHistory.log")
    traits = tmp_path / "traits.json"
    traits.write_text(
        json.dumps(
            {
                "trait": "cfrD",
                "states": ["0", "1"],
                "state_code_map": {"0": "s0", "1": "s1"},
            }
        )
    )
    prefix = tmp_path / "out"
    assert main([
        "--jump-history",
        str(tmp_path / "chain1.jumpHistory.log"),
        str(tmp_path / "chain2.jumpHistory.log"),
        "--trait-json", str(traits),
        "--tag", "cfrD",
        "--prefix-id", "cfrD",
        "--mrsd", "2020-01-01",
        "--output-prefix", str(prefix),
    ]) == 0

    summary = json.loads((tmp_path / "out.summary.json").read_text())
    assert summary["jump_history"]["n_chains"] == 2
    assert summary["jump_history"]["n_samples_kept"] == 6
    counts = {row["direction"]: row for row in summary["directions"]}
    assert counts["0->1"]["n_events"] == 4
    assert counts["0->1"]["p_positive"] == pytest.approx(4 / 6)


def test_read_count_registers_without_registers(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("state\tposterior\n0\t-1\n")
    with pytest.raises(JumpHistoryError, match="no c_"):
        read_count_registers(path, "cfrD")


# --------------------------------------------------------------------------
# direction coverage and acquisitions
# --------------------------------------------------------------------------
def test_expected_directions_lists_every_ordered_pair():
    directions = expected_directions(["0", "1", "2"])
    assert len(directions) == 6
    assert set(directions) == {
        "0->1", "0->2", "1->0", "1->2", "2->0", "2->1",
    }


def test_direction_counts_include_a_direction_that_never_occurred(tmp_path):
    samples = parse_jump_history(
        write_log(tmp_path, ["1000\t1\t{{1,2.0,s0,s1}} 1\n"]), burnin_states=0
    )[0]
    counts = direction_counts(samples, directions=expected_directions(["s0", "s1"]))
    by_direction = counts.set_index("direction")
    assert by_direction.loc["s1->s0", "n_events"] == 0
    assert by_direction.loc["s1->s0", "p_positive"] == 0.0
    assert by_direction.loc["s0->s1", "n_events"] == 1


def test_acquisition_takes_the_earliest_entry_from_any_source():
    samples = [
        JumpSample(0, 2, [(1, 5.0, "s0", "s2"), (1, 8.0, "s1", "s2")], chain=1),
        JumpSample(1000, 1, [(1, 6.0, "s0", "s2")], chain=1),
    ]
    mrsd_year = decimal_year("2020-01-01")
    table = acquisition_summary(samples, mrsd="2020-01-01", states=["s2"]).set_index("state")
    # heights 8 and 6 are the oldest events into s2 in the two samples, and a
    # date is mrsd - height
    assert table.loc["s2", "p_acquired"] == 1.0
    assert table.loc["s2", "first_median"] == pytest.approx(mrsd_year - 7)
    assert table.loc["s2", "first_lo95"] < mrsd_year - 7 < table.loc["s2", "first_hi95"]


def test_acquisition_is_zero_for_a_state_that_is_never_entered():
    samples = [JumpSample(0, 1, [(1, 5.0, "s0", "s1")], chain=1)]
    table = acquisition_summary(samples, states=["s0", "s1"]).set_index("state")
    assert table.loc["s0", "p_acquired"] == 0.0
    assert np.isnan(table.loc["s0", "first_median"])
    assert table.loc["s1", "p_acquired"] == 1.0


# --------------------------------------------------------------------------
# per-chain burn-in
# --------------------------------------------------------------------------
def test_read_root_states_burnin_runs_per_file(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    first.write_text(TREES)
    second.write_text(TREES)
    table, info = read_root_states([first, second], tag="loc", burnin_fraction=0.5)
    by_state = table.set_index("state")
    # one tree is dropped from the head of each file: the leading x from both
    assert info["n_trees_kept"] == 4
    assert by_state.loc["x", "n"] == 2
    assert by_state.loc["y", "n"] == 2
