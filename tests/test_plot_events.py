"""WP7: row order, arrivals and the through-time density panel."""
import pandas as pd
import pytest

from plot_events import (
    first_acquisition,
    plot_arrivals,
    states_by_acquisition,
    states_by_mrca,
)


def summary(states, dates):
    return pd.DataFrame({"state": states, "date_median": dates})


def test_rows_follow_the_median_mrca_oldest_first():
    table = summary(
        ["Delta", "Alpha", "Gamma"], ["2020-05-01", "2019-12-01", "2021-03-01"]
    )
    assert states_by_mrca(table) == ["Alpha", "Delta", "Gamma"]


def test_equal_mrcas_keep_the_table_order():
    table = summary(["Zeta", "Alpha", "Mu"], ["2019-06-16"] * 3)
    assert states_by_mrca(table) == ["Zeta", "Alpha", "Mu"]


def test_states_without_a_median_mrca_go_last():
    table = summary(["Bheri", "Teku", "Koshi"], ["2019-06-16", "", "2018-01-01"])
    assert states_by_mrca(table) == ["Koshi", "Bheri", "Teku"]
    nan_table = summary(["A", "B"], [float("nan"), "2019-01-01"])
    assert states_by_mrca(nan_table) == ["B", "A"]

# --------------------------------------------------------------------------
# arrivals: when each state was entered
# --------------------------------------------------------------------------
def acquisitions(rows):
    return pd.DataFrame(
        rows,
        columns=["state", "p_acquired", "first_median", "first_lo95", "first_hi95"],
    )


def test_first_acquisition_reads_the_median_and_interval():
    table = acquisitions([["B", 0.4, 2015.5, 2013.0, 2019.0]])
    assert first_acquisition(table, "B") == (2015.5, 0.4, 2013.0, 2019.0)


def test_first_acquisition_is_empty_for_a_state_that_was_never_entered():
    table = acquisitions([["A", 0.0, float("nan"), float("nan"), float("nan")]])
    assert first_acquisition(table, "A") == (None, None, None, None)
    assert first_acquisition(table, "absent") == (None, None, None, None)
    assert first_acquisition(None, "A") == (None, None, None, None)


def test_rows_are_ordered_by_first_acquisition_oldest_first():
    table = acquisitions(
        [
            ["late", 1.0, 2019.0, 2018.0, 2020.0],
            ["never", 0.0, float("nan"), float("nan"), float("nan")],
            ["early", 1.0, 2015.0, 2014.0, 2016.0],
        ]
    )
    assert states_by_acquisition(table, ["late", "never", "early"]) == [
        "early", "late", "never",
    ]


def test_arrivals_renders_and_needs_dates(tmp_path):
    prevalence = pd.DataFrame(
        {
            "date": [2014.0, 2015.0, 2016.0],
            "state": ["A"] * 3,
            "proportion_mean": [10.0, 20.0, 30.0],
            "proportion_lo95": [5.0, 10.0, 20.0],
            "proportion_hi95": [15.0, 30.0, 40.0],
        }
    )
    events = pd.DataFrame({"dest": ["A", "A"], "date": [2015.2, 2015.8]})
    table = acquisitions([["A", 1.0, 2015.4, 2015.1, 2015.9]])
    output = tmp_path / "arrivals.png"
    plot_arrivals(prevalence, events, table, output)
    assert output.exists() and output.stat().st_size > 0

    with pytest.raises(ValueError, match="--mrsd"):
        plot_arrivals(prevalence, events.drop(columns=["date"]), table, output)
