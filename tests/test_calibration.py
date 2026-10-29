"""WP6: the calibration harness itself (scenarios and coverage accounting)."""
import pytest

from calibration import SCENARIOS, build_parser, summarise


def test_scenarios_are_well_formed():
    for name, options in SCENARIOS.items():
        assert options["root_state"] in ("0", "1"), name
        assert options["rates"], name
        assert 0.0 <= options.get("min_derived_fraction", 0.0) <= 1.0, name
        assert 0.0 <= options.get("max_derived_fraction", 1.0) <= 1.0, name


def test_scenarios_cover_the_documented_cases():
    assert {"single_gain", "recurrent_gain", "single_loss", "near_fixation", "sparse"} <= set(
        SCENARIOS
    )


def test_parser_defaults():
    args = build_parser().parse_args(["--output-prefix", "x"])
    assert args.replicates == 16
    assert args.beast_bin == "beast"


def record(**overrides):
    base = {
        "scenario": "single_gain",
        "replicate": 0,
        "direction": "0->1",
        "true_count": 1,
        "true_earliest": 2018.5,
        "true_root": "0",
        "derived_fraction": 0.5,
        "posterior_count_mean": 1.0,
        "posterior_count_lo95": 0.0,
        "posterior_count_hi95": 3.0,
        "p_positive": 0.7,
        "first_lo95": 2017.0,
        "first_hi95": 2019.5,
        "pooled_lo95": 2017.5,
        "pooled_hi95": 2020.0,
        "count_covered": True,
        "first_covered": True,
        "pooled_covered": True,
        "true_root_probability": 0.6,
        "count_check": 0.05,
    }
    base.update(overrides)
    return base


def test_summarise_computes_coverage():
    records = [
        record(replicate=0),
        record(replicate=1, first_covered=False, count_check=0.2),
    ]
    summary = summarise(records)["single_gain"]
    assert summary["n"] == 2
    assert summary["coverage_first_date"] == pytest.approx(0.5)
    assert summary["coverage_count"] == pytest.approx(1.0)
    assert summary["max_count_check_difference"] == pytest.approx(0.2)
    assert summary["mean_posterior_count"] == pytest.approx(1.0)


def test_summarise_handles_failed_replicates():
    records = [record(), {"scenario": "single_gain", "replicate": 1, "error": "boom"}]
    summary = summarise(records)["single_gain"]
    assert summary["n"] == 1
    assert summary["n_failed"] == 1


def test_summarise_all_failed():
    summary = summarise([{"scenario": "sparse", "replicate": 0, "error": "boom"}])["sparse"]
    assert summary == {"n": 1, "n_failed": 1}
