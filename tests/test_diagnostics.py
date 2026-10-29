"""WP5/WP6: MCMC diagnostics."""
import numpy as np
import pandas as pd

from diagnostics import effective_sample_size, log_diagnostics, main


def test_ess_of_independent_samples_is_close_to_n():
    rng = np.random.default_rng(0)
    values = rng.normal(size=2000)
    ess = effective_sample_size(values)
    assert 1000 < ess <= 2000


def test_ess_of_autocorrelated_chain_is_much_smaller():
    rng = np.random.default_rng(0)
    values = [0.0]
    for _ in range(1999):
        values.append(0.95 * values[-1] + rng.normal(scale=0.3))
    ess = effective_sample_size(np.asarray(values))
    assert ess < 300


def test_ess_of_constant_series_is_undefined():
    # a constant column has no autocorrelation to estimate; returning n would
    # make a stuck parameter look like a passing one
    assert np.isnan(effective_sample_size(np.ones(100)))


def test_constant_column_does_not_pass(tmp_path):
    path = tmp_path / "constant.log"
    path.write_text(
        "state\tposterior\tstuck\n"
        + "".join(f"{i * 1000}\t{i}.0\t1.0\n" for i in range(1, 200))
    )
    table, _ = log_diagnostics(path, burnin_fraction=0.0)
    stuck = table.set_index("column").loc["stuck"]
    assert np.isnan(stuck["ess"])
    assert not stuck["passes"]


def test_ess_of_short_series():
    assert effective_sample_size([1.0, 2.0]) == 2.0


def write_log(tmp_path, rows=400, name="run.log"):
    rng = np.random.default_rng(1)
    path = tmp_path / name
    with open(path, "w") as handle:
        handle.write("# BEAST v1.10.4\n# generated\n# -overwrite x.xml\n")
        handle.write("state\tposterior\tCurrent Tree\tc_Into.s1[1]\n")
        for i in range(rows):
            handle.write(
                f"{i * 1000}\t{rng.normal():.6f}\tTree{i}\t{i % 3}\n"
            )
    return path


def test_log_diagnostics_skips_non_numeric_columns(tmp_path):
    table, info = log_diagnostics(write_log(tmp_path), burnin_fraction=0.0)
    assert set(table["column"]) == {"posterior", "c_Into.s1[1]"}
    assert "Current Tree" not in set(table["column"])
    assert info["ess_threshold"] == 200.0


def test_log_diagnostics_burnin_reduces_samples(tmp_path):
    table, _ = log_diagnostics(write_log(tmp_path), burnin_fraction=0.5)
    assert table["n_samples"].max() == 200


def test_log_diagnostics_threshold_flag(tmp_path):
    table, _ = log_diagnostics(write_log(tmp_path), burnin_fraction=0.0, ess_threshold=1e9)
    assert not table["passes"].any()


def test_cli_writes_table(tmp_path, capsys):
    log = write_log(tmp_path)
    output = tmp_path / "ess.tsv"
    assert main([str(log), "--output", str(output), "--burnin-fraction", "0"]) == 0
    table = pd.read_csv(output, sep="\t")
    assert {"column", "ess", "chain"} <= set(table.columns)
    assert "ESS threshold" in capsys.readouterr().out


def test_cli_warns_below_threshold(tmp_path, capsys):
    log = write_log(tmp_path)
    assert main([str(log), "--output", str(tmp_path / "ess.tsv"), "--ess-threshold", "1e9"]) == 0
    assert "below the ESS threshold" in capsys.readouterr().err
