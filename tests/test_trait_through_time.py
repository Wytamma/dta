"""WP7: the prevalence / TMRCA command line tool."""
import json

import pytest

from trait_through_time import build_parser, collect_trees, main, state_appears

TREES = """#NEXUS

Begin taxa;
\tDimensions ntax=4;
\tTaxlabels
\t\ta
\t\tb
\t\tc
\t\td
\t;
End;

Begin trees;
\tTranslate
\t\t1 'a',
\t\t2 'b',
\t\t3 'c',
\t\t4 'd'
\t;
tree STATE_0 = [&R] ((1[&st="s1"]:1,2[&st="s1"]:1)[&st="s1"]:[&branch.history={{2.5,s0,s1}}]2,(3[&st="s0"]:1,4[&st="s0"]:1)[&st="s0"]:2)[&st="s0"]:0;
tree STATE_1000 = [&R] ((1[&st="s1"]:1,3[&st="s0"]:1)[&st="s0"]:2,(2[&st="s1"]:1,4[&st="s0"]:1)[&st="s1"]:2)[&st="s0"]:0;
End;
"""

TRAITS = {
    "trait": "cfrD",
    "states": ["0", "1"],
    "state_code_map": {"0": "s0", "1": "s1"},
}


@pytest.fixture
def inputs(tmp_path):
    trees = tmp_path / "run.trees"
    trees.write_text(TREES)
    traits = tmp_path / "run.traits.json"
    traits.write_text(json.dumps(TRAITS))
    return trees, traits, tmp_path / "out"


def read_table(path):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_parser_requires_mrsd():
    args = build_parser().parse_args(["--trees", "x", "--output-prefix", "y", "--tag", "st", "--mrsd", "2020-01-01"])
    assert args.mrsd == "2020-01-01"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--trees", "x", "--output-prefix", "y", "--tag", "st"])


def test_collect_trees_burnin_and_thinning(inputs):
    trees, _, _ = inputs
    kept = collect_trees(trees, burnin_fraction=0.5, max_trees=10)
    assert [number for _, _, number in kept] == [1000]


def test_state_appears_sees_history_and_tips(inputs):
    from trait_trees import read_trees

    trees, _, _ = inputs
    root = next(iter(read_trees(trees)))[1]
    assert state_appears(root, "st", ["s0", "s1"]) == {"s0", "s1"}
    assert state_appears(root, "st", ["s2"]) == set()


def test_cli_writes_prevalence_and_tmrca(inputs, capsys):
    trees, traits, prefix = inputs
    assert main([
        "--trees", str(trees),
        "--tag", "st",
        "--trait-json", str(traits),
        "--mrsd", "2020-01-01",
        "--burnin-fraction", "0",
        "--output-prefix", str(prefix),
    ]) == 0

    prevalence = read_table(prefix.with_name("out.prevalence.tsv"))
    assert {row["state"] for row in prevalence} == {"0", "1"}  # BEAST codes mapped back
    newest = [row for row in prevalence if row["state"] == "1"][-1]
    assert float(newest["proportion_mean"]) == pytest.approx(0.5)
    assert float(newest["proportion_lo95"]) == pytest.approx(0.5)
    assert float(newest["n_lineages_mean"]) == pytest.approx(4.0)
    oldest = [row for row in prevalence if row["state"] == "1"][0]
    assert float(oldest["n_lineages_mean"]) == pytest.approx(2.0)

    tmrca = {row["state"]: row for row in read_table(prefix.with_name("out.tmrca.tsv"))}
    assert float(tmrca["1"]["n_tips"]) == pytest.approx(2.0)
    assert float(tmrca["1"]["appears_in"]) == pytest.approx(1.0)
    # tree 1 puts the carrier MRCA one year below the root, tree 2 at the root
    assert float(tmrca["1"]["tmrca_mean"]) == pytest.approx(2.0)
    assert float(tmrca["1"]["tmrca_lo95"]) == pytest.approx(1.05)
    assert float(tmrca["1"]["tmrca_hi95"]) == pytest.approx(2.95)
    assert tmrca["1"]["date_median"] == "2018-01-01"
    # tree 1 puts the carrier MRCA below the root, tree 2 at the root
    assert float(tmrca["1"]["at_root"]) == pytest.approx(0.5)
    assert float(tmrca["1"]["tree_tmrca"]) == pytest.approx(3.0)

    samples = read_table(prefix.with_name("out.tmrca_samples.tsv"))
    assert sorted({row["state"] for row in samples}) == ["0", "1"]
    assert len(samples) == 4  # two states in two trees

    assert "MRCA" in capsys.readouterr().out


def test_cli_warns_without_branch_histories(inputs, capsys):
    trees, traits, prefix = inputs
    text = trees.read_text().replace(":[&branch.history={{2.5,s0,s1}}]2", ":2")
    trees.write_text(text)
    assert main([
        "--trees", str(trees),
        "--tag", "st",
        "--trait-json", str(traits),
        "--mrsd", "2020-01-01",
        "--burnin-fraction", "0",
        "--output-prefix", str(prefix),
    ]) == 0
    assert "no per-branch histories" in capsys.readouterr().err
