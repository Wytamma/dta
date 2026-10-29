"""WP6: simulated trait histories with known events."""
import json
import random
from pathlib import Path

import pytest

from simulate_trait import (
    SimulationError,
    build_rate_matrix,
    coalescent_tree,
    main,
    node_distances,
    parse_newick,
    read_tree,
    simulate_history,
    tip_nodes,
    to_newick,
    tree_height,
)


def test_coalescent_tree_is_ultrametric():
    root = coalescent_tree(30, 2.5, random.Random(1))
    distances = node_distances(root)
    tips = [distances[id(tip)] for tip in tip_nodes(root)]
    assert max(tips) == pytest.approx(2.5, abs=1e-9)
    assert max(tips) - min(tips) < 1e-9


def test_newick_round_trip():
    root = coalescent_tree(8, 1.0, random.Random(3))
    text = to_newick(root) + ";"
    reparsed = parse_newick(text)
    assert len(tip_nodes(reparsed)) == 8
    assert tree_height(reparsed) == pytest.approx(tree_height(root), rel=1e-9)
    assert sorted(t.name for t in tip_nodes(reparsed)) == sorted(t.name for t in tip_nodes(root))


def test_parse_newick_strips_annotations():
    root = parse_newick("(a[&x=1]:0.5,b[&y={1,2}]:0.5)[&R]:0.0;")
    assert len(tip_nodes(root)) == 2


def test_parse_newick_rejects_unclosed_groups():
    with pytest.raises(SimulationError):
        parse_newick("(a:1,b:1")


def test_read_tree_from_nexus(tmp_path):
    path = tmp_path / "t.nex"
    path.write_text("#NEXUS\nBegin trees;\ntree x = [&R] (a:1,b:1);\nEnd;\n")
    assert len(tip_nodes(read_tree(path))) == 2


def test_build_rate_matrix_row_major():
    q = build_rate_matrix(["0", "1"], "asymmetric", [2.0, 3.0])
    assert q[0][1] == 2.0 and q[1][0] == 3.0
    assert q[0][0] == -2.0 and q[1][1] == -3.0


def test_build_rate_matrix_equal_model():
    q = build_rate_matrix(["a", "b", "c"], "equal", [1.5])
    assert q[0][1] == q[0][2] == 1.5
    assert q[0][0] == pytest.approx(-3.0)


def test_build_rate_matrix_wrong_dimension():
    with pytest.raises(SimulationError, match="row-major"):
        build_rate_matrix(["a", "b", "c"], "asymmetric", [1.0, 2.0])


def test_simulate_history_events_are_inside_the_tree():
    root = coalescent_tree(12, 2.0, random.Random(5))
    q = build_rate_matrix(["0", "1"], "equal", [1.0])
    events, tip_states = simulate_history(root, ["0", "1"], q, 0, random.Random(5))
    height = tree_height(root)
    assert len(tip_states) == 12
    for event_height, source, dest in events:
        assert 0.0 <= event_height <= height
        assert source != dest
    assert events == sorted(events, key=lambda e: -e[0])


def test_cli_single_event_dataset(tmp_path):
    prefix = tmp_path / "run"
    assert main([
        "--output-prefix", str(prefix),
        "--n-tips", "16", "--tree-height", "2.0",
        "--states", "0,1", "--model", "asymmetric", "--rates", "0.05,0.000001",
        "--root-state", "0", "--n-events", "1", "--mrsd", "2020-01-01", "--seed", "11",
    ]) == 0

    summary = json.loads(Path(f"{prefix}.sim.json").read_text())
    assert summary["n_events"] == 1
    assert summary["n_events_by_direction"] == {"0->1": 1}
    assert summary["tip_state_counts"]["1"] >= 1

    truth = Path(f"{prefix}.truth.tsv").read_text().splitlines()
    assert len(truth) == 2
    columns = truth[1].split("\t")
    assert columns[4] == "0" and columns[5] == "1"

    # the tree file is readable and ultrametric
    root = read_tree(f"{prefix}.trees")
    assert len(tip_nodes(root)) == 16

    # tip states in the metadata match the simulation
    traits = dict(
        line.split(",") for line in Path(f"{prefix}.traits.csv").read_text().splitlines()[1:]
    )
    assert sum(1 for value in traits.values() if value == "1") == summary["tip_state_counts"]["1"]


def test_cli_can_use_an_existing_tree(tmp_path):
    tree = tmp_path / "in.nwk"
    tree.write_text("((a:1,b:1):1,(c:1,d:1):1);")
    prefix = tmp_path / "run"
    assert main([
        "--output-prefix", str(prefix),
        "--input-tree", str(tree),
        "--states", "0,1", "--model", "equal", "--rates", "0.2",
        "--root-state", "0", "--n-events", "1", "--seed", "2",
    ]) == 0
    assert Path(f"{prefix}.trees").exists()


def test_cli_fails_when_the_event_count_is_unreachable(tmp_path):
    with pytest.raises(SimulationError, match="could not simulate"):
        main([
            "--output-prefix", str(tmp_path / "run"),
            "--n-tips", "8", "--tree-height", "1.0",
            "--states", "0,1", "--model", "equal", "--rates", "0.000001",
            "--root-state", "0", "--n-events", "5", "--max-attempts", "50", "--seed", "1",
        ])
