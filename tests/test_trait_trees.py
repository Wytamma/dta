"""WP7: annotated tree parsing, prevalence over time and TMRCA."""
import numpy as np
import pytest

from trait_trees import (
    TreeParseError,
    branch_states,
    node_heights,
    parse_annotations,
    parse_history_events,
    parse_newick,
    parse_translate,
    prevalence_counts,
    read_trees,
    tip_states,
    tmrca_by_state,
    tree_count,
    tree_state_number,
)

# a and b carry the trait; it was acquired once, on the branch above their MRCA
NEWICK = (
    "[&R] ((a[&st=\"1\"]:1,b[&st=\"1\"]:1)[&st=\"1\"]:"
    "[&branch.history={{2.5,0,1}}]2,(c[&st=\"0\"]:1,d[&st=\"0\"]:1)[&st=\"0\"]:2)[&st=\"0\"]:0"
)
# the same tree without any sampled history annotation
NEWICK_NO_HISTORY = (
    "[&R] ((a[&st=\"1\"]:1,b[&st=\"1\"]:1)[&st=\"1\"]:2,"
    "(c[&st=\"0\"]:1,d[&st=\"0\"]:1)[&st=\"0\"]:2)[&st=\"0\"]:0"
)

TREES = """#NEXUS

Begin taxa;
\tDimensions ntax=2;
\tTaxlabels
\t\ta
\t\tb
\t;
End;

Begin trees;
\tTranslate
\t\t1 'a',
\t\t2 'b'
\t;
tree STATE_0 = [&R] (1[&st="1"]:1,2[&st="1"]:1)[&st="1"]:[&branch.history={{2.5,0,1}}]2;
tree STATE_1000 = [&R] (1[&st="1"]:1,2[&st="0"]:1)[&st="1"]:2;
End;
"""


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def carrier_branch(root):
    """The internal node whose branch carries the sampled acquisition."""
    return next(node for node in walk(root) if "branch.history" in node.annotations)


@pytest.fixture
def tree_file(tmp_path):
    path = tmp_path / "run.trees"
    path.write_text(TREES)
    return path


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def test_parse_annotations_keeps_nested_values():
    parsed = parse_annotations('st="1",branch.history={{2.5,0,1}},height=3')
    assert parsed["st"] == "1"
    assert parsed["branch.history"] == "{{2.5,0,1}}"
    assert parsed["height"] == "3"


def test_parse_newick_collects_node_and_branch_annotations():
    root = parse_newick(NEWICK)
    assert node_heights(root)[id(root)] == 3.0
    assert len(root.children) == 2
    a = next(node for node in walk(root) if node.name == "a")
    assert a.annotations["st"] == "1"
    assert a.parent.annotations["branch.history"] == "{{2.5,0,1}}"


def test_parse_translate_table():
    assert parse_translate(TREES) == {"1": "a", "2": "b"}


def test_read_trees_resolves_the_translate_table(tree_file):
    trees = list(read_trees(tree_file))
    assert [tree_state_number(name) for name, _ in trees] == [0, 1000]
    assert tip_states(trees[0][1], "st") == {"a": "1", "b": "1"}
    assert tip_states(trees[1][1], "st") == {"a": "1", "b": "0"}


def test_read_trees_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.trees"
    path.write_text("#NEXUS\nEnd;\n")
    with pytest.raises(TreeParseError, match="no tree statements"):
        list(read_trees(path))


def test_tree_count_counts_statements_without_parsing(tree_file):
    assert tree_count(tree_file) == 2


def test_read_trees_only_parses_the_requested_positions(tmp_path):
    malformed = TREES.replace(
        '(1[&st="1"]:1,2[&st="1"]:1)[&st="1"]:', '((1[&st="1"]:1,2[&st="1"]:1)[&st="1"]:'
    )
    path = tmp_path / "run.trees"
    path.write_text(malformed)
    # the discarded tree is unbalanced: reading it would fail
    assert [name for name, _ in read_trees(path, indices={1})] == ["STATE_1000"]
    with pytest.raises(TreeParseError):
        list(read_trees(path))


def test_parse_newick_rejects_unbalanced_groups():
    with pytest.raises(TreeParseError):
        parse_newick("(a:1,b:1));")


# --------------------------------------------------------------------------
# history annotations
# --------------------------------------------------------------------------
def test_parse_history_events_three_and_four_fields():
    assert parse_history_events("{{1,2.5,s0,s1}}") == [(2.5, "s0", "s1")]
    assert parse_history_events("{{2.5,s0,s1}}") == [(2.5, "s0", "s1")]


def test_parse_history_events_oldest_first():
    assert parse_history_events("{{3.0,a,b},{1.5,b,a}}") == [(3.0, "a", "b"), (1.5, "b", "a")]


def test_parse_history_events_empty():
    assert parse_history_events(None) == []
    assert parse_history_events("{}") == []


# --------------------------------------------------------------------------
# prevalence
# --------------------------------------------------------------------------
def test_branch_states_above_and_below_the_event():
    root = parse_newick(NEWICK)
    heights = node_heights(root)
    grid = np.array([2.9, 2.0, 0.5])
    states = branch_states(carrier_branch(root), ["0", "1"], "st", grid, heights)
    # above the acquisition the lineage is still in state 0, below it in state 1
    assert list(states) == [0, 1, 1]


def test_prevalence_counts_by_height():
    root = parse_newick(NEWICK)
    grid = np.array([2.9, 2.0, 0.5])
    counts = prevalence_counts(root, ["0", "1"], "st", grid)
    # two lineages just below the root, four at the tips
    assert counts.sum(axis=0).tolist() == [2.0, 2.0, 4.0]
    assert counts[:, 0].tolist() == [2.0, 0.0]  # both root lineages are still state 0
    assert counts[:, 1].tolist() == [1.0, 1.0]  # the carrier lineage has flipped
    assert counts[:, 2].tolist() == [2.0, 2.0]


def test_prevalence_falls_back_to_the_child_state_without_history():
    root = parse_newick(NEWICK_NO_HISTORY)
    grid = np.array([2.5])
    counts = prevalence_counts(root, ["0", "1"], "st", grid)
    assert counts.sum(axis=0).tolist() == [2.0]
    # both internal branches are inside the grid point; without the history the
    # carrier branch keeps the state of its child node
    assert counts[:, 0].tolist() == [1.0, 1.0]


def test_branch_without_a_state_annotation_is_skipped():
    root = parse_newick("[&R] (a:1,b:1);")
    counts = prevalence_counts(root, ["0", "1"], "st", np.array([0.5]))
    assert counts.sum() == 0.0


# --------------------------------------------------------------------------
# TMRCA
# --------------------------------------------------------------------------
def test_tmrca_by_state():
    root = parse_newick(NEWICK)
    result = tmrca_by_state(root, "st", ["0", "1"])
    assert result["1"] == (2, 1.0)  # a and b meet one year before the tips
    assert result["0"] == (2, 1.0)
    assert tmrca_by_state(root, "st", ["absent"])["absent"] == (0, None)


def test_tmrca_of_a_single_tip_is_present_day():
    root = parse_newick(NEWICK)
    next(node for node in walk(root) if node.name == "a").annotations["st"] = "2"
    assert tmrca_by_state(root, "st", ["2"])["2"] == (1, 0.0)
