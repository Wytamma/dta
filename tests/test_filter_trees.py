"""Input tree burn-in: the leading trees of a posterior are dropped before BEAST."""
import pytest

from filter_trees import filter_trees, tree_count

NEXUS = """#NEXUS
# Produced by delphy

Begin taxa;
\tDimensions ntax=2;
\tTaxlabels
\t\ta
\t\tb
\t;
End;

Begin trees;
\ttree STATE_0 = [&R] (a:1,b:1);
\ttree STATE_100 = [&R] (a:1,b:1);
\ttree STATE_200 = [&R] (a:1,b:1);
End;
"""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "in.trees"
    path.write_text(NEXUS)
    return path


def test_tree_count_counts_statements(source):
    assert tree_count(source) == 3


def test_filter_drops_the_leading_trees_and_keeps_the_header(source, tmp_path):
    target = tmp_path / "out.trees"
    kept, total = filter_trees(source, target, 0.34)  # int(3 * 0.34) == 1
    assert (kept, total) == (2, 3)
    text = target.read_text()
    assert "Dimensions ntax=2" in text and "Taxlabels" in text and "End;" in text
    assert "STATE_0 " not in text
    assert "STATE_100" in text and "STATE_200" in text


def test_filter_never_drops_every_tree(source, tmp_path):
    target = tmp_path / "out.trees"
    kept, total = filter_trees(source, target, 1.0)
    assert (kept, total) == (1, 3)
    assert "STATE_200" in target.read_text()


def test_filter_rejects_a_file_without_trees(tmp_path):
    path = tmp_path / "empty.trees"
    path.write_text("#NEXUS\nEnd;\n")
    with pytest.raises(ValueError, match="no tree statements"):
        filter_trees(path, tmp_path / "out.trees", 0.1)
