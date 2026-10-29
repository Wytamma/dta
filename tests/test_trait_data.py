"""WP1: external trait inputs, taxon matching, missing states."""
import json

import pytest

from trait_data import (
    MISSING,
    TraitError,
    assign_from_labels,
    assign_traits,
    extract_label_value,
    extract_taxa_from_nexus,
    natural_sort_key,
    read_assignment,
    read_trait_table,
    to_beast_code,
    write_assignment,
)

NEXUS = """#NEXUS
Begin taxa;
    Dimensions ntax=3;
        Taxlabels
            A|India|2020-01-01
            B|Koshi|2020-02-01
            C|India|2020-03-01
;
End;
Begin trees;
    Translate
        1 A|India|2020-01-01,
        2 B|Koshi|2020-02-01,
        3 C|India|2020-03-01
;
tree STATE_0 = (1:1,2:1,3:1);
End;
"""

LABELS = ["A|India|2020-01-01", "B|Koshi|2020-02-01", "C|India|2020-03-01"]


@pytest.fixture
def nexus_file(tmp_path):
    path = tmp_path / "trees.trees"
    path.write_text(NEXUS)
    return path


def test_extract_taxa_from_nexus(nexus_file):
    assert extract_taxa_from_nexus(nexus_file) == LABELS


def test_extract_taxa_handles_single_line_block(tmp_path):
    path = tmp_path / "t.nex"
    path.write_text("#NEXUS\nBegin taxa;\nDimensions ntax=2;\nTaxlabels 'a b' c;\nEnd;\n")
    assert extract_taxa_from_nexus(path) == ["a b", "c"]


def test_extract_taxa_missing_block(tmp_path):
    path = tmp_path / "t.nex"
    path.write_text("#NEXUS\nBegin trees;\nEnd;\n")
    with pytest.raises(TraitError):
        extract_taxa_from_nexus(path)


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,mode,kwargs,expected",
    [
        ("A|India|2020", "exact", {}, "A|India|2020"),
        ("A|India|2020", "field", {"delimiter": "|", "index": 1}, "India"),
        ("A|India|2020", "field", {"delimiter": "|", "index": -1}, "2020"),
        ("A|India|2020", "field", {"delimiter": "|", "index": 9}, None),
        ("A_India_2020", "regex", {"regex": r"^([^_]+)_"}, "A"),
    ],
)
def test_extract_label_value(label, mode, kwargs, expected):
    assert extract_label_value(label, mode, **kwargs) == expected


def test_extract_label_value_regex_without_pattern():
    with pytest.raises(TraitError):
        extract_label_value("A", "regex")


def test_extract_label_value_unknown_mode():
    with pytest.raises(TraitError):
        extract_label_value("A", "nope")


# --------------------------------------------------------------------------
# reading the metadata table
# --------------------------------------------------------------------------
def test_read_trait_table_infers_columns(tmp_path):
    path = tmp_path / "traits.csv"
    path.write_text("taxon,cfrD\nA,1\nB,0\n")
    table = read_trait_table(path)
    assert table.trait == "cfrD"
    assert table.observations == {"A": "1", "B": "0"}


def test_read_trait_table_selects_named_column(tmp_path):
    path = tmp_path / "traits.tsv"
    path.write_text("taxon\tcfrD\tgyrA\nA\t1\tS\nB\t0\tR\n")
    table = read_trait_table(path, trait="gyrA")
    assert table.observations == {"A": "S", "B": "R"}


def test_read_trait_table_ambiguous_without_trait(tmp_path):
    path = tmp_path / "traits.csv"
    path.write_text("taxon,a,b\nA,1,2\n")
    with pytest.raises(TraitError):
        read_trait_table(path)


def test_read_trait_table_duplicate_taxon(tmp_path):
    path = tmp_path / "traits.csv"
    path.write_text("taxon,cfrD\nA,1\nA,0\n")
    with pytest.raises(TraitError, match="duplicate"):
        read_trait_table(path)


def test_read_trait_table_missing_column(tmp_path):
    path = tmp_path / "traits.csv"
    path.write_text("taxon,cfrD\nA,1\n")
    with pytest.raises(TraitError, match="no column"):
        read_trait_table(path, trait="nope")


def test_read_trait_table_missing_file(tmp_path):
    with pytest.raises(TraitError, match="not found"):
        read_trait_table(tmp_path / "absent.csv")


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------
def make_table(mapping, trait="cfrD"):
    from trait_data import TraitTable

    return TraitTable(trait=trait, observations=dict(mapping), source="test", columns=[])


def test_assign_traits_exact_and_order():
    assignment = assign_traits(LABELS, make_table({label: str(i % 2) for i, label in enumerate(LABELS)}))
    assert assignment.taxa == LABELS
    assert assignment.states == ["0", "1"]
    assert assignment.values == ["0", "1", "0"]
    assert assignment.counts() == {"0": 2, "1": 1}
    assert assignment.frequencies() == [2 / 3, 1 / 3]


def test_assign_traits_missing_taxon_becomes_unknown():
    assignment = assign_traits(LABELS, make_table({LABELS[0]: "1", LABELS[1]: "0"}))
    assert assignment.values == ["1", "0", MISSING]
    assert assignment.missing == [LABELS[2]]
    assert assignment.to_dict()["n_missing"] == 1


def test_assign_traits_missing_taxon_can_be_fatal():
    with pytest.raises(TraitError, match="no trait observation"):
        assign_traits(LABELS, make_table({LABELS[0]: "1"}), on_missing="error")


def test_assign_traits_missing_token_in_table():
    assignment = assign_traits(
        LABELS, make_table({LABELS[0]: "1", LABELS[1]: "NA", LABELS[2]: "0"})
    )
    assert assignment.values == ["1", MISSING, "0"]


def test_assign_traits_field_matching():
    table = make_table({"India": "Asia", "Koshi": "Oceania"})
    assignment = assign_traits(LABELS, table, mode="field", delimiter="|", index=1)
    assert assignment.values == ["Asia", "Oceania", "Asia"]
    assert assignment.states == ["Asia", "Oceania"]


def test_assign_traits_reports_unmatched_metadata():
    assignment = assign_traits(
        LABELS, make_table({LABELS[0]: "1", LABELS[1]: "0", "Z": "0", LABELS[2]: "1"})
    )
    assert assignment.unmatched_observations == ["Z"]


def test_assign_traits_declared_states_are_enforced():
    with pytest.raises(TraitError, match="not part of the declared state set"):
        assign_traits(LABELS, make_table({label: "2" for label in LABELS}), states=["0", "1"])


def test_assign_traits_needs_two_states():
    with pytest.raises(TraitError, match="at least two states"):
        assign_traits(LABELS, make_table({label: "1" for label in LABELS}))


def test_assign_traits_warns_about_unobserved_states(capsys):
    assign_traits(
        LABELS, make_table({LABELS[0]: "1", LABELS[1]: "1", LABELS[2]: "1"}), states=["0", "1"]
    )
    assert "never observed" in capsys.readouterr().err


def test_assign_traits_require_variation():
    with pytest.raises(TraitError, match="constant"):
        assign_traits(
            LABELS,
            make_table({label: "1" for label in LABELS}),
            states=["0", "1"],
            require_variation=True,
        )


def test_assign_traits_rejects_log_breaking_state_codes():
    with pytest.raises(TraitError, match="break the BEAST XML"):
        assign_traits(LABELS, make_table({label: "a,b" for label in LABELS}))


def test_assign_from_labels_legacy_mode():
    assignment = assign_from_labels(LABELS, delimiter="|", index=1)
    assert assignment.states == ["India", "Koshi"]
    assert assignment.values == ["India", "Koshi", "India"]


def test_assign_from_labels_rejects_out_of_range_index():
    with pytest.raises(TraitError, match="could not extract"):
        assign_from_labels(LABELS, delimiter="|", index=7)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_natural_sort_key_orders_numbers_numerically():
    assert sorted(["10", "2", "1"], key=natural_sort_key) == ["1", "2", "10"]


@pytest.mark.parametrize(
    "code,expected",
    [("0", "s0"), ("1", "s1"), ("1.5", "s1_5"), ("-2", "s_2"), ("absent", "absent"), ("s0", "s0")],
)
def test_to_beast_code(code, expected):
    assert to_beast_code(code) == expected


def test_write_and_read_assignment_round_trip(tmp_path):
    assignment = assign_traits(LABELS, make_table({LABELS[0]: "1", LABELS[1]: "0"}))
    tsv = tmp_path / "x.traits.tsv"
    js = tmp_path / "x.traits.json"
    write_assignment(assignment, tsv, js)
    taxa, values = read_assignment(tsv)
    assert taxa == LABELS
    assert values == ["1", "0", MISSING]
    report = json.loads(js.read_text())
    assert report["state_code_map"] == {"0": "s0", "1": "s1"}
    assert report["n_observed"] == 2


def test_write_state_labels_maps_beast_codes_back(tmp_path):
    from trait_data import write_state_labels

    assignment = assign_traits(LABELS, make_table({LABELS[0]: "1", LABELS[1]: "0"}))
    path = tmp_path / "x.state_labels.tsv"
    write_state_labels(assignment, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "beast_code\tlabel"
    assert sorted(lines[1:]) == ["s0\t0", "s1\t1"]


def test_read_assignment_rejects_wrong_header(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("a\tb\n1\t2\n")
    with pytest.raises(TraitError, match="expected a 'taxon"):
        read_assignment(path)


def test_cli_end_to_end(tmp_path, nexus_file):
    from trait_data import main

    table = tmp_path / "traits.csv"
    table.write_text(f"taxon,cfrD\n{LABELS[0]},1\n{LABELS[1]},0\n")
    prefix = tmp_path / "out"
    assert main(
        [
            "--input-trees",
            str(nexus_file),
            "--trait-data",
            str(table),
            "--trait",
            "cfrD",
            "--output-prefix",
            str(prefix),
        ]
    ) == 0
    report = json.loads((tmp_path / "out.traits.json").read_text())
    assert report["states"] == ["0", "1"]
    assert report["n_missing"] == 1
