"""WP2/WP3: BEAST XML generation (models, root priors, complete histories)."""
import xml.etree.ElementTree as ET

import pytest

from generate_beast_xml import build_registers, build_root_prior, main, sanitise_name

LABELS = ["A", "B", "C", "D"]

NEXUS = """#NEXUS
Begin taxa;
    Dimensions ntax=4;
    Taxlabels
        A
        B
        C
        D
;
End;
Begin trees;
tree STATE_0 = (A:1,B:1,C:1,D:1);
End;
"""


@pytest.fixture
def trees(tmp_path):
    path = tmp_path / "trees.trees"
    path.write_text(NEXUS)
    return path


@pytest.fixture
def traits(tmp_path):
    path = tmp_path / "x.traits.tsv"
    path.write_text("taxon\tstate\nA\t0\nB\t0\nC\t1\nD\t?\n")
    return path


def generate(trees, template_path, output, *extra):
    argv = [
        "--input_template", str(template_path),
        "--input_trees", str(trees),
        "--output_xml", str(output),
        "--name", "run",
        *extra,
    ]
    assert main(argv) == 0
    return ET.parse(output).getroot()


def find_ids(root):
    return {element.get("id") for element in root.iter() if element.get("id")}


# --------------------------------------------------------------------------
# registers
# --------------------------------------------------------------------------
def test_build_registers_binary():
    matrices, rewards = build_registers(["0", "1"], "Trait")
    assert matrices["total"] == "0 1 1 0"
    # Into.<dest> marks the (from, dest) cell of the row-major rate matrix
    assert matrices["Into.1"] == "0 1 0 0"
    assert matrices["OutOf.1"] == "0 0 1 0"
    assert matrices["Trait.0.1"] == "0 1 0 0"
    assert matrices["Trait.1.0"] == "0 0 1 0"
    assert rewards["rewards_1"] == "0 1"


def test_build_registers_multistate_dimension():
    states = ["a", "b", "c"]
    matrices, rewards = build_registers(states, "T")
    assert len(matrices["total"].split()) == 9
    assert matrices["T.a.c"].split().index("1") == 2
    assert len(rewards["rewards_b"].split()) == 3


# --------------------------------------------------------------------------
# root priors
# --------------------------------------------------------------------------
def test_root_prior_uniform():
    prior = build_root_prior("uniform", ["0", "1"], ["0", "0", "1"])
    assert prior["estimated"] is True
    assert prior["vector"] == "1.0 1.0"


def test_root_prior_empirical():
    prior = build_root_prior("empirical", ["0", "1"], ["0", "0", "1"])
    assert prior["vector"] == "0.666667 0.333333"


def test_root_prior_fixed():
    prior = build_root_prior("fixed:1", ["0", "1"], ["0", "0"])
    assert prior["estimated"] is False
    assert prior["vector"] == "0.0 1.0"


def test_root_prior_fixed_unknown_state():
    with pytest.raises(Exception):
        build_root_prior("fixed:9", ["0", "1"], ["0", "0"])


def test_root_prior_unknown():
    with pytest.raises(Exception):
        build_root_prior("nope", ["0", "1"], ["0", "0"])


def test_sanitise_name():
    assert sanitise_name("cfrD") == "cfrD"
    assert sanitise_name("my trait!") == "my_trait_"
    assert sanitise_name("2x") == "trait_2x"


# --------------------------------------------------------------------------
# XML structure
# --------------------------------------------------------------------------
def test_binary_trait_asymmetric(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits),
        "--id-prefix", "cfrD", "--trait-model", "asymmetric",
        "--log-complete-history", "--chain_length", "1000", "--sample_every", "100",
    )
    ids = find_ids(root)
    assert {"cfrD.model", "cfrD.treeLikelihood", "cfrD.rates", "cfrD.root.frequencies"} <= ids
    assert "cfrD.indicators" not in ids
    assert "cfrD.nonZeroRates" not in ids
    assert "Into.s0" in ids and "OutOf.s0" in ids and "cfrD.s0.s1" in ids

    rates = root.find(".//parameter[@id='cfrD.rates']")
    assert rates.get("dimension") == "2"
    assert len(rates.get("value").split()) == 2

    likelihood = root.find(".//markovJumpsTreeLikelihood")
    assert likelihood.get("saveCompleteHistory") == "true"
    assert likelihood.get("useUniformization") == "true"
    assert likelihood.get("numberOfSimulants") == "1"
    assert root.find(".//completeHistoryLogger") is not None
    history_log = root.find(".//log[@id='completeJumpHistory']")
    assert history_log.get("fileName") == "run.jumpHistory.log"
    assert history_log.get("logEvery") == "100"


def test_missing_state_is_coded_unknown(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD",
    )
    attributes = {t.get("id"): t.find("attr").text for t in root.iter("taxon")}
    assert attributes == {"A": "s0", "B": "s0", "C": "s1", "D": "?"}


def test_state_code_style_as_is_keeps_numeric_codes(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD",
        "--state-code-style", "as-is",
    )
    attributes = {t.get("id"): t.find("attr").text for t in root.iter("taxon")}
    assert attributes["A"] == "0"
    assert "Into.0" in find_ids(root)


def test_bssvs_model_adds_indicators_and_prior(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD",
        "--trait-model", "bssvs", "--bssvs-prior-mean", "7",
    )
    ids = find_ids(root)
    assert "cfrD.indicators" in ids and "cfrD.nonZeroRates" in ids
    poisson = root.find(".//poissonPrior")
    assert float(poisson.get("mean")) == 7.0
    assert root.find(".//bitFlipOperator") is not None


def test_bssvs_prior_mean_defaults_to_half_the_ordered_routes(trees, traits, template_path, tmp_path):
    # two states -> two ordered routes -> Poisson mean 1 (Gao et al. 2023);
    # a fixed 3 would put most of the mass beyond the maximum of 2
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD", "--trait-model", "bssvs",
    )
    assert float(root.find(".//poissonPrior").get("mean")) == pytest.approx(1.0)


def test_default_model_is_asymmetric(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD",
    )
    assert "cfrD.indicators" not in find_ids(root)
    assert root.find(".//poissonPrior") is None
    rates = root.find(".//parameter[@id='cfrD.rates']")
    assert len(rates.get("value").split()) == 2


def test_equal_model_fixes_rates(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD",
        "--trait-model", "equal",
    )
    ids = find_ids(root)
    assert "cfrD.indicators" not in ids
    assert not any(op.tag == "scaleOperator" and op.find("parameter").get("idref") == "cfrD.rates"
                   for op in root.iter("scaleOperator"))
    rates = root.find(".//parameter[@id='cfrD.rates']")
    assert set(rates.get("value").split()) == {"1.0"}


def test_root_state_fixed(trees, traits, template_path, tmp_path):
    root = generate(
        trees, template_path, tmp_path / "run.xml",
        "--traits", str(traits), "--id-prefix", "cfrD",
        "--root-state", "fixed:s1",
    )
    frequencies = root.find(".//parameter[@id='cfrD.root.frequencies']")
    assert frequencies.get("value") == "0.0 1.0"
    assert root.find(".//uniformPrior") is None
    assert root.find(".//deltaExchange") is None


def test_label_sourced_trait_keeps_the_historical_identifiers(tmp_path, template_path):
    """A label-sourced trait named `location` is the geographic analysis."""
    from trait_data import main as trait_data_main

    labels = tmp_path / "labels.trees"
    labels.write_text(
        "#NEXUS\nBegin taxa;\n    Dimensions ntax=4;\n    Taxlabels\n"
        "        A|India\n        B|India\n        C|Koshi\n        D|Koshi\n;\nEnd;\n"
        "Begin trees;\ntree STATE_0 = (A|India:1,B|India:1,C|Koshi:1,D|Koshi:1);\nEnd;\n"
    )
    prefix = tmp_path / "run"
    assert trait_data_main([
        "--input-trees", str(labels),
        "--source", "label",
        "--trait", "location",
        "--label-index", "1",
        "--output-prefix", str(prefix),
    ]) == 0

    root = generate(
        labels, template_path, tmp_path / "run.xml",
        "--traits", f"{prefix}.traits.tsv",
        "--id-prefix", "location", "--pair-prefix", "Location",
        "--trait-model", "bssvs",
    )
    ids = find_ids(root)
    assert {
        "location.model",
        "location.treeLikelihood",
        "Into.India",
        "OutOf.Koshi",
        "Location.India.Koshi",
    } <= ids
