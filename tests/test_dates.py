from dates import infer_mrsd, label_date


def nexus(*tips: str) -> str:
    return "#NEXUS\nBegin trees;\ntree STATE_0 = [&R] (" + ",".join(tips) + ");\nEnd;\n"


def test_label_date_takes_the_last_date_in_the_label():
    assert label_date("EPI_ISL_2546697|India|B.1.617.2|2021-05-05") == "2021-05-05"
    assert label_date("t89") is None
    assert label_date(None) is None


def test_infer_mrsd_from_dated_tips(tmp_path):
    # a dated tree is in years before its newest tip: date + height == mrsd
    path = tmp_path / "dated.trees"
    path.write_text(nexus("'a|2020-01-01':0.5", "'b|2021-01-01':1.5"))
    assert infer_mrsd(path) == "2021-01-01"


def test_infer_mrsd_resolves_the_translate_table(tmp_path):
    path = tmp_path / "translated.trees"
    path.write_text(
        "#NEXUS\nBegin trees;\n"
        "\tTranslate\n\t\t1 'a|2020-01-01',\n\t\t2 'b|2021-01-01'\n\t;\n"
        "tree STATE_0 = [&R] (2:1.5,1:0.5);\nEnd;\n"
    )
    assert infer_mrsd(path) == "2021-01-01"


def test_infer_mrsd_needs_dated_labels(tmp_path):
    path = tmp_path / "undated.trees"
    path.write_text(nexus("t1:1", "t2:1", "t3:1"))
    assert infer_mrsd(path) is None


def test_infer_mrsd_rejects_a_tree_that_is_not_a_clock(tmp_path):
    # the same dates under branch lengths that do not encode time
    path = tmp_path / "scaled.trees"
    path.write_text(nexus("'a|2020-01-01':1", "'b|2021-01-01':0"))
    assert infer_mrsd(path) is None


def test_infer_mrsd_ignores_a_stray_label(tmp_path):
    tips = [f"'t{index}|2021-01-01':1" for index in range(40)]
    tips.append("'stray|1999-01-01':1")
    path = tmp_path / "stray.trees"
    path.write_text(nexus(*tips))
    assert infer_mrsd(path) == "2021-01-01"
