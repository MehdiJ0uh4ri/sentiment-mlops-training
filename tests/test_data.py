import json

import pytest

from sentiment import config
from sentiment.data import (
    Example,
    assert_no_leakage,
    deduplicate,
    load_examples,
    prepare,
    read_csv,
    split_examples,
)

FILES = {"src_a": "a.txt", "src_b": "b.txt"}


def _write(tmp_path, name, lines):
    (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic(tmp_path, n=40):
    """120 unique rows, 60 positive and 60 negative."""
    _write(
        tmp_path,
        "a.txt",
        [f"great product number {i}\t1" for i in range(n)]
        + [f"awful product number {i}\t0" for i in range(n // 2)],
    )
    _write(
        tmp_path,
        "b.txt",
        [f"lovely movie {i}\t1" for i in range(n // 2)]
        + [f"dreadful movie {i}\t0" for i in range(n)],
    )


def test_load_examples_parses_label_and_source(tmp_path):
    _synthetic(tmp_path)
    examples = load_examples(tmp_path, FILES)
    assert len(examples) == 120
    assert {e.source for e in examples} == {"src_a", "src_b"}
    assert {e.label for e in examples} == {0, 1}


def test_load_examples_rejects_malformed_lines(tmp_path):
    _write(tmp_path, "a.txt", ["ok line\t1", "no tab here", "bad label\t2"])
    _write(tmp_path, "b.txt", ["fine\t0"])
    with pytest.raises(ValueError, match="Malformed"):
        load_examples(tmp_path, FILES)


def test_load_examples_missing_file_points_to_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_examples(tmp_path, FILES)


def test_deduplicate_removes_normalized_duplicates():
    examples = [
        Example("Good food", 1, "a"),
        Example("  good   FOOD ", 1, "b"),
        Example("fine", 0, "a"),
    ]
    kept, report = deduplicate(examples)
    assert [e.text for e in kept] == ["Good food", "fine"]
    assert report.n_exact_duplicates_removed == 1
    assert report.n_final == 2


def test_deduplicate_drops_conflicting_labels_entirely():
    examples = [
        Example("meh", 1, "a"),
        Example("MEH", 0, "b"),
        Example("ok", 1, "a"),
    ]
    kept, report = deduplicate(examples)
    assert [e.text for e in kept] == ["ok"]
    assert report.n_conflicting_removed == 2


def test_split_is_deterministic_for_a_seed(tmp_path):
    _synthetic(tmp_path)
    examples = load_examples(tmp_path, FILES)
    assert split_examples(examples, seed=1) == split_examples(examples, seed=1)
    assert split_examples(examples, seed=1).test != split_examples(examples, seed=2).test


def test_split_is_stratified(tmp_path):
    _synthetic(tmp_path)
    examples = load_examples(tmp_path, FILES)
    split = split_examples(examples, test_size=0.25, seed=0)
    pos = sum(e.label for e in split.test)
    neg = len(split.test) - pos
    assert len(split.test) == 30
    assert abs(pos - neg) <= 1


def test_split_has_no_text_overlap(tmp_path):
    _synthetic(tmp_path)
    cleaned, _ = deduplicate(load_examples(tmp_path, FILES))
    assert_no_leakage(split_examples(cleaned))


def test_prepare_writes_splits_and_manifest(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    _synthetic(raw)
    manifest = prepare(raw_dir=raw, out_dir=out, files=FILES)

    assert (out / "train.csv").exists() and (out / "test.csv").exists()
    assert manifest["train"]["n"] + manifest["test"]["n"] == manifest["cleaning"]["n_final"]
    assert len(read_csv(out / "train.csv")) == manifest["train"]["n"]
    on_disk = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


RAW_PRESENT = all((config.RAW_DIR / f).exists() for f in config.RAW_FILES.values())


@pytest.mark.skipif(not RAW_PRESENT, reason="raw data not downloaded")
def test_real_dataset_loads_cleanly_without_leakage():
    cleaned, report = deduplicate(load_examples())
    assert report.n_raw > 0 and report.n_final > 0
    assert {e.label for e in cleaned} == {0, 1}
    assert_no_leakage(split_examples(cleaned))