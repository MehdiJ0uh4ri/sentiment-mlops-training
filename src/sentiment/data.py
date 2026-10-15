"""Load, validate, deduplicate and split the sentiment dataset.

Order matters for leakage prevention: clean and deduplicate first, then split,
then (in later milestones) fit anything learned from data on the train split only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from sentiment import config

VALID_LABELS = {"0": 0, "1": 1}


@dataclass(frozen=True)
class Example:
    text: str
    label: int
    source: str


@dataclass(frozen=True)
class CleaningReport:
    n_raw: int
    n_exact_duplicates_removed: int
    n_conflicting_removed: int
    n_final: int


@dataclass(frozen=True)
class DataSplit:
    train: list[Example]
    test: list[Example]


def normalize_key(text: str) -> str:
    """Key used to detect duplicates: lowercase, collapsed whitespace."""
    return " ".join(text.lower().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(
    raw_dir: Path = config.RAW_DIR, files: dict[str, str] | None = None
) -> list[Example]:
    """Parse `text<TAB>label` files. Fails loudly on any malformed line."""
    files = files or config.RAW_FILES
    examples: list[Example] = []
    problems: list[str] = []

    for source in sorted(files):
        path = raw_dir / files[source]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/download_data.py"
            )
        with path.open(encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                text, sep, label = line.rpartition("\t")
                text, label = text.strip(), label.strip()
                if not sep or not text or label not in VALID_LABELS:
                    problems.append(f"{path.name}:{line_no}: {line[:60]!r}")
                    continue
                examples.append(Example(text, VALID_LABELS[label], source))

    if problems:
        raise ValueError(
            f"Malformed lines found ({len(problems)}):\n" + "\n".join(problems[:5])
        )
    return examples


def deduplicate(examples: list[Example]) -> tuple[list[Example], CleaningReport]:
    """Drop repeated texts; drop texts that appear with conflicting labels."""
    labels_by_key: dict[str, set[int]] = {}
    for ex in examples:
        labels_by_key.setdefault(normalize_key(ex.text), set()).add(ex.label)
    conflicting = {k for k, v in labels_by_key.items() if len(v) > 1}

    kept: list[Example] = []
    seen: set[str] = set()
    n_dupes = 0
    n_conflicts = 0
    for ex in examples:
        key = normalize_key(ex.text)
        if key in conflicting:
            n_conflicts += 1
            continue
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        kept.append(ex)

    return kept, CleaningReport(len(examples), n_dupes, n_conflicts, len(kept))


def split_examples(
    examples: list[Example],
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> DataSplit:
    """Deterministic stratified split on label."""
    labels = [e.label for e in examples]
    train_idx, test_idx = train_test_split(
        list(range(len(examples))),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return DataSplit(
        train=[examples[i] for i in train_idx],
        test=[examples[i] for i in test_idx],
    )


def assert_no_leakage(split: DataSplit) -> None:
    train_keys = {normalize_key(e.text) for e in split.train}
    test_keys = {normalize_key(e.text) for e in split.test}
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} texts appear in both train and test, "
            f"e.g. {sorted(overlap)[:3]}"
        )


def write_csv(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label", "source"])
        for ex in examples:
            writer.writerow([ex.text, ex.label, ex.source])


def read_csv(path: Path) -> list[Example]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            Example(row["text"], int(row["label"]), row["source"])
            for row in csv.DictReader(fh)
        ]


def _label_counts(examples: list[Example]) -> dict[str, int]:
    counts = Counter(e.label for e in examples)
    return {str(k): counts[k] for k in sorted(counts)}


def prepare(
    raw_dir: Path = config.RAW_DIR,
    out_dir: Path = config.PROCESSED_DIR,
    files: dict[str, str] | None = None,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> dict:
    """Load, clean, split, verify no leakage, write CSVs and a manifest."""
    files = files or config.RAW_FILES
    examples = load_examples(raw_dir, files)
    cleaned, report = deduplicate(examples)
    split = split_examples(cleaned, test_size=test_size, seed=seed)
    assert_no_leakage(split)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "train.csv", split.train)
    write_csv(out_dir / "test.csv", split.test)

    manifest = {
        "seed": seed,
        "test_size": test_size,
        "raw_files_sha256": {
            name: sha256_file(raw_dir / fname) for name, fname in sorted(files.items())
        },
        "cleaning": asdict(report),
        "train": {"n": len(split.train), "label_counts": _label_counts(split.train)},
        "test": {"n": len(split.test), "label_counts": _label_counts(split.test)},
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = prepare()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nWrote splits to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()