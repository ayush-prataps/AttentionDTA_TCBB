"""Create a reproducible drug-disjoint train/test split for Davis."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets" / "Davis.txt"


def load_rows(path: Path) -> list[str]:
    rows = path.read_text().splitlines()
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    malformed = [line_number for line_number, row in enumerate(rows, 1) if len(row.split()) != 5]
    if malformed:
        raise ValueError(f"Malformed rows in {path}: {malformed[:5]}")
    return rows


def make_split(rows: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str], set[str], set[str]]:
    rows_by_drug: dict[str, list[str]] = {}
    for row in rows:
        drug_id = row.split()[0]
        rows_by_drug.setdefault(drug_id, []).append(row)

    drug_ids = sorted(rows_by_drug)
    test_count = max(1, math.ceil(len(drug_ids) * test_fraction))
    if test_count >= len(drug_ids):
        raise ValueError("test_fraction must leave at least one training drug")

    shuffled_drugs = drug_ids[:]
    random.Random(seed).shuffle(shuffled_drugs)
    test_drugs = set(shuffled_drugs[:test_count])
    train_drugs = set(shuffled_drugs[test_count:])

    train_rows = [row for row in rows if row.split()[0] in train_drugs]
    test_rows = [row for row in rows if row.split()[0] in test_drugs]
    if train_drugs & test_drugs:
        raise AssertionError("Drug overlap detected between train and test")
    if len(train_rows) + len(test_rows) != len(rows):
        raise AssertionError("Split does not preserve all input rows")
    return train_rows, test_rows, train_drugs, test_drugs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "splits" / "Davis_drug_disjoint")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=4321)
    args = parser.parse_args()

    rows = load_rows(DATASET_PATH)
    train_rows, test_rows, train_drugs, test_drugs = make_split(rows, args.test_fraction, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.txt").write_text("\n".join(train_rows) + "\n")
    (args.output / "test.txt").write_text("\n".join(test_rows) + "\n")

    print(f"Dataset: {DATASET_PATH}")
    print(f"Seed: {args.seed}")
    print(f"Train drugs: {len(train_drugs)}")
    print(f"Test drugs: {len(test_drugs)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Drug overlap: {len(train_drugs & test_drugs)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()