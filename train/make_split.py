#!/usr/bin/env python3
"""Create a fixed, event-stratified 60/20/20 patient split as a tab-separated TXT."""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


def read_labels(path):
    labels = {}
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            case, event, duration = row["case_id"], int(row["event"]), float(row["follow_up_years"])
            if case in labels or event not in (0, 1) or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Duplicate patient or invalid survival label: {case}")
            labels[case] = dict(event=event, follow_up_years=duration)
    if not labels:
        raise ValueError("No patients in labels CSV")
    return labels


def read_split(path, labels):
    split = {}
    with Path(path).open() as file:
        lines = [line.split() for line in file if line.strip() and not line.lstrip().startswith("#")]
    if not lines or lines[0] != ["case_id", "split"]:
        raise ValueError("Split TXT must start with 'case_id split' (tabs or spaces)")
    for fields in lines[1:]:
        if len(fields) != 2:
            raise ValueError(f"Expected case_id and split: {fields}")
        case, group = fields
        if case in split or group not in ("train", "val", "test"):
            raise ValueError(f"Duplicate patient or invalid split: {case}")
        split[case] = group
    if set(split) != set(labels):
        raise ValueError("Split must assign every patient in the labels CSV exactly once")
    if set(split.values()) != {"train", "val", "test"}:
        raise ValueError("Train, val and test must all be nonempty")
    if len({Path(case).stem for case in split}) != len(split):
        raise ValueError("Case IDs must have unique filename stems")
    return split


def make_split(labels, seed=42):
    ids = sorted(labels)
    n_train, n_val = round(0.6 * len(ids)), round(0.2 * len(ids))
    development, test = train_test_split(
        ids,
        test_size=len(ids) - n_train - n_val,
        stratify=[labels[case]["event"] for case in ids],
        random_state=seed,
    )
    train, val = train_test_split(
        development,
        test_size=n_val,
        stratify=[labels[case]["event"] for case in development],
        random_state=seed,
    )
    return {
        case: group for group, cases in (("train", train), ("val", val), ("test", test)) for case in cases
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    labels = read_labels(args.labels)
    split = make_split(labels, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace a split that may already have been used by experiments.
    with args.output.open("x") as file:
        file.write("case_id\tsplit\n")
        file.writelines(f"{case}\t{split[case]}\n" for case in sorted(split))
    for group in ("train", "val", "test"):
        events = Counter(labels[case]["event"] for case in split if split[case] == group)
        print(f"{group}: {sum(events.values())} patients, {events[1]} events, {events[0]} censored")
    print(f"Split saved to {args.output}; seed={args.seed}")


if __name__ == "__main__":
    main()
