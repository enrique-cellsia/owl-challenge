#!/usr/bin/env python3
"""Create two fixed 60/20/20 partitions with event stratification and disjoint tests."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


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


def make_splits(labels, seed=42):
    ids = sorted(labels)
    n = len(ids)
    n_train, n_val = round(0.6 * n), round(0.2 * n)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Not enough patients for nonempty train/val/test")
    if len({Path(case).stem for case in ids}) != n:
        raise ValueError("Case IDs must have unique filename stems")
    by_event = {event: sorted(case for case in ids if labels[case]["event"] == event)
                for event in (0, 1)}
    test_events = round(n_test * len(by_event[1]) / n)
    test_counts = {0: n_test - test_events, 1: test_events}
    val_events = round(n_val * (len(by_event[1]) - test_events) / (n - n_test))
    val_counts = {0: n_val - val_events, 1: val_events}
    for event in (0, 1):
        if (2 * test_counts[event] > len(by_event[event])
                or val_counts[event] > len(by_event[event]) - test_counts[event]):
            raise ValueError("Not enough patients to stratify two disjoint tests")
    if min(test_events, val_events, len(by_event[1]) - test_events - val_events) < 1:
        raise ValueError("Each subset must contain at least one observed event")

    rng = np.random.default_rng(seed)
    tests = [set(), set()]
    for event in (0, 1):
        shuffled = rng.permutation(by_event[event])
        size = test_counts[event]
        for i in range(2):
            tests[i].update(shuffled[i * size:(i + 1) * size].tolist())
    splits = []
    for i in range(2):
        rng_val = np.random.default_rng(seed + i + 1)
        val = set()
        for event in (0, 1):
            eligible = [case for case in by_event[event] if case not in tests[i]]
            val.update(rng_val.choice(eligible, size=val_counts[event], replace=False).tolist())
        splits.append({case: "test" if case in tests[i] else "val" if case in val else "train"
                       for case in ids})
    return splits


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_benchmark(labels_path, output, seed=42):
    labels = read_labels(labels_path)
    splits = make_splits(labels, seed)
    # Never overwrite an official benchmark, even when the directory is empty.
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for i, split in enumerate(splits):
        path = output / f"split_{i}.txt"
        with path.open("x") as file:
            file.write("case_id\tsplit\n")
            file.writelines(f"{case}\t{split[case]}\n" for case in sorted(split))
        counts = {}
        for group in ("train", "val", "test"):
            patients = [case for case in split if split[case] == group]
            events = sum(labels[case]["event"] for case in patients)
            counts[group] = dict(patients=len(patients), events=events, censored=len(patients) - events)
            print(f"split_{i} {group}: {len(patients)} patients, {events} events")
        summaries.append(dict(id=f"split_{i}", file=path.name, sha256=sha256(path),
                              validation_seed=seed + i + 1, counts=counts))
    tests = [{case for case, group in split.items() if group == "test"} for split in splits]
    if tests[0] & tests[1]:
        raise ValueError("Test sets overlap")
    for i, split in enumerate(splits):
        if read_split(output / f"split_{i}.txt", labels) != split:
            raise ValueError("Written split does not match generated partition")
    protocol = dict(
        benchmark_id="owl_internal_benchmark_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Internal development leaderboard with repeated feedback; not an untouched final test.",
        labels=dict(path=str(labels_path.resolve()), sha256=sha256(labels_path), patients=len(labels),
                    events=sum(row["event"] for row in labels.values())),
        requested_fractions=dict(train=0.6, val=0.2, test=0.2),
        generation=dict(random_generator="numpy.random.Generator(PCG64)", numpy_version=np.__version__,
                        test_seed=seed, stratification="event",
                        method="Sorted IDs by event class (0 then 1); a single permutation per class allocates two disjoint test blocks. Validation sampled from each non-test pool with seed+1 and seed+2. Class counts rounded proportionally; train is the remainder."),
        splits=summaries,
        checks=dict(test_sets_disjoint=True, unique_test_patients=len(tests[0] | tests[1]),
                    never_test_patients=len(labels) - len(tests[0] | tests[1])),
        scoring=dict(
            prediction_files=["split_0/predictions.csv", "split_1/predictions.csv"],
            required_columns=["case_id", "risk"], risk_direction="higher_is_higher_recurrence_risk",
            required_patients="Exactly each official test set; unique IDs and finite risks. Read ground truth from official labels.",
            metric="Harrell concordance_index_censored",
            aggregation="Arithmetic mean of the two split C-indices; do not pool predictions.",
            bootstrap=dict(replicates=1000, seed=42,
                           resampling="Resample patients with replacement independently within each test, preserving test sizes; compute the mean of the two C-indices in each replicate.",
                           interval="95% percentile interval of valid replicate means (2.5th and 97.5th percentiles)",
                           invalid_replicates="Exclude and report replicas where either C-index is undefined or nonfinite; fail if none are valid.",
                           scope="Conditional on fixed models; excludes retraining and adaptive selection uncertainty."),
            checkpoint_selection="Use validation only; declare the selection rule before evaluating test.",
            status="Scoring contract only; scorer not implemented here."),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with (output / "summary.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["partition", "subset", "patients", "events", "censored"])
        for summary in summaries:
            for group, counts in summary["counts"].items():
                writer.writerow([summary["id"], group, counts["patients"], counts["events"], counts["censored"]])
    (output / "README.md").write_text(
        "# OWL: benchmark interno v1\n\n"
        "Dos particiones 60/20/20 estratificadas por evento, con tests disjuntos. "
        "Todos los experimentos usan estos TXT sin modificarlos. Cada modelo se entrena desde cero; "
        "sus ajustes usan solo train y la selección del checkpoint solo val. "
        "Entre particiones puede haber solapamiento de train/val, incluso con el test de la otra partición.\n\n"
        "El leaderboard usa la media de ambos C-index de Harrell. El IC bootstrap remuestrea pacientes "
        "dentro de cada test y calcula esa misma media por réplica, condicionado a modelos fijos. "
        "protocol.json contiene hashes y contrato de scoring; summary.csv resume los recuentos.\n"
    )
    print(f"Benchmark saved to {output}; no overlap between tests")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New benchmark directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    write_benchmark(args.labels, args.output, args.seed)


if __name__ == "__main__":
    main()
