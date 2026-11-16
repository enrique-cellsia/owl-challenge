#!/usr/bin/env python3
"""Score two prediction CSVs against the fixed OWL internal benchmark v1 (CPU only)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored

from make_split import read_labels, read_split

# Official benchmark fingerprints: custom splits cannot be submitted as benchmark v1.
LABELS_SHA256 = "04d8229b58348a317790d279ae134d84f6a795ab956fd52208c08f2279ff1186"
SPLIT_SHA256 = [
    "d3afa0553eeb57b0d5b507fdc884121da1774aa1098ac548f532fe8c4ba94274",
    "7a4325bb07094353533e267ae05df0fe867bb2ba2b8b7d54e75cc6f87c981d91",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concordance(events, times, risk):
    if not events.any():
        return None
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            value = float(concordance_index_censored(events, times, risk)[0])
        return value if math.isfinite(value) else None
    except NoComparablePairException:
        return None


def read_predictions(path, expected):
    risks = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not {"case_id", "risk"} <= set(reader.fieldnames):
            raise ValueError(f"CSV requires unique case_id and risk columns: {path}")
        for row in reader:
            case = row["case_id"]
            if case in risks:
                raise ValueError(f"Duplicate patient in {path}: {case}")
            value = float(row["risk"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite risk in {path}: {case}")
            risks[case] = value
    if set(risks) != expected:
        raise ValueError(f"Patient mismatch in {path}: missing={sorted(expected - set(risks))}; "
                         f"unexpected={sorted(set(risks) - expected)}")
    return risks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/training/benchmark_v1"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New output directory; default: EXPERIMENT/scoring")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.experiment / "scoring"
    if output.exists():
        raise FileExistsError(output)
    protocol_path = args.benchmark / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["benchmark_id"] != "owl_internal_benchmark_v1":
        raise ValueError("Unsupported benchmark version")
    labels_path = Path(protocol["labels"]["path"])
    if sha256(labels_path) != LABELS_SHA256 or protocol["labels"]["sha256"] != LABELS_SHA256:
        raise ValueError("Labels do not match official benchmark v1")
    labels = read_labels(labels_path)
    if len(protocol["splits"]) != 2:
        raise ValueError("Benchmark v1 requires exactly two partitions")
    bootstrap = protocol["scoring"]["bootstrap"]
    if bootstrap["replicates"] != 1000 or bootstrap["seed"] != 42:
        raise ValueError("Benchmark v1 requires 1000 bootstrap replicas and seed 42")
    arrays, reports, test_sets = [], [], []
    for i, expected_hash in enumerate(SPLIT_SHA256):
        name = f"split_{i}"
        definition = protocol["splits"][i]
        if definition["id"] != name or definition["sha256"] != expected_hash:
            raise ValueError(f"Invalid official split definition: {name}")
        split_path = args.benchmark / definition["file"]
        if sha256(split_path) != expected_hash:
            raise ValueError(f"Official split was modified: {split_path}")
        split = read_split(split_path, labels)
        expected = {case for case, group in split.items() if group == "test"}
        test_sets.append(expected)
        path = args.experiment / name / "predictions.csv"
        risks = read_predictions(path, expected)
        ids = sorted(expected)
        events = np.array([bool(labels[case]["event"]) for case in ids])
        times = np.array([labels[case]["follow_up_years"] for case in ids], dtype=float)
        risk = np.array([risks[case] for case in ids], dtype=float)
        value = concordance(events, times, risk)
        if value is None:
            raise ValueError(f"C-index undefined in {name}")
        arrays.append((events, times, risk))
        reports.append(dict(split=name, c_index=value, n_patients=len(ids), n_events=int(events.sum()),
                            predictions_sha256=sha256(path), split_sha256=expected_hash))
    if test_sets[0] & test_sets[1]:
        raise ValueError("Evaluation patients overlap")
    rng = np.random.default_rng(42)
    samples, boot_rows = [], []
    for replica in range(1000):
        values = []
        for events, times, risk in arrays:
            ix = rng.integers(0, len(events), size=len(events))
            values.append(concordance(events[ix], times[ix], risk[ix]))
        mean = float(np.mean(values)) if all(value is not None for value in values) else None
        if mean is not None:
            samples.append(mean)
        boot_rows.append(dict(replica=replica + 1, split_0=values[0], split_1=values[1], mean_c_index=mean))
    if not samples:
        raise ValueError("No valid joint bootstrap replicas")
    report = dict(benchmark_id=protocol["benchmark_id"], experiment=str(args.experiment.resolve()),
                  splits=reports, mean_c_index=float(np.mean([r["c_index"] for r in reports])),
                  mean_c_index_ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
                  bootstrap_replicates=1000, bootstrap_valid=len(samples),
                  bootstrap_invalid=1000 - len(samples), bootstrap_seed=42,
                  confidence_interval="95% percentile patient bootstrap within each test, for the mean of both C-indices; conditional on fixed models",
                  labels_sha256=LABELS_SHA256, protocol_sha256=sha256(protocol_path), scorer_sha256=sha256(Path(__file__)))
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "bootstrap.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["replica", "split_0", "split_1", "mean_c_index"])
        writer.writeheader()
        writer.writerows(boot_rows)
    with (output / "leaderboard_row.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "split_0_c_index", "split_1_c_index", "mean_c_index", "ci95_low", "ci95_high"])
        writer.writerow([args.experiment.name, reports[0]["c_index"], reports[1]["c_index"],
                         report["mean_c_index"], *report["mean_c_index_ci95"]])
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved scoring in {output}", flush=True)


if __name__ == "__main__":
    main()
