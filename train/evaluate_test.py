#!/usr/bin/env python3
"""Generate test risk predictions from a MIL checkpoint selected on validation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from train_mil import Bags, MIL, read_labels, read_split, survival_and_risk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, help="Override saved HDF5 folder if moved")
    parser.add_argument("--output", type=Path, help="CSV path; default: SPLIT_RUN/predictions.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-in-ram", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=16)
    args = parser.parse_args()
    if args.cache_workers < 1:
        parser.error("Cache worker count must be positive")
    checkpoint = args.checkpoint.resolve()
    experiment = checkpoint.parent.parent
    output = args.output if args.output is not None else experiment / "predictions.csv"
    metadata_path = output.with_suffix(".json")
    if output.suffix != ".csv":
        parser.error("Output must be a .csv file")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Predictions or metadata already exist: {output}")
    # Checkpoints are locally produced training artifacts.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    labels_path, split_path = experiment / "labels.csv", experiment / "split.txt"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != config["split_sha256"]:
        raise ValueError("Saved split differs from checkpoint training split")
    labels = read_labels(labels_path)
    split = read_split(split_path, labels)
    cases = sorted(case for case, group in split.items() if group == "test")
    cuts = np.asarray(config["time_cuts_years"], dtype=np.float64)
    if (len(cuts) != config["model"]["bins"] + 1 or cuts[0] != 0
            or not np.isfinite(cuts).all() or not np.all(np.diff(cuts) > 0)):
        raise ValueError("Invalid time boundaries in checkpoint")
    embeddings = args.embeddings if args.embeddings is not None else Path(config["embeddings"])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = MIL(**config["model"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    dataset = Bags(cases, labels, embeddings, config["model"]["input_dim"],
                   cache_in_ram=args.cache_in_ram, cache_workers=args.cache_workers)
    print(f"Test inference: {len(cases)} patients; epoch {saved['epoch']}; device={device}", flush=True)
    rows = []
    started = last_log = time.monotonic()
    with torch.inference_mode():
        for index in range(len(dataset)):
            features, _, _, case = dataset[index]
            logits = model(features.to(device=device, dtype=torch.float32))
            if not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite prediction: {case}")
            _, risk = survival_and_risk(logits, cuts)
            if not torch.isfinite(risk).all():
                raise ValueError(f"Nonfinite risk: {case}")
            rows.append(dict(case_id=case, risk=risk.item()))
            now = time.monotonic()
            if now - last_log >= 10 or len(rows) == len(cases):
                print(f"Test inference: {len(rows)}/{len(cases)} WSI", flush=True)
                last_log = now
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["case_id", "risk"])
        writer.writeheader()
        writer.writerows(rows)
    report = dict(checkpoint=str(checkpoint), epoch=int(saved["epoch"]),
                  split_sha256=split_hash, labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                  predictions_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  embeddings=str(embeddings.resolve()), time_cuts_years=cuts.tolist(),
                  risk=config["risk"], n_patients=len(rows), seconds=time.monotonic() - started)
    with metadata_path.open("x") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output} and {metadata_path}; no scoring performed.", flush=True)


if __name__ == "__main__":
    main()
