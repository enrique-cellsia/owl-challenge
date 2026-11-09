#!/usr/bin/env python3
"""Evaluate one previously selected MIL checkpoint on its saved patient test split."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored
import torch

from make_split import read_labels, read_split
from train_mil import Bags, MIL, save_predictions, survival_and_risk, survival_loss


def concordance(events, times, risk):
    if not events.any():
        return None
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            value = float(concordance_index_censored(events, times, risk)[0])
        return value if np.isfinite(value) else None
    except NoComparablePairException:
        return None


def test_metrics(rows, bootstrap=1000, seed=42):
    events = np.array([bool(row["event"]) for row in rows])
    times = np.array([row["follow_up_years"] for row in rows])
    risk = np.array([row["risk"] for row in rows])
    c_index = concordance(events, times, risk)
    samples = []
    rng = np.random.default_rng(seed)
    if c_index is not None:
        for _ in range(bootstrap):
            indices = rng.integers(0, len(rows), size=len(rows))
            value = concordance(events[indices], times[indices], risk[indices])
            if value is not None:
                samples.append(value)
    return dict(
        n_patients=len(rows),
        n_events=int(events.sum()),
        n_censored=int((~events).sum()),
        test_c_index=c_index,
        test_c_index_ci95=np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
        bootstrap_samples=bootstrap,
        bootstrap_valid=len(samples),
        bootstrap_seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Checkpoint already selected on validation"
    )
    parser.add_argument("--embeddings", type=Path, help="Override the saved HDF5 folder if it was moved")
    parser.add_argument("--output", type=Path, help="Default: EXPERIMENT/test/CHECKPOINT_STEM")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Patient bootstrap samples; 0 disables CI"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-in-ram", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=16)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.seed < 0 or args.cache_workers < 1:
        parser.error("Invalid bootstrap count, seed or cache worker count")
    checkpoint = args.checkpoint.resolve()
    experiment = checkpoint.parent.parent
    # Only load checkpoints produced locally by this training pipeline.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    labels_path, split_path = experiment / "labels.csv", experiment / "split.txt"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != config["split_sha256"]:
        raise ValueError("Saved split differs from the checkpoint's training split")
    labels = read_labels(labels_path)
    split = read_split(split_path, labels)
    cases = sorted(case for case, group in split.items() if group == "test")
    cuts = np.asarray(config["time_cuts_years"], dtype=np.float64)
    if (
        len(cuts) != config["model"]["bins"] + 1
        or not np.isfinite(cuts).all()
        or not np.all(np.diff(cuts) > 0)
    ):
        raise ValueError("Invalid time boundaries in checkpoint")
    embeddings = args.embeddings if args.embeddings is not None else Path(config["embeddings"])
    output = args.output if args.output is not None else experiment / "test" / checkpoint.stem
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = MIL(**config["model"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["PYCOX_DATA_DIR"] = str((output / "cache" / "pycox").resolve())
    dataset = Bags(
        cases,
        labels,
        embeddings,
        config["model"]["input_dim"],
        cache_in_ram=args.cache_in_ram,
        cache_workers=args.cache_workers,
    )
    print(f"Test: {len(cases)} patients; checkpoint epoch {saved['epoch']}; device={device}", flush=True)
    rows, total_loss = [], 0.0
    started = last_log = time.monotonic()
    with torch.inference_mode():
        for index in range(len(dataset)):
            features, duration, event, case = dataset[index]
            logits = model(features.to(device=device, dtype=torch.float32))
            if not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite prediction: {case}")
            loss = survival_loss(
                logits,
                torch.tensor([duration], device=device, dtype=torch.float64),
                torch.tensor([event], device=device, dtype=torch.float32),
                cuts,
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite test loss: {case}")
            total_loss += loss.item()
            survival, risk = survival_and_risk(logits, cuts)
            row = dict(case_id=case, event=event, follow_up_years=duration, risk=risk.item())
            row.update({f"hazard_{i}": value for i, value in enumerate(logits.sigmoid()[0].tolist())})
            row.update({f"survival_{i}": value for i, value in enumerate(survival[0].tolist())})
            rows.append(row)
            now = time.monotonic()
            if now - last_log >= 10 or len(rows) == len(cases):
                print(f"Test inference: {len(rows)}/{len(cases)} WSI", flush=True)
                last_log = now
    save_predictions(output / "predictions.csv", rows, config["model"]["bins"])
    print(f"Calculating metrics and {args.bootstrap} patient bootstrap samples", flush=True)
    report = test_metrics(rows, args.bootstrap, args.seed)
    report.update(
        test_loss=total_loss / len(rows),
        checkpoint=str(checkpoint),
        epoch=int(saved["epoch"]),
        checkpoint_validation_metrics=saved["metrics"],
        split_sha256=split_hash,
        labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        embeddings=str(embeddings.resolve()),
        time_cuts_years=cuts.tolist(),
        risk=config["risk"],
        confidence_interval="95% percentile patient bootstrap, conditional on this fixed model",
        seconds=time.monotonic() - started,
    )
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (output / "metrics.json").write_text(text)
    print(text, end="", flush=True)
    print(f"Saved metrics.json and predictions.csv in {output}", flush=True)


if __name__ == "__main__":
    main()
