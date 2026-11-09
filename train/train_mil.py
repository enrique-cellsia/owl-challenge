#!/usr/bin/env python3
"""Train a survival MIL head from cached WSI embeddings and a patient split TXT."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time

import h5py
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from make_split import read_labels, read_split


def time_cuts(records, bins):
    """Event quantiles plus a final boundary at maximum TRAIN follow-up."""
    event_times = [row["follow_up_years"] for row in records if row["event"]]
    if not event_times:
        raise ValueError("Training split has no observed events")
    cuts = np.r_[
        0.0,
        np.quantile(event_times, np.arange(1, bins) / bins),
        max(row["follow_up_years"] for row in records),
    ]
    if not np.all(np.diff(cuts) > 0):
        raise ValueError("Time boundaries are not distinct; use fewer bins")
    return cuts


def survival_loss(logits, durations, events, cuts):
    """pycox NLL; a censor only contributes through fully observed intervals."""
    from pycox.models.loss import nll_logistic_hazard

    ends = torch.as_tensor(cuts[1:], device=logits.device, dtype=durations.dtype)
    # Beyond the training horizon, only survival through that horizon is observed.
    events = events * (durations <= ends[-1])
    event_index = torch.searchsorted(ends, durations, right=False)
    censor_index = torch.searchsorted(ends, durations, right=True) - 1
    index = torch.where(events.bool(), event_index, censor_index)
    observed = index >= 0
    values = nll_logistic_hazard(logits, index.clamp(0, len(ends) - 1), events, reduction="none")
    # Censoring before the first endpoint supplies no complete discrete interval.
    return (values * observed).mean()


def survival_and_risk(logits, cuts):
    survival = torch.sigmoid(-logits).cumprod(dim=-1)
    previous = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=-1)
    widths = torch.as_tensor(np.diff(cuts), device=logits.device, dtype=logits.dtype)
    # Negative trapezoidal area under the survival curve, on a common train horizon.
    risk = -((previous + survival) * 0.5 * widths).sum(dim=-1)
    return survival, risk


class MIL(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, bins=4, dropout=0.25, pooling="attention"):
        super().__init__()
        self.pooling = pooling
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.gate_v = nn.Linear(hidden_dim, 128)
        self.gate_u = nn.Linear(hidden_dim, 128)
        self.attention = nn.Linear(128, 1)
        self.head = nn.Linear(hidden_dim, bins)

    def forward(self, embeddings):
        patches = self.project(embeddings)
        if self.pooling == "attention":
            scores = self.attention(torch.tanh(self.gate_v(patches)) * torch.sigmoid(self.gate_u(patches)))
            bag = (scores.softmax(dim=0) * patches).sum(dim=0)
        else:
            bag = patches.mean(dim=0)
        return self.head(bag).unsqueeze(0)


def read_embeddings(path, input_dim):
    with h5py.File(path, "r") as file:
        features = file["embeddings"]
        if (
            not file.attrs.get("complete", False)
            or file.attrs.get("n_written") != len(features)
            or len(features.shape) != 2
            or features.shape[1] != input_dim
            or not len(features)
        ):
            raise ValueError(f"Incomplete or incompatible embeddings: {path}")
        values = features[:]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite embeddings: {path}")
    return values


class Bags(Dataset):
    """One patient / one WSI per bag. HDF5 is the persistent embedding cache."""

    def __init__(self, cases, labels, folder, input_dim, cache_in_ram=False, cache_workers=4):
        self.cases, self.labels, self.folder = cases, labels, folder
        self.input_dim, self.cache = input_dim, {}
        if cache_in_ram:
            self.preload(cache_workers)

    def preload(self, workers):
        # Processes give each reader its own HDF5 lock; threads share h5py's lock.
        tasks = [(case, self.folder / f"{Path(case).stem}.h5") for case in self.cases]
        sizes = {case: path.stat().st_size for case, path in tasks}
        total = sum(sizes.values())
        loaded = source_loaded = 0
        started = last_log = time.monotonic()
        print(f"RAM cache: {len(tasks)} WSI, ~{total / 1e9:.2f} GB, {workers} readers", flush=True)

        def store(case, values):
            nonlocal loaded, source_loaded, last_log
            self.cache[case] = torch.from_numpy(values)
            loaded += values.nbytes
            source_loaded += sizes[case]
            now = time.monotonic()
            if now - last_log >= 10 or len(self.cache) == len(tasks):
                elapsed = max(now - started, 1e-9)
                remaining = elapsed * (total - source_loaded) / max(source_loaded, 1)
                minutes, seconds = divmod(round(remaining), 60)
                print(
                    f"RAM cache: {len(self.cache)}/{len(tasks)} WSI; {loaded / 1e9:.2f} GB; "
                    f"{loaded / elapsed / 1e9:.3f} GB/s; ETA ~{minutes:02d}:{seconds:02d}",
                    flush=True,
                )
                last_log = now

        if workers == 1:
            for case, path in tasks:
                store(case, read_embeddings(path, self.input_dim))
            return
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            iterator = iter(tasks)
            pending = {}

            def submit_next():
                item = next(iterator, None)
                if item is not None:
                    case, path = item
                    pending[pool.submit(read_embeddings, path, self.input_dim)] = case

            for _ in range(min(workers, len(tasks))):
                submit_next()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    case = pending.pop(future)
                    store(case, future.result())
                    submit_next()

    def __len__(self):
        return len(self.cases)

    def read(self, case):
        path = self.folder / f"{Path(case).stem}.h5"
        return torch.from_numpy(read_embeddings(path, self.input_dim))

    def __getitem__(self, index):
        case = self.cases[index]
        embeddings = self.cache[case] if case in self.cache else self.read(case)
        label = self.labels[case]
        return embeddings, label["follow_up_years"], label["event"], case


def one_bag(sample):
    return sample


def save_predictions(path, predictions, bins):
    fields = ["case_id", "event", "follow_up_years", "risk"]
    fields += [f"hazard_{i}" for i in range(bins)] + [f"survival_{i}" for i in range(bins)]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("labels", "split", "embeddings", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--pooling", choices=("attention", "mean"), default="attention")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-in-ram", action="store_true")
    parser.add_argument(
        "--cache-workers", type=int, default=4, help="Parallel RAM preload readers (default: 4)"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        min(args.epochs, args.input_dim, args.hidden_dim, args.bins, args.accumulation) < 1
        or args.workers < 0
        or args.cache_workers < 1
        or not 0 <= args.dropout < 1
        or not 0 < args.lr < math.inf
        or not 0 <= args.weight_decay < math.inf
    ):
        parser.error("Invalid model or training settings")
    if args.cache_in_ram and args.workers:
        parser.error("Use --workers 0 with --cache-in-ram to avoid duplicating the RAM cache")
    labels = read_labels(args.labels)
    split = read_split(args.split, labels)
    groups = {
        group: sorted(case for case in split if split[case] == group) for group in ("train", "val", "test")
    }
    for group, cases in groups.items():
        print(
            f"{group}: {len(cases)} patients; {sum(labels[case]['event'] for case in cases)} events",
            flush=True,
        )
    cuts = time_cuts([labels[case] for case in groups["train"]], args.bins)
    # Only train and validation have datasets/loaders. Test is never evaluated here.
    for case in groups["train"] + groups["val"]:
        if not (args.embeddings / f"{Path(case).stem}.h5").is_file():
            raise FileNotFoundError(f"No cached embeddings for {case}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    model_args = dict(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        bins=args.bins,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    model = MIL(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # A fresh output directory prevents accidentally mixing runs/checkpoints.
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYCOX_DATA_DIR"] = str((args.output / "cache" / "pycox").resolve())
    checkpoints, predictions = args.output / "checkpoints", args.output / "predictions"
    checkpoints.mkdir()
    predictions.mkdir()
    shutil.copyfile(args.split, args.output / "split.txt")
    shutil.copyfile(args.labels, args.output / "labels.csv")
    shutil.copyfile(__file__, args.output / "train_mil.py")
    shutil.copyfile(Path(__file__).with_name("make_split.py"), args.output / "make_split.py")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "requirements.txt", args.output / "requirements.txt"
    )
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        time_cuts_years=cuts.tolist(),
        risk="negative trapezoidal RMST",
        model=model_args,
        split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
        software=dict(
            torch=torch.__version__,
            numpy=np.__version__,
            pycox=__import__("pycox").__version__,
            sksurv=__import__("sksurv").__version__,
        ),
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    loaders = {}
    for group in ("train", "val"):
        if args.cache_in_ram:
            print(f"Loading {group} embeddings into RAM", flush=True)
        dataset = Bags(
            groups[group], labels, args.embeddings, args.input_dim, args.cache_in_ram, args.cache_workers
        )
        loaders[group] = DataLoader(
            dataset,
            batch_size=None,
            shuffle=group == "train",
            collate_fn=one_bag,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(args.seed),
            **(dict(multiprocessing_context="spawn") if args.workers else {}),
        )
    print(f"Time boundaries (train only): {cuts.tolist()}; device={device}", flush=True)
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "val_loss", "val_c_index", "seconds"]
        )
        writer.writeheader()
        file.flush()
        for epoch in range(1, args.epochs + 1):
            begin = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            n_train = len(loaders["train"])
            for step, (features, duration, event, _) in enumerate(loaders["train"]):
                logits = model(features.to(device=device, dtype=torch.float32))
                loss = survival_loss(
                    logits,
                    torch.tensor([duration], device=device, dtype=torch.float64),
                    torch.tensor([event], device=device, dtype=torch.float32),
                    cuts,
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                group_size = min(args.accumulation, n_train - (step // args.accumulation) * args.accumulation)
                (loss / group_size).backward()
                train_loss += loss.item()
                if (step + 1) % args.accumulation == 0 or step + 1 == n_train:
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            model.eval()
            rows, val_loss = [], 0.0
            with torch.inference_mode():
                for features, duration, event, case in loaders["val"]:
                    logits = model(features.to(device=device, dtype=torch.float32))
                    loss = survival_loss(
                        logits,
                        torch.tensor([duration], device=device, dtype=torch.float64),
                        torch.tensor([event], device=device, dtype=torch.float32),
                        cuts,
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite validation loss")
                    val_loss += loss.item()
                    survival, risk = survival_and_risk(logits, cuts)
                    row = dict(case_id=case, event=event, follow_up_years=duration, risk=risk.item())
                    row.update({f"hazard_{i}": value for i, value in enumerate(logits.sigmoid()[0].tolist())})
                    row.update({f"survival_{i}": value for i, value in enumerate(survival[0].tolist())})
                    rows.append(row)
            try:
                c_index = concordance_index_censored(
                    np.array([bool(row["event"]) for row in rows]),
                    np.array([row["follow_up_years"] for row in rows]),
                    np.array([row["risk"] for row in rows]),
                )[0]
            except ValueError:
                c_index = float("nan")  # No comparable validation pairs.
            metrics = dict(
                epoch=epoch,
                train_loss=train_loss / n_train,
                val_loss=val_loss / len(rows),
                val_c_index=c_index,
                seconds=time.monotonic() - begin,
            )
            destination = checkpoints / f"epoch_{epoch:03d}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(
                dict(
                    epoch=epoch,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    config=config,
                    metrics=metrics,
                ),
                temporary,
            )
            temporary.replace(destination)
            save_predictions(predictions / f"val_epoch_{epoch:03d}.csv", rows, args.bins)
            writer.writerow(metrics)
            file.flush()
            print(json.dumps(metrics), flush=True)
    print(f"Finished: {args.output}. All epochs saved; no best selected and no test evaluation.", flush=True)


if __name__ == "__main__":
    main()
