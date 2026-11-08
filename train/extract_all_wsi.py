#!/usr/bin/env python3
"""Distribute OWL slides between GPUs; details go to OUTPUT/logs/<case>.log."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

from extract_wsi_embeddings import validate_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "snapshot", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--level", type=int, default=1, help="Native OpenSlide level (default: 1)")
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--min-tissue", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        parser.error("GPU indices must be unique nonnegative integers")
    if args.level < 0 or args.stride < 1 or not 0 < args.min_tissue <= 1:
        parser.error("Invalid level, stride or tissue threshold")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or worker count")
    for name in ("config.json", "model.safetensors"):
        if not (args.snapshot / name).is_file():
            parser.error(f"Missing local encoder file: {name}")
    queue = Queue()
    names = set()
    for wsi in sorted((args.raw / "images").rglob("*.tif")):
        mask = args.raw / "tissue_masks" / f"{wsi.stem}_tissue.tif"
        if not mask.is_file() or wsi.stem in names:
            parser.error(f"Missing mask or duplicate case: {wsi}")
        names.add(wsi.stem)
        queue.put(
            SimpleNamespace(
                wsi=wsi,
                mask=mask,
                encoder_snapshot=args.snapshot,
                output=args.output / f"{wsi.stem}.h5",
                level=args.level,
                stride=args.stride,
                min_tissue=args.min_tissue,
            )
        )
    if not names:
        parser.error("No WSI found")
    print(
        f"{len(names)} WSI; GPU {gpus}; OpenSlide level {args.level}; tissue >= {args.min_tissue}", flush=True
    )
    if args.dry_run:
        return
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stop, children, failures = threading.Event(), set(), []

    def interrupt(*_):
        stop.set()
        for process in list(children):
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def worker(gpu):
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except Empty:
                return
            try:
                if job.output.exists():
                    validate_cache(job)
                    print(f"GPU {gpu}: reuse {job.wsi.stem}", flush=True)
                    continue
                command = [sys.executable, "-u", str(Path(__file__).with_name("extract_wsi_embeddings.py"))]
                for name in ("wsi", "mask", "encoder_snapshot", "output", "level", "stride", "min_tissue"):
                    command.extend(["--" + name.replace("_", "-"), str(getattr(job, name))])
                command.extend(["--batch-size", str(args.batch_size), "--workers", str(args.workers)])
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    HF_MODULES_CACHE=str((logs / f"hf_modules_gpu{gpu}").resolve()),
                )
                with (logs / f"{job.wsi.stem}.log").open("a") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    children.add(process)
                    if stop.is_set():
                        process.send_signal(signal.SIGINT)
                    code = process.wait()
                    children.discard(process)
                if stop.is_set():
                    return
                if code:
                    raise RuntimeError(f"extraction exited with {code}; see {log.name}")
                validate_cache(job)
                print(f"GPU {gpu}: complete {job.wsi.stem}", flush=True)
            except Exception as error:
                if stop.is_set():
                    return
                failures.append(job.wsi.stem)
                print(f"GPU {gpu}: FAILED {job.wsi.stem}: {error}", file=sys.stderr, flush=True)

    with (args.output / ".global-extraction.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
    print(f"{'Stopped' if stop.is_set() else 'Finished'}; failed cases: {failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
