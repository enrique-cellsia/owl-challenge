#!/usr/bin/env python3
"""Extract frozen Hibou-L embeddings from one WSI into a resumable HDF5."""

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
from types import MethodType

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

MEAN = (0.7068, 0.5755, 0.7220)
STD = (0.1950, 0.2316, 0.1816)
POOLING = "concat(CLS, mean(patch_tokens)); excludes 4 register tokens"


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def starts(length, size, stride):
    if min(length, size, stride) < 1 or size > length:
        raise ValueError("Invalid slide, patch or stride size")
    return np.unique([*range(0, length - size + 1, stride), length - size])


def plan(args):
    with openslide.OpenSlide(str(args.wsi)) as slide:
        width, height = slide.dimensions
        mx, my = [float(slide.properties[f"openslide.mpp-{axis}"]) for axis in ("x", "y")]
        if not all(math.isfinite(v) and v > 0 for v in (mx, my)):
            raise ValueError("Invalid native MPP")
        if args.mpp < max(mx, my) * 0.95:
            raise ValueError("Requested MPP would substantially upsample the source")
        fw, fh = round(224 * args.mpp / mx), round(224 * args.mpp / my)
        sx, sy = round(args.stride * args.mpp / mx), round(args.stride * args.mpp / my)
        level = slide.get_best_level_for_downsample(args.mpp / max(mx, my))
        down = slide.level_downsamples[level]
        geometry = dict(
            wsi_dimensions=[width, height],
            native_mpp=[mx, my],
            footprint_level0=[fw, fh],
            effective_mpp=[fw * mx / 224, fh * my / 224],
            read_level=level,
            read_size=[max(1, round(fw / down)), max(1, round(fh / down))],
            mask_mapping="same full-slide extent, origin (0,0), no rotation",
            edge_policy="append last full patch aligned to each edge",
        )
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(args.mask) as image:
        if len(image.getbands()) != 1:
            raise ValueError("Expected a single-channel binary tissue mask")
        mask = (np.asarray(image) > 0).astype(np.uint8)
    mh, mw = mask.shape
    if abs(width * mh / height - mw) > 2 or mask.size >= np.iinfo(np.int32).max:
        raise ValueError("Mask geometry is incompatible or too large")
    geometry["mask_dimensions"] = [mw, mh]
    integral = cv2.integral(mask, sdepth=cv2.CV_32S)
    xs = starts(width, fw, sx)
    left = np.floor(xs * mw / width).astype(int)
    right = np.minimum(mw, np.ceil((xs + fw) * mw / width).astype(int))
    coords, fractions = [], []
    for y in starts(height, fh, sy):
        top, bottom = int(np.floor(y * mh / height)), min(mh, int(np.ceil((y + fh) * mh / height)))
        count = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
        fraction = count / ((bottom - top) * (right - left))
        keep = fraction >= args.min_tissue
        if keep.any():
            coords.append(np.column_stack((xs[keep], np.full(keep.sum(), y))))
            fractions.append(fraction[keep].astype(np.float32))
    if not coords:
        raise ValueError("No tissue patches retained")
    return np.concatenate(coords), np.concatenate(fractions), geometry


class Tiles(Dataset):
    def __init__(self, wsi, coords, geometry):
        self.wsi, self.coords, self.geometry, self.slide = str(wsi), coords, geometry, None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.wsi)
        image = self.slide.read_region(
            tuple(map(int, self.coords[index])),
            self.geometry["read_level"],
            tuple(self.geometry["read_size"]),
        )
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous()


def sdpa_forward(self, hidden_states, head_mask=None, output_attentions=False):
    if head_mask is not None or output_attentions:
        raise ValueError("Attention maps are not supported by this inference adapter")
    q, k, v = [
        self.transpose_for_scores(layer(hidden_states)) for layer in (self.query, self.key, self.value)
    ]
    values = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    values = values.permute(0, 2, 1, 3).contiguous()
    return (values.view(values.shape[0], values.shape[1], self.all_head_size),)


def contract(args, geometry=None, amp=None):
    result = dict(
        schema_version=1,
        encoder="histai/hibou-L",
        embedding_dim=2048,
        wsi=identity(args.wsi),
        mask=identity(args.mask),
        encoder_snapshot=str(args.encoder_snapshot.resolve()),
        encoder_weights=identity(args.encoder_snapshot / "model.safetensors"),
        encoder_config_sha256=hashlib.sha256(
            (args.encoder_snapshot / "config.json").read_bytes()
        ).hexdigest(),
        pooling=POOLING,
        target_mpp=args.mpp,
        patch_size=224,
        stride=args.stride,
        min_tissue=args.min_tissue,
        normalization=dict(mean=list(MEAN), std=list(STD)),
        interpolation="PIL bicubic",
        storage_dtype="float16",
        coords="top-left x,y in level-0 pixels",
    )
    if geometry is not None:
        result.update(
            geometry=geometry,
            inference_dtype=amp,
            software=dict(
                torch=torch.__version__,
                transformers=__import__("transformers").__version__,
                openslide=openslide.__version__,
                pillow=Image.__version__,
                numpy=np.__version__,
            ),
        )
    return result


def validate_cache(args):
    with h5py.File(args.output, "r") as cache:
        metadata = json.loads(cache.attrs["contract"])
        count, dim = cache["embeddings"].shape
        valid = (
            bool(cache.attrs["complete"])
            and int(cache.attrs["n_written"]) == count
            and count > 0
            and dim == 2048
            and cache["embeddings"].dtype == np.dtype("float16")
            and cache["coords_level0"].shape == (count, 2)
            and cache["tissue_fraction"].shape == (count,)
        )
        if not valid or any(metadata.get(k) != v for k, v in contract(args).items()):
            raise ValueError(f"Incomplete or incompatible cache: {args.output}")
        return count


def extract(args):
    torch.set_num_threads(4)
    coords, fractions, geometry = plan(args)
    print(json.dumps(dict(stage="plan", n_patches=len(coords), geometry=geometry)), flush=True)
    if args.plan_only:
        return
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    amp = str(dtype).split(".")[-1] if device.type == "cuda" else "float32"
    metadata = json.dumps(contract(args, geometry, amp), sort_keys=True)
    partial = Path(str(args.output) + ".partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(args.output) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            raise FileExistsError(args.output)
        with h5py.File(partial, "r+" if partial.exists() else "w") as cache:
            if "contract" in cache.attrs:
                if cache.attrs["contract"] != metadata or not np.array_equal(
                    cache["coords_level0"][:], coords
                ):
                    raise ValueError("Partial cache has a different extraction contract")
            else:
                cache.attrs.update(contract=metadata, n_written=0, complete=False)
                cache.create_dataset(
                    "embeddings",
                    (len(coords), 2048),
                    dtype="float16",
                    chunks=(min(args.batch_size, len(coords)), 2048),
                )
                cache.create_dataset("coords_level0", data=coords)
                cache.create_dataset("tissue_fraction", data=fractions)
            offset = initial = int(cache.attrs["n_written"])
            if not 0 <= offset <= len(coords):
                raise ValueError("Invalid partial progress")
            if offset < len(coords):
                from transformers import AutoModel

                config = json.loads((args.encoder_snapshot / "config.json").read_text())
                if config["hidden_size"] != 1024 or config.get("num_register_tokens") != 4:
                    raise ValueError("This adapter supports Hibou-L only")
                model = (
                    AutoModel.from_pretrained(
                        str(args.encoder_snapshot), local_files_only=True, trust_remote_code=True
                    )
                    .eval()
                    .to(device)
                )
                model.requires_grad_(False)
                attention = [m for m in model.modules() if m.__class__.__name__ == "Dinov2SelfAttention"]
                if len(attention) != config["num_hidden_layers"]:
                    raise ValueError("Unexpected attention architecture")
                for module in attention:
                    module.forward = MethodType(sdpa_forward, module)
                loader = DataLoader(
                    Subset(Tiles(args.wsi, coords, geometry), range(offset, len(coords))),
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=device.type == "cuda",
                    **(dict(multiprocessing_context="spawn", prefetch_factor=2) if args.workers else {}),
                )
                mean, std = [torch.tensor(v, device=device).view(1, 3, 1, 1) for v in (MEAN, STD)]
                begin, last_log = time.monotonic(), 0.0
                with torch.inference_mode():
                    for pixels in loader:
                        pixels = (pixels.to(device=device, dtype=torch.float32) / 255.0 - mean) / std
                        with torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext():
                            tokens = model(pixel_values=pixels).last_hidden_state
                            values = (
                                torch.cat((tokens[:, 0], tokens[:, 5:].mean(1)), dim=1)
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float16)
                            )
                        if values.shape != (len(pixels), 2048) or not np.isfinite(values).all():
                            raise ValueError("Invalid embeddings")
                        cache["embeddings"][offset : offset + len(values)] = values
                        cache.flush()
                        offset += len(values)
                        cache.attrs.modify("n_written", offset)
                        cache.flush()
                        now = time.monotonic()
                        if now - last_log >= 30 or offset == len(coords):
                            print(
                                json.dumps(
                                    dict(
                                        done=offset,
                                        total=len(coords),
                                        patches_per_second=round((offset - initial) / (now - begin), 1),
                                    )
                                ),
                                flush=True,
                            )
                            last_log = now
            cache.attrs.modify("complete", True)
        partial.replace(args.output)
    print(f"Complete: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("wsi", "mask", "encoder-snapshot", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--min-tissue", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (
        not math.isfinite(args.mpp)
        or args.mpp <= 0
        or args.stride < 1
        or not 0 < args.min_tissue <= 1
        or args.batch_size < 1
        or args.workers < 0
    ):
        parser.error("Invalid resolution, tissue threshold or loader settings")
    extract(args)


if __name__ == "__main__":
    main()
