"""CPU checks for geometry, resume and cache compatibility; no model download."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from PIL import Image
import torch
import extract_wsi_embeddings as wsi
import extract_all_wsi as batch


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "images").mkdir()
        (root / "tissue_masks").mkdir()
        snapshot = root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text(
            json.dumps(dict(hidden_size=1024, num_register_tokens=4, num_hidden_layers=0))
        )
        (snapshot / "model.safetensors").write_bytes(b"weights")
        self.args = SimpleNamespace(
            wsi=root / "images/a.tif",
            mask=root / "tissue_masks/a_tissue.tif",
            output=root / "output/a.h5",
            encoder_snapshot=snapshot,
            mpp=0.5,
            stride=224,
            min_tissue=0.05,
            batch_size=1,
            workers=0,
            device="cpu",
            plan_only=False,
        )
        self.args.wsi.write_bytes(b"WSI")
        Image.fromarray(np.array([[1, 0], [1, 0]], dtype=np.uint8)).save(self.args.mask)

    def test_geometry_and_native_level(self):
        class Slide:
            dimensions = (1792, 896)
            properties = {"openslide.mpp-x": ".25", "openslide.mpp-y": ".25"}
            level_downsamples = [1, 4]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def get_best_level_for_downsample(self, _):
                return 1

        self.args.mpp = 1.0
        with patch.object(wsi.openslide, "OpenSlide", return_value=Slide()):
            xy, fraction, geometry = wsi.plan(self.args)
        np.testing.assert_array_equal(xy, [[0, 0]])
        np.testing.assert_array_equal(fraction, [1.0])
        self.assertEqual(geometry["read_size"], [224, 224])
        self.assertEqual(geometry["read_level"], 1)
        self.assertEqual(wsi.starts(1000, 224, 224).tolist(), [0, 224, 448, 672, 776])

    def test_resume_preserves_rows_and_excludes_register_tokens(self):
        class Encoder(torch.nn.Module):
            def forward(self, pixel_values):
                tokens = torch.full((len(pixel_values), 7, 1024), 100.0)
                tokens[:, 0], tokens[:, 5], tokens[:, 6] = 2.0, 4.0, 6.0
                return SimpleNamespace(last_hidden_state=tokens)

        module = SimpleNamespace(
            __version__="test", AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: Encoder())
        )
        xy = np.array([[0, 0], [224, 0]], dtype=np.int64)
        geometry = dict(read_level=0, read_size=[224, 224])

        def tile(_, index):
            if index == 1:
                raise RuntimeError("interrupted read")
            return torch.zeros((3, 224, 224), dtype=torch.uint8)

        with (
            patch.dict(sys.modules, transformers=module),
            patch.object(wsi, "plan", return_value=(xy, np.ones(2, dtype=np.float32), geometry)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with patch.object(wsi.Tiles, "__getitem__", tile):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    wsi.extract(self.args)
            self.assertFalse(self.args.output.exists())
            with h5py.File(str(self.args.output) + ".partial", "r") as cache:
                self.assertEqual(cache.attrs["n_written"], 1)
            with patch.object(
                wsi.Tiles, "__getitem__", return_value=torch.zeros((3, 224, 224), dtype=torch.uint8)
            ):
                wsi.extract(self.args)
        self.assertEqual(wsi.validate_cache(self.args), 2)
        with h5py.File(self.args.output, "r") as cache:
            values = cache["embeddings"][:]
            np.testing.assert_array_equal(values[:, :1024], 2.0)
            np.testing.assert_array_equal(values[:, 1024:], 5.0)
        with self.assertRaises(FileExistsError):
            with (
                patch.object(wsi, "plan", return_value=(xy, np.ones(2), geometry)),
                patch.dict(sys.modules, transformers=module),
            ):
                wsi.extract(self.args)
        self.args.mpp = 1.0
        with self.assertRaises(ValueError):
            wsi.validate_cache(self.args)
        self.args.mpp = 0.5
        (self.args.encoder_snapshot / "model.safetensors").write_bytes(b"other weights")
        with self.assertRaises(ValueError):
            wsi.validate_cache(self.args)

    def test_batch_dry_run_creates_nothing(self):
        argv = [
            "batch",
            "--raw",
            str(self.args.wsi.parent.parent),
            "--snapshot",
            str(self.args.encoder_snapshot),
            "--output",
            str(self.args.output.parent),
            "--dry-run",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(batch.subprocess, "Popen") as popen,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            batch.main()
        popen.assert_not_called()
        self.assertFalse(self.args.output.parent.exists())


if __name__ == "__main__":
    unittest.main()
