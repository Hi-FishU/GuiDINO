import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - handled by unittest skip
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled by unittest skip
    Image = None

try:
    import albumentations  # noqa: F401
    import pytorch_lightning  # noqa: F401
except ImportError:  # pragma: no cover - handled by unittest skip
    albumentations = None
    pytorch_lightning = None

from data import segmentation as seg


def _save_rgb(path: Path, color: int = 0) -> None:
    array = np.full((4, 4, 3), color, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _save_mask(path: Path, value: int = 0) -> None:
    array = np.full((4, 4), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


@unittest.skipUnless(torch is not None, "torch is required for segmentation tests")
@unittest.skipUnless(Image is not None, "Pillow is required for segmentation tests")
@unittest.skipUnless(
    albumentations is not None and pytorch_lightning is not None,
    "albumentations and pytorch_lightning are required for segmentation tests",
)
class SegmentationDataTests(unittest.TestCase):
    def test_drive_mask_name(self):
        self.assertEqual(seg._drive_mask_name("01_test", use_manual=True), "01_manual1.gif")
        self.assertEqual(seg._drive_mask_name("01_test", use_manual=False), "01_test_mask.gif")

    def test_discover_drive_samples_filters_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "drive"
            image_dir = root / "training" / "images"
            manual_dir = root / "training" / "1st_manual"
            image_dir.mkdir(parents=True)
            manual_dir.mkdir(parents=True)

            _save_rgb(image_dir / "01_test.png", color=10)
            _save_rgb(image_dir / "02_test.png", color=20)
            _save_mask(manual_dir / "01_manual1.gif", value=255)

            samples = seg.discover_drive_samples(root, split="training", use_manual=True)
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.name, "01_test")
            self.assertEqual(sample.source, "drive")
            self.assertEqual(sample.image_path.name, "01_test.png")
            self.assertEqual(sample.mask_path.name, "01_manual1.gif")

    def test_discover_kvasir_samples(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "kvasir"
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            _save_rgb(image_dir / "k1.png", color=10)
            _save_mask(mask_dir / "k1.png", value=255)
            _save_rgb(image_dir / "k2.png", color=20)

            samples = seg.discover_kvasir_samples(root)
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.name, "k1")
            self.assertEqual(sample.source, "kvasir")

    def test_split_samples_deterministic(self):
        samples = [
            seg.SegmentationSample(Path(f"img_{i}.png"), Path(f"mask_{i}.png"), f"name_{i}", "src")
            for i in range(5)
        ]
        train_a, val_a = seg._split_samples(samples, val_split=0.2, seed=123)
        train_b, val_b = seg._split_samples(samples, val_split=0.2, seed=123)
        self.assertEqual([s.name for s in train_a], [s.name for s in train_b])
        self.assertEqual([s.name for s in val_a], [s.name for s in val_b])
        self.assertEqual(len(val_a), 1)
        self.assertEqual(len(train_a), 4)

    def test_generic_dataset_requires_samples(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            seg.GenericSegmentationDataset([])

    def test_generic_dataset_without_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "img.png"
            mask_path = Path(tmp_dir) / "mask.png"
            _save_rgb(image_path, color=255)
            _save_mask(mask_path, value=128)

            sample = seg.SegmentationSample(image_path, mask_path, "sample", "src")
            dataset = seg.GenericSegmentationDataset([sample], transform=None, mask_threshold=0.5)
            image, mask = dataset[0]
            self.assertEqual(image.shape, (3, 4, 4))
            self.assertEqual(mask.shape, (1, 4, 4))
            self.assertTrue(torch.all((mask == 0) | (mask == 1)))
            self.assertTrue(torch.isfinite(image).all())

    def test_datamodule_setup_and_dataloaders(self):
        def _simple_transform(image, mask):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).float()
            return {"image": image_tensor, "mask": mask_tensor}

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(seg, "get_segmentation_train_transform", lambda *_: _simple_transform),
            mock.patch.object(seg, "get_segmentation_val_transform", lambda *_: _simple_transform),
        ):
            drive_root = Path(tmp_dir) / "drive"
            drive_images = drive_root / "training" / "images"
            drive_masks = drive_root / "training" / "1st_manual"
            drive_images.mkdir(parents=True)
            drive_masks.mkdir(parents=True)
            for idx in range(2):
                _save_rgb(drive_images / f"{idx}_img.png", color=10 + idx)
                _save_mask(drive_masks / f"{idx}_manual1.gif", value=255)

            kvasir_root = Path(tmp_dir) / "kvasir"
            kvasir_images = kvasir_root / "images"
            kvasir_masks = kvasir_root / "masks"
            kvasir_images.mkdir(parents=True)
            kvasir_masks.mkdir(parents=True)
            for idx in range(2):
                _save_rgb(kvasir_images / f"k{idx}.png", color=20 + idx)
                _save_mask(kvasir_masks / f"k{idx}.png", value=255)

            dm = seg.MedTokenSegmentationDataModule(
                drive_root=str(drive_root),
                kvasir_root=str(kvasir_root),
                batch_size=2,
                num_workers=0,
                drive_val_split=0.5,
                kvasir_val_split=0.5,
                seed=0,
                pin_memory=False,
                persistent_workers=False,
            )
            dm.setup()

            train_loader = dm.train_dataloader()
            val_loader = dm.val_dataloader()
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)

            images, masks = next(iter(train_loader))
            self.assertEqual(images.shape[0], 2)
            self.assertEqual(masks.shape[1], 1)
