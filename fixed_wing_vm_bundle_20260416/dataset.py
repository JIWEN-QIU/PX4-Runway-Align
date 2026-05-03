from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
    import cv2
except ImportError:
    A = None
    cv2 = None


if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
else:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


@dataclass(frozen=True)
class SampleRecord:
    dataset_id: str
    frame_id: str
    image_path: Path
    mask_path: Path
    json_path: Path
    width: int
    height: int
    runway_ratio: float


CSV_FIELDNAMES = [
    "dataset",
    "frame_id",
    "image_relpath",
    "mask_relpath",
    "json_relpath",
    "width",
    "height",
    "runway_ratio",
]


def read_split_csv(split_csv_path: str | Path, data_root: str | Path) -> list[SampleRecord]:
    split_csv_path = Path(split_csv_path)
    data_root = Path(data_root)
    records: list[SampleRecord] = []

    with split_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SampleRecord(
                    dataset_id=row["dataset"],
                    frame_id=row["frame_id"],
                    image_path=(data_root / row["image_relpath"]).resolve(),
                    mask_path=(data_root / row["mask_relpath"]).resolve(),
                    json_path=(data_root / row["json_relpath"]).resolve(),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    runway_ratio=float(row["runway_ratio"]),
                )
            )

    return records


def get_all_samples_csv_path(metadata_root: str | Path) -> Path:
    metadata_root = Path(metadata_root)
    all_samples_csv = metadata_root / "all_samples.csv"
    if not all_samples_csv.exists():
        raise FileNotFoundError(f"Could not find all_samples.csv under {metadata_root}")
    return all_samples_csv


def get_all_data_split_paths(metadata_root: str | Path, split_prefix: str) -> tuple[Path, Path, Path]:
    metadata_root = Path(metadata_root)
    split_dir = metadata_root / "splits"
    train_csv = split_dir / f"{split_prefix}_train.csv"
    monitor_csv = split_dir / f"{split_prefix}_monitor.csv"
    summary_json = split_dir / f"{split_prefix}_summary.json"
    return train_csv, monitor_csv, summary_json


def get_fold_split_paths(metadata_root: str | Path, test_dataset: str) -> tuple[Path, Path]:
    metadata_root = Path(metadata_root)
    split_dir = metadata_root / "splits"
    train_csv = split_dir / f"loo_{test_dataset}_train.csv"
    test_csv = split_dir / f"loo_{test_dataset}_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Could not find LOO split files for {test_dataset} under {split_dir}"
        )

    return train_csv, test_csv


def write_split_csv(records: list[SampleRecord], path: str | Path, data_root: str | Path) -> None:
    path = Path(path)
    data_root = Path(data_root).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset": record.dataset_id,
                    "frame_id": record.frame_id,
                    "image_relpath": record.image_path.resolve().relative_to(data_root).as_posix(),
                    "mask_relpath": record.mask_path.resolve().relative_to(data_root).as_posix(),
                    "json_relpath": record.json_path.resolve().relative_to(data_root).as_posix(),
                    "width": record.width,
                    "height": record.height,
                    "runway_ratio": record.runway_ratio,
                }
            )


def _frame_sort_key(frame_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in frame_id if ch.isdigit())
    if digits:
        return int(digits), frame_id
    return -1, frame_id


def _select_monitor_chunk(
    ordered_records: list[SampleRecord],
    monitor_count: int,
    monitor_position: str,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if monitor_count <= 0 or monitor_count >= len(ordered_records):
        raise ValueError("monitor_count must be between 1 and len(records) - 1")

    if monitor_position == "tail":
        start = len(ordered_records) - monitor_count
    elif monitor_position == "head":
        start = 0
    elif monitor_position == "middle":
        start = (len(ordered_records) - monitor_count) // 2
    else:
        raise ValueError(f"Unsupported monitor_position: {monitor_position}")

    end = start + monitor_count
    monitor_records = ordered_records[start:end]
    train_records = ordered_records[:start] + ordered_records[end:]
    return train_records, monitor_records


def build_all_data_monitor_split(
    all_records: list[SampleRecord],
    monitor_ratio: float,
    min_monitor_samples_per_dataset: int,
    monitor_position: str,
) -> tuple[list[SampleRecord], list[SampleRecord], dict[str, Any]]:
    dataset_groups: dict[str, list[SampleRecord]] = {}
    for record in all_records:
        dataset_groups.setdefault(record.dataset_id, []).append(record)

    train_records: list[SampleRecord] = []
    monitor_records: list[SampleRecord] = []
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_id in sorted(dataset_groups.keys()):
        ordered_records = sorted(dataset_groups[dataset_id], key=lambda item: _frame_sort_key(item.frame_id))
        dataset_size = len(ordered_records)
        requested_monitor = max(min_monitor_samples_per_dataset, int(round(dataset_size * monitor_ratio)))
        monitor_count = min(max(1, requested_monitor), dataset_size - 1)
        dataset_train, dataset_monitor = _select_monitor_chunk(
            ordered_records=ordered_records,
            monitor_count=monitor_count,
            monitor_position=monitor_position,
        )

        train_records.extend(dataset_train)
        monitor_records.extend(dataset_monitor)
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "total_samples": dataset_size,
                "train_samples": len(dataset_train),
                "monitor_samples": len(dataset_monitor),
                "monitor_first_frame": dataset_monitor[0].frame_id,
                "monitor_last_frame": dataset_monitor[-1].frame_id,
            }
        )

    split_summary = {
        "monitor_ratio": monitor_ratio,
        "min_monitor_samples_per_dataset": min_monitor_samples_per_dataset,
        "monitor_position": monitor_position,
        "num_train_samples": len(train_records),
        "num_monitor_samples": len(monitor_records),
        "datasets": dataset_summaries,
    }
    return train_records, monitor_records, split_summary


def create_all_data_monitor_split_files(
    data_root: str | Path,
    metadata_root: str | Path,
    split_prefix: str,
    monitor_ratio: float,
    min_monitor_samples_per_dataset: int,
    monitor_position: str,
) -> dict[str, Any]:
    data_root = Path(data_root)
    metadata_root = Path(metadata_root)
    all_records = load_all_records(data_root=data_root, metadata_root=metadata_root)
    train_records, monitor_records, split_summary = build_all_data_monitor_split(
        all_records=all_records,
        monitor_ratio=monitor_ratio,
        min_monitor_samples_per_dataset=min_monitor_samples_per_dataset,
        monitor_position=monitor_position,
    )
    train_csv, monitor_csv, summary_json = get_all_data_split_paths(metadata_root, split_prefix)
    write_split_csv(train_records, train_csv, data_root)
    write_split_csv(monitor_records, monitor_csv, data_root)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    import json

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2)

    return {
        "train_csv": train_csv,
        "monitor_csv": monitor_csv,
        "summary_json": summary_json,
        "summary": split_summary,
    }


def load_all_records(
    data_root: str | Path,
    metadata_root: str | Path,
    max_samples: int | None = None,
) -> list[SampleRecord]:
    all_samples_csv = get_all_samples_csv_path(metadata_root)
    records = read_split_csv(all_samples_csv, data_root)
    if max_samples is not None:
        records = records[:max_samples]
    return records


def load_all_data_monitor_records(
    data_root: str | Path,
    metadata_root: str | Path,
    split_prefix: str,
    monitor_ratio: float,
    min_monitor_samples_per_dataset: int,
    monitor_position: str,
    max_train_samples: int | None = None,
    max_monitor_samples: int | None = None,
) -> tuple[list[SampleRecord], list[SampleRecord], dict[str, Any]]:
    split_artifacts = create_all_data_monitor_split_files(
        data_root=data_root,
        metadata_root=metadata_root,
        split_prefix=split_prefix,
        monitor_ratio=monitor_ratio,
        min_monitor_samples_per_dataset=min_monitor_samples_per_dataset,
        monitor_position=monitor_position,
    )
    train_records = read_split_csv(split_artifacts["train_csv"], data_root)
    monitor_records = read_split_csv(split_artifacts["monitor_csv"], data_root)

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_monitor_samples is not None:
        monitor_records = monitor_records[:max_monitor_samples]

    split_info = {
        "split_prefix": split_prefix,
        "train_csv": str(split_artifacts["train_csv"]),
        "monitor_csv": str(split_artifacts["monitor_csv"]),
        "summary_json": str(split_artifacts["summary_json"]),
        "summary": split_artifacts["summary"],
    }
    return train_records, monitor_records, split_info


def load_fold_records(
    data_root: str | Path,
    metadata_root: str | Path,
    test_dataset: str,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    train_csv, test_csv = get_fold_split_paths(metadata_root, test_dataset)
    train_records = read_split_csv(train_csv, data_root)
    val_records = read_split_csv(test_csv, data_root)

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_val_samples is not None:
        val_records = val_records[:max_val_samples]

    return train_records, val_records


class RunwaySegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        image_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        augment: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.augment = augment
        self.augmentation_cfg = augmentation_cfg or {}
        self.augmentation_backend = str(self.augmentation_cfg.get("backend", "pil_light")).lower()
        self.albumentations_transform = self._build_albumentations_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path).convert("L")

        if self.augment and self.augmentation_cfg.get("enabled", False):
            image, mask = self._apply_augmentation(image, mask)

        image = image.resize(self.image_size, BILINEAR)
        mask = mask.resize(self.image_size, NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_np = np.transpose(image_np, (2, 0, 1))

        mask_np = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        mask_np = np.expand_dims(mask_np, axis=0)

        return {
            "image": torch.from_numpy(image_np),
            "mask": torch.from_numpy(mask_np),
            "dataset_id": record.dataset_id,
            "frame_id": record.frame_id,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "json_path": str(record.json_path),
            "original_width": record.width,
            "original_height": record.height,
            "runway_ratio": record.runway_ratio,
        }

    def _apply_augmentation(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.augmentation_backend == "albumentations_v1":
            return self._apply_albumentations_augmentation(image, mask)
        return self._apply_light_augmentation(image, mask)

    def _apply_light_augmentation(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        image, mask = self._random_crop_zoom(image, mask)

        brightness = float(self.augmentation_cfg.get("brightness", 0.0))
        if brightness > 0:
            factor = random.uniform(1.0 - brightness, 1.0 + brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        contrast = float(self.augmentation_cfg.get("contrast", 0.0))
        if contrast > 0:
            factor = random.uniform(1.0 - contrast, 1.0 + contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        blur_prob = float(self.augmentation_cfg.get("blur_prob", 0.0))
        if random.random() < blur_prob:
            blur_radius_max = float(self.augmentation_cfg.get("blur_radius_max", 1.0))
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, blur_radius_max)))

        return image, mask

    def _build_albumentations_transform(self) -> Any:
        if self.augmentation_backend != "albumentations_v1":
            return None

        if A is None or cv2 is None:
            raise ImportError(
                "albumentations_v1 augmentation requested, but albumentations/cv2 is not available."
            )

        transforms: list[Any] = []

        affine_prob = float(self.augmentation_cfg.get("affine_prob", 0.0))
        if affine_prob > 0:
            transforms.append(
                A.Affine(
                    scale=tuple(self.augmentation_cfg.get("affine_scale", [0.97, 1.03])),
                    translate_percent=tuple(self.augmentation_cfg.get("affine_translate_percent", [-0.03, 0.03])),
                    rotate=tuple(self.augmentation_cfg.get("affine_rotate", [-2.0, 2.0])),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=affine_prob,
                )
            )

        perspective_prob = float(self.augmentation_cfg.get("perspective_prob", 0.0))
        if perspective_prob > 0:
            transforms.append(
                A.Perspective(
                    scale=tuple(self.augmentation_cfg.get("perspective_scale", [0.02, 0.05])),
                    keep_size=True,
                    fit_output=False,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=perspective_prob,
                )
            )

        brightness_contrast_prob = float(self.augmentation_cfg.get("brightness_contrast_prob", 0.0))
        if brightness_contrast_prob > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=float(self.augmentation_cfg.get("brightness_limit", 0.12)),
                    contrast_limit=float(self.augmentation_cfg.get("contrast_limit", 0.12)),
                    p=brightness_contrast_prob,
                )
            )

        gamma_prob = float(self.augmentation_cfg.get("gamma_prob", 0.0))
        if gamma_prob > 0:
            gamma_limit = self.augmentation_cfg.get("gamma_limit", [90, 110])
            transforms.append(
                A.RandomGamma(
                    gamma_limit=(int(gamma_limit[0]), int(gamma_limit[1])),
                    p=gamma_prob,
                )
            )

        noise_prob = float(self.augmentation_cfg.get("noise_prob", 0.0))
        if noise_prob > 0:
            noise_std_range = self.augmentation_cfg.get("noise_std_range", [0.01, 0.03])
            transforms.append(
                A.GaussNoise(
                    std_range=(float(noise_std_range[0]), float(noise_std_range[1])),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    noise_scale_factor=1.0,
                    p=noise_prob,
                )
            )

        compression_prob = float(self.augmentation_cfg.get("compression_prob", 0.0))
        if compression_prob > 0:
            compression_quality = self.augmentation_cfg.get("compression_quality_range", [70, 95])
            transforms.append(
                A.ImageCompression(
                    quality_range=(int(compression_quality[0]), int(compression_quality[1])),
                    compression_type="jpeg",
                    p=compression_prob,
                )
            )

        if not transforms:
            return None

        return A.Compose(transforms)

    def _apply_albumentations_augmentation(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if self.albumentations_transform is None:
            return image, mask

        image_np = np.asarray(image, dtype=np.uint8)
        mask_np = np.asarray(mask, dtype=np.uint8)
        transformed = self.albumentations_transform(image=image_np, mask=mask_np)
        transformed_image = Image.fromarray(transformed["image"], mode="RGB")
        transformed_mask = Image.fromarray(transformed["mask"], mode="L")
        return transformed_image, transformed_mask

    def _random_crop_zoom(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        crop_scale_min = float(self.augmentation_cfg.get("crop_scale_min", 1.0))
        crop_scale_min = min(max(crop_scale_min, 0.8), 1.0)

        if crop_scale_min >= 0.999:
            return image, mask

        scale = random.uniform(crop_scale_min, 1.0)
        crop_width = max(1, int(round(image.width * scale)))
        crop_height = max(1, int(round(image.height * scale)))

        if crop_width == image.width and crop_height == image.height:
            return image, mask

        left = random.randint(0, image.width - crop_width)
        top = random.randint(0, image.height - crop_height)
        box = (left, top, left + crop_width, top + crop_height)

        cropped_image = image.crop(box).resize(image.size, BILINEAR)
        cropped_mask = mask.crop(box).resize(mask.size, NEAREST)
        return cropped_image, cropped_mask
