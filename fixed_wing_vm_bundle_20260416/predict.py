from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import (
    RunwaySegmentationDataset,
    get_all_data_split_paths,
    load_all_records,
    load_fold_records,
    read_split_csv,
)
from model import LightweightUNet
from utils import (
    build_overlay,
    compute_segmentation_metrics,
    denormalize_image,
    ensure_dir,
    draw_mask_panel,
    extract_mask_geometry,
    resize_binary_mask,
    save_image_grid,
    save_json,
    save_sequence_gif,
)


def build_model(config: dict[str, Any]) -> LightweightUNet:
    model_cfg = config["model"]
    return LightweightUNet(
        in_channels=3,
        out_channels=1,
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg["dropout"]),
        norm_type=str(model_cfg.get("norm_type", "batchnorm")),
        group_norm_groups=int(model_cfg.get("group_norm_groups", 8)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and visualization for runway segmentation.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--test-dataset", default=None, help="Dataset split name, e.g. dataset_1.")
    parser.add_argument("--split-csv", default=None, help="Optional explicit CSV path.")
    parser.add_argument(
        "--split-name",
        default=None,
        choices=["train", "test", "all_data", "monitor"],
        help="Defaults to test for loo checkpoints and all_data for full-data checkpoints.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for smoke tests.")
    return parser.parse_args()


def pick_device(explicit_device: str | None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config_from_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if args.config:
        import json

        with Path(args.config).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return checkpoint["config"]


def resolve_split_name(args: argparse.Namespace, checkpoint: dict[str, Any]) -> str:
    if args.split_name:
        return args.split_name
    if checkpoint.get("mode") == "all_data" or checkpoint.get("split_name") == "all_data":
        return "all_data"
    return "test"


def load_records(config: dict[str, Any], args: argparse.Namespace, checkpoint: dict[str, Any]) -> tuple[list[Any], str]:
    data_root = config["paths"]["data_root"]
    metadata_root = config["paths"]["metadata_root"]
    split_name = resolve_split_name(args, checkpoint)

    if args.split_csv:
        records = read_split_csv(args.split_csv, data_root)
        split_name = split_name or "custom"
    elif split_name == "monitor":
        split_prefix = checkpoint.get("monitor_split", {}).get("split_prefix") or config["all_data"]["split_prefix"]
        _, monitor_csv, _ = get_all_data_split_paths(metadata_root, split_prefix)
        records = read_split_csv(monitor_csv, data_root)
    elif split_name == "all_data":
        records = load_all_records(
            data_root=data_root,
            metadata_root=metadata_root,
        )
    else:
        test_dataset = args.test_dataset or checkpoint.get("test_dataset")
        if not test_dataset:
            raise ValueError("Need --test-dataset when checkpoint does not store a fold name.")
        train_records, test_records = load_fold_records(
            data_root=data_root,
            metadata_root=metadata_root,
            test_dataset=test_dataset,
        )
        records = train_records if split_name == "train" else test_records

    if args.max_samples is not None:
        records = records[: args.max_samples]
    return records, split_name


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config_from_args(args, checkpoint)
    device = pick_device(args.device)
    threshold = float(config["predict"]["threshold"])

    records, split_name = load_records(config, args, checkpoint)
    dataset = RunwaySegmentationDataset(
        records=records,
        image_size=tuple(config["data"]["image_size"]),
        mean=tuple(config["data"]["mean"]),
        std=tuple(config["data"]["std"]),
        augment=False,
        augmentation_cfg=config["augmentation"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    checkpoint_path = Path(args.checkpoint).resolve()
    default_output_dir = checkpoint_path.parent.parent / "predictions"
    output_dir = ensure_dir(args.output_dir or default_output_dir)
    masks_dir = ensure_dir(output_dir / "masks")
    overlays_dir = ensure_dir(output_dir / "overlays")
    comparisons_dir = ensure_dir(output_dir / "comparisons")
    gifs_dir = ensure_dir(output_dir / "gifs")
    geometry_dir = ensure_dir(output_dir / "geometry")

    metrics_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    representative_panels: list[Image.Image] = []
    overlay_frames: dict[str, list[Any]] = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            pred_mask = (probs >= threshold).float()

            metrics = compute_segmentation_metrics(logits, masks, threshold=threshold)

            image_np = denormalize_image(
                images[0].cpu(),
                mean=tuple(config["data"]["mean"]),
                std=tuple(config["data"]["std"]),
            )
            gt_mask_np = masks[0, 0].cpu().numpy().astype(np.uint8)
            pred_mask_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8)
            dataset_id = batch["dataset_id"][0]
            frame_id = batch["frame_id"][0]
            file_stem = f"{dataset_id}_{frame_id}"

            pred_image = build_overlay(image_np, pred_mask_np, color="#f97316")
            comparison_panel = draw_mask_panel(image_np, gt_mask_np, pred_mask_np, title=file_stem)
            if len(representative_panels) < int(config["predict"]["representative_frames"]):
                representative_panels.append(comparison_panel)

            resized_pred_mask = resize_binary_mask(
                pred_mask_np,
                (int(batch["original_width"][0]), int(batch["original_height"][0])),
            )
            geometry = extract_mask_geometry(resized_pred_mask)
            Image.fromarray((resized_pred_mask * 255).astype(np.uint8), mode="L").save(
                masks_dir / f"{file_stem}.png"
            )
            pred_image.save(overlays_dir / f"{file_stem}.jpg", quality=95)
            comparison_panel.save(comparisons_dir / f"{file_stem}.jpg", quality=95)

            overlay_frames.setdefault(dataset_id, []).append(pred_image)
            metrics_rows.append(
                {
                    "dataset_id": dataset_id,
                    "frame_id": frame_id,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "mask_path": str(masks_dir / f"{file_stem}.png"),
                    "overlay_path": str(overlays_dir / f"{file_stem}.jpg"),
                }
            )
            geometry_rows.append(
                {
                    "dataset_id": dataset_id,
                    "frame_id": frame_id,
                    "foreground_pixels": geometry["foreground_pixels"],
                    "runway_ratio": geometry["runway_ratio"],
                    "bbox": geometry["bbox"],
                    "geometry_width": int(batch["original_width"][0]),
                    "geometry_height": int(batch["original_height"][0]),
                    "principal_direction": geometry["principal_direction"],
                    "centerline": geometry["centerline"],
                }
            )

    write_csv(metrics_rows, output_dir / "metrics.csv")
    save_json(
        {"predictions": geometry_rows},
        geometry_dir / "geometry_summary.json",
    )

    if representative_panels:
        save_image_grid(representative_panels, output_dir / "representative_panels.jpg", columns=2)

    for dataset_id, frames in overlay_frames.items():
        save_sequence_gif(
            frames,
            gifs_dir / f"{dataset_id}_{split_name}.gif",
            duration_ms=int(config["predict"]["gif_frame_duration_ms"]),
        )

    summary = {
        "mode": checkpoint.get("mode", "loo"),
        "split_name": split_name,
        "num_samples": len(metrics_rows),
        "mean_dice": float(np.mean([row["dice"] for row in metrics_rows])) if metrics_rows else 0.0,
        "mean_iou": float(np.mean([row["iou"] for row in metrics_rows])) if metrics_rows else 0.0,
        "output_dir": str(output_dir),
    }
    save_json(summary, output_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
