#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BUNDLE_DIR = REPO_ROOT / "fixed_wing_vm_bundle_20260416"

if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from control_interface import compute_visual_control_interface  # type: ignore  # noqa: E402
from model import LightweightUNet  # type: ignore  # noqa: E402
from utils import draw_geometry_overlay, extract_mask_geometry, resize_binary_mask  # type: ignore  # noqa: E402


if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
else:
    BILINEAR = Image.BILINEAR


@dataclass
class RunwayInferenceConfig:
    checkpoint_path: str
    config_path: str | None = None
    device: str | None = None
    threshold: float | None = None
    mask_cleaning: str = "bottom_anchor"
    mask_anchor_row_ratio: float = 0.65
    mask_bottom_row_ratio: float = 0.98
    temporal_smoothing: bool = True
    control_row_ratio: float = 0.82
    search_radius: int = 16
    prefer_observed_span_for_lateral: bool = True


def blend_scalar(current: float | None, previous: float | None, alpha: float) -> float | None:
    if current is None:
        return previous
    if previous is None:
        return current
    return float((1.0 - alpha) * previous + alpha * current)


def blend_point(
    current: dict[str, float] | None,
    previous: dict[str, float] | None,
    alpha: float,
) -> dict[str, float] | None:
    if current is None:
        return copy.deepcopy(previous)
    if previous is None:
        return copy.deepcopy(current)
    return {
        "x": float((1.0 - alpha) * previous["x"] + alpha * current["x"]),
        "y": float((1.0 - alpha) * previous["y"] + alpha * current["y"]),
    }


def estimate_temporal_alpha(current: dict[str, Any], previous: dict[str, Any]) -> float:
    current_trend = current.get("trusted_trend", {})
    previous_trend = previous.get("trusted_trend", {})
    current_reliable = int(current_trend.get("reliable_support_points", 0) or 0)
    current_support = int(current_trend.get("support_points", 0) or 0)
    previous_reliable = int(previous_trend.get("reliable_support_points", 0) or 0)

    alpha = 0.55
    if current_reliable <= 2:
        alpha = 0.18
    elif current_reliable == 3:
        alpha = 0.28
    elif current_reliable <= 5:
        alpha = 0.40

    if previous_reliable >= 6 and current_reliable <= 2:
        alpha = min(alpha, 0.12)

    if previous_reliable - current_reliable >= 4:
        alpha = min(alpha, 0.20)

    if current_support <= 4:
        alpha = min(alpha, 0.16)

    return float(np.clip(alpha, 0.08, 0.70))


def should_hold_previous_trend(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    current_trend = current.get("trusted_trend", {})
    previous_trend = previous.get("trusted_trend", {})
    current_angle = current.get("principal_direction", {}).get("angle_deg")
    previous_angle = previous.get("principal_direction", {}).get("angle_deg")

    if not current_trend.get("available") and previous_trend.get("available"):
        return True

    current_reliable = int(current_trend.get("reliable_support_points", 0) or 0)
    previous_reliable = int(previous_trend.get("reliable_support_points", 0) or 0)
    if current_reliable > 2 or previous_reliable < 6:
        return False

    if current_angle is None or previous_angle is None:
        return False

    angle_delta = abs(float(current_angle) - float(previous_angle))
    return angle_delta <= 0.25


def stabilize_video_geometry(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if previous is None:
        return copy.deepcopy(current)

    current_trend = current.get("trusted_trend", {})
    previous_trend = previous.get("trusted_trend", {})
    if not current_trend and not previous_trend:
        return copy.deepcopy(current)

    stabilized = copy.deepcopy(current)

    if should_hold_previous_trend(current, previous):
        held = copy.deepcopy(previous_trend)
        held["support_points"] = current_trend.get("support_points", held.get("support_points"))
        held["reliable_support_points"] = current_trend.get(
            "reliable_support_points",
            held.get("reliable_support_points"),
        )
        held["auxiliary_support_points"] = current_trend.get(
            "auxiliary_support_points",
            held.get("auxiliary_support_points"),
        )
        stabilized["trusted_trend"] = held
        stabilized["principal_direction"] = copy.deepcopy(previous.get("principal_direction", {}))
        return stabilized

    alpha = estimate_temporal_alpha(current, previous)
    far_alpha = max(0.08, alpha * 0.45)
    near_alpha = max(0.10, alpha * 0.80)

    stabilized_trend = copy.deepcopy(current_trend)
    for key in ("start", "end", "observed_start", "observed_end", "center", "ray_anchor", "projected_down_start"):
        stabilized_trend[key] = blend_point(current_trend.get(key), previous_trend.get(key), near_alpha)
    for key in ("ray_end", "projected_down_end"):
        stabilized_trend[key] = blend_point(current_trend.get(key), previous_trend.get(key), far_alpha)

    stabilized_trend["slope_x_per_y"] = blend_scalar(
        current_trend.get("slope_x_per_y"),
        previous_trend.get("slope_x_per_y"),
        alpha,
    )
    stabilized_trend["intercept_x"] = blend_scalar(
        current_trend.get("intercept_x"),
        previous_trend.get("intercept_x"),
        alpha,
    )
    stabilized["trusted_trend"] = stabilized_trend

    stabilized_principal = copy.deepcopy(current.get("principal_direction", {}))
    stabilized_principal["angle_deg"] = blend_scalar(
        current.get("principal_direction", {}).get("angle_deg"),
        previous.get("principal_direction", {}).get("angle_deg"),
        alpha,
    )
    stabilized["principal_direction"] = stabilized_principal
    return stabilized


def clean_mask_by_bottom_anchor(
    mask: np.ndarray,
    anchor_row_ratio: float = 0.65,
    bottom_row_ratio: float = 0.98,
) -> np.ndarray:
    """Keep only mask components connected to the lower/near-field anchor band."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary

    height, _ = binary.shape
    anchor_y = int(np.clip(round((height - 1) * anchor_row_ratio), 0, height - 1))
    bottom_y = int(np.clip(round((height - 1) * bottom_row_ratio), anchor_y, height - 1))
    anchor_band = binary[anchor_y : bottom_y + 1]
    if not anchor_band.any():
        return np.zeros_like(binary, dtype=np.uint8)

    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    anchored_labels = np.unique(labels[anchor_y : bottom_y + 1][anchor_band > 0])
    anchored_labels = anchored_labels[anchored_labels > 0]
    if anchored_labels.size == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    return np.isin(labels, anchored_labels).astype(np.uint8)


class RunwayInferenceEngine:
    def __init__(self, cfg: RunwayInferenceConfig):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)

        checkpoint_path = Path(cfg.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.runtime_config = self._load_runtime_config(cfg, checkpoint)

        self.model = self._build_model(self.runtime_config)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state")
        if state_dict is None:
            raise KeyError("Checkpoint missing model state. Expected 'model_state_dict' or 'model_state'.")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        data_cfg = self.runtime_config["data"]
        self.image_size = tuple(int(value) for value in data_cfg["image_size"])
        mean_values = data_cfg.get("normalize_mean", data_cfg.get("mean"))
        std_values = data_cfg.get("normalize_std", data_cfg.get("std"))
        if mean_values is None or std_values is None:
            raise KeyError("Config missing normalization stats. Expected data.mean/std or data.normalize_mean/std.")
        self.mean = tuple(float(value) for value in mean_values)
        self.std = tuple(float(value) for value in std_values)

        predict_cfg = self.runtime_config.get("predict", self.runtime_config.get("inference", {}))
        self.threshold = float(cfg.threshold if cfg.threshold is not None else predict_cfg["threshold"])

        self._previous_geometry: dict[str, Any] | None = None

    def infer(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        start_time = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prob_map = self._predict_probability(frame_rgb)
        mask = self._postprocess_mask(prob_map, frame_bgr.shape[1], frame_bgr.shape[0])
        if self.cfg.mask_cleaning == "bottom_anchor":
            mask = clean_mask_by_bottom_anchor(
                mask,
                anchor_row_ratio=self.cfg.mask_anchor_row_ratio,
                bottom_row_ratio=self.cfg.mask_bottom_row_ratio,
            )
        geometry = extract_mask_geometry(mask)
        if self.cfg.temporal_smoothing:
            geometry = stabilize_video_geometry(geometry, self._previous_geometry)
        self._previous_geometry = copy.deepcopy(geometry)

        control = compute_visual_control_interface(
            mask=mask,
            geometry=geometry,
            frame_width=frame_bgr.shape[1],
            frame_height=frame_bgr.shape[0],
            control_row_ratio=self.cfg.control_row_ratio,
            search_radius=self.cfg.search_radius,
            prefer_observed_span_for_lateral=self.cfg.prefer_observed_span_for_lateral,
        )

        overlay_bgr = self._build_overlay(frame_rgb, mask, geometry, control)
        elapsed_ms = 1000.0 * (time.perf_counter() - start_time)

        return {
            "mask": (mask.astype(np.uint8) * 255),
            "geometry": geometry,
            "control": control,
            "overlay_bgr": overlay_bgr,
            "prob_map": prob_map,
            "timing": {
                "inference_ms": float(elapsed_ms),
            },
        }

    def reset_temporal_state(self) -> None:
        self._previous_geometry = None

    def _pick_device(self, explicit_device: str | None) -> torch.device:
        if explicit_device:
            return torch.device(explicit_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_runtime_config(self, cfg: RunwayInferenceConfig, checkpoint: dict[str, Any]) -> dict[str, Any]:
        if cfg.config_path:
            config_path = Path(cfg.config_path).expanduser().resolve()
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return checkpoint["config"]

    def _build_model(self, runtime_config: dict[str, Any]) -> LightweightUNet:
        model_cfg = runtime_config["model"]
        return LightweightUNet(
            in_channels=3,
            out_channels=1,
            base_channels=int(model_cfg["base_channels"]),
            dropout=float(model_cfg["dropout"]),
            norm_type=str(model_cfg.get("norm_type", "batchnorm")),
            group_norm_groups=int(model_cfg.get("group_norm_groups", 8)),
        )

    def _preprocess_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        resized = Image.fromarray(frame_rgb, mode="RGB").resize(self.image_size, BILINEAR)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)) / np.array(
            self.std,
            dtype=np.float32,
        ).reshape(1, 1, 3)
        image_np = np.transpose(image_np, (2, 0, 1))
        return torch.from_numpy(image_np).unsqueeze(0)

    def _predict_probability(self, frame_rgb: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess_frame(frame_rgb).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    def _postprocess_mask(self, prob_map: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        binary_small = (prob_map >= self.threshold).astype(np.uint8)
        return resize_binary_mask(binary_small, (frame_width, frame_height))

    def _build_overlay(
        self,
        frame_rgb: np.ndarray,
        mask: np.ndarray,
        geometry: dict[str, Any],
        control: dict[str, Any],
    ) -> np.ndarray:
        overlay_pil = draw_geometry_overlay(frame_rgb, mask, geometry)
        overlay_rgb = np.asarray(overlay_pil.convert("RGB"), dtype=np.uint8).copy()
        self._draw_control_text(overlay_rgb, control)
        return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    def _draw_control_text(self, overlay_rgb: np.ndarray, control: dict[str, Any]) -> None:
        lines = [
            f"valid={int(bool(control['interface_valid']))}",
            f"conf={float(control['confidence']):.3f}",
            f"head_deg={self._fmt(control['heading_error_deg'])}",
            f"lat_hw={self._fmt(control['lateral_error_runway_half_width'])}",
            f"src={control.get('geometry_source') or 'unknown'}",
            f"lat_src={control.get('lateral_source') or 'unknown'}",
        ]

        y = 28
        for line in lines:
            cv2.putText(
                overlay_rgb,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return "None"
        return f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image runway inference smoke test.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave unset for auto.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional segmentation threshold override.")
    parser.add_argument(
        "--mask-cleaning",
        default="bottom_anchor",
        choices=["off", "bottom_anchor"],
        help="Optional mask cleaning strategy.",
    )
    parser.add_argument("--mask-anchor-row-ratio", type=float, default=0.65)
    parser.add_argument("--mask-bottom-row-ratio", type=float, default=0.98)
    parser.add_argument("--output-dir", default=None, help="Optional output directory for mask/overlay/json.")
    parser.add_argument(
        "--temporal-smoothing",
        action="store_true",
        help="Enable temporal smoothing even in standalone mode.",
    )
    return parser.parse_args()


def standalone_main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else image_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    engine = RunwayInferenceEngine(
        RunwayInferenceConfig(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            threshold=args.threshold,
            mask_cleaning=args.mask_cleaning,
            mask_anchor_row_ratio=args.mask_anchor_row_ratio,
            mask_bottom_row_ratio=args.mask_bottom_row_ratio,
            temporal_smoothing=bool(args.temporal_smoothing),
        )
    )
    result = engine.infer(frame_bgr)

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    json_path = output_dir / f"{stem}_control.json"

    cv2.imwrite(str(mask_path), result["mask"])
    cv2.imwrite(str(overlay_path), result["overlay_bgr"])
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result["control"], handle, indent=2)

    print(f"mask: {mask_path}")
    print(f"overlay: {overlay_path}")
    print(f"control: {json_path}")


if __name__ == "__main__":
    standalone_main()
