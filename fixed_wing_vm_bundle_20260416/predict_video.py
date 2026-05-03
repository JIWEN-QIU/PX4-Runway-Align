from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from control_interface import compute_visual_control_interface
from model import LightweightUNet
from utils import draw_geometry_overlay, ensure_dir, extract_mask_geometry, resize_binary_mask


if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
else:
    BILINEAR = Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runway segmentation and geometry visualization on a video.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint, typically best.pt.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--config", default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--output-video", default=None, help="Optional output .mp4 path.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional prediction threshold override.")
    parser.add_argument(
        "--visualization",
        default="side_by_side",
        choices=["overlay", "side_by_side"],
        help="Visualization layout for the output video.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--fps", type=float, default=None, help="Optional output fps override.")
    parser.add_argument("--codec", default="MJPG", help="FourCC codec for VideoWriter, e.g. MJPG, mp4v, or avc1.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N processed frames.")
    parser.add_argument(
        "--temporal-smoothing",
        default="trend",
        choices=["off", "trend"],
        help="Apply lightweight temporal stabilization for video geometry overlays.",
    )
    return parser.parse_args()


def pick_device(explicit_device: str | None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config_from_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return checkpoint["config"]


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


def default_output_video_path(config: dict[str, Any], video_path: Path, output_dir_override: str | None) -> Path:
    if output_dir_override:
        output_dir = ensure_dir(output_dir_override)
    else:
        output_root = Path(config["paths"]["output_root"]).resolve()
        output_dir = ensure_dir(output_root / "test_videos" / video_path.stem)
    return output_dir / f"{video_path.stem}_overlay.avi"


def preprocess_frame(
    frame_rgb: np.ndarray,
    image_size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    resized = Image.fromarray(frame_rgb, mode="RGB").resize(image_size, BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    image_np = (image_np - np.array(mean, dtype=np.float32).reshape(1, 1, 3)) / np.array(
        std,
        dtype=np.float32,
    ).reshape(1, 1, 3)
    image_np = np.transpose(image_np, (2, 0, 1))
    return torch.from_numpy(image_np).unsqueeze(0)


def format_overlay_text(frame_index: int, timestamp_sec: float, geometry: dict[str, Any]) -> str:
    trusted_trend = geometry.get("trusted_trend", {})
    principal_direction = geometry.get("principal_direction", {})
    parts = [
        f"frame={frame_index:06d}",
        f"time={timestamp_sec:.2f}s",
        f"ratio={geometry.get('runway_ratio', 0.0):.4f}",
        f"support={int(trusted_trend.get('support_points', 0) or 0)}",
        f"reliable={int(trusted_trend.get('reliable_support_points', 0) or 0)}",
    ]
    angle_deg = principal_direction.get("angle_deg")
    if angle_deg is not None:
        parts.append(f"angle={float(angle_deg):.2f}")
    return " | ".join(parts)


def annotate_image(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    bar_height = 30
    draw.rectangle((0, 0, canvas.width, bar_height), fill=(18, 18, 18))
    draw.text((10, 8), text, fill=(235, 235, 235))
    return canvas


def compose_visualization(
    frame_rgb: np.ndarray,
    overlay_image: Image.Image,
    frame_index: int,
    timestamp_sec: float,
    geometry: dict[str, Any],
    visualization: str,
) -> np.ndarray:
    original = Image.fromarray(frame_rgb, mode="RGB")
    overlay = overlay_image.convert("RGB")
    text = format_overlay_text(frame_index, timestamp_sec, geometry)

    if visualization == "overlay":
        composed = annotate_image(overlay, text)
        return np.asarray(composed, dtype=np.uint8)

    left = annotate_image(original, text)
    right = annotate_image(overlay, text)
    canvas = Image.new("RGB", (left.width + right.width, left.height), color=(10, 10, 10))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return np.asarray(canvas, dtype=np.uint8)


def build_frame_row(frame_index: int, timestamp_sec: float, geometry: dict[str, Any]) -> dict[str, Any]:
    bbox = geometry.get("bbox")
    trusted_trend = geometry.get("trusted_trend", {})
    principal_direction = geometry.get("principal_direction", {})
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 6),
        "foreground_pixels": int(geometry.get("foreground_pixels", 0)),
        "runway_ratio": float(geometry.get("runway_ratio", 0.0)),
        "bbox_xmin": bbox["xmin"] if bbox else None,
        "bbox_ymin": bbox["ymin"] if bbox else None,
        "bbox_xmax": bbox["xmax"] if bbox else None,
        "bbox_ymax": bbox["ymax"] if bbox else None,
        "principal_angle_deg": principal_direction.get("angle_deg"),
        "support_points": int(trusted_trend.get("support_points", 0) or 0),
        "reliable_support_points": int(trusted_trend.get("reliable_support_points", 0) or 0),
        "auxiliary_support_points": int(trusted_trend.get("auxiliary_support_points", 0) or 0),
    }


def build_control_row(
    frame_index: int,
    timestamp_sec: float,
    control: dict[str, Any],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 6),
        "interface_valid": bool(control["interface_valid"]),
        "target_direction_assumption": control["target_direction_assumption"],
        "heading_error_deg": control["heading_error_deg"],
        "lateral_error_px": control["lateral_error_px"],
        "lateral_error_norm": control["lateral_error_norm"],
        "lateral_error_runway_half_width": control["lateral_error_runway_half_width"],
        "centerline_x_at_control_row": control["centerline_x_at_control_row"],
        "near_center_x": control["near_center_x"],
        "far_center_x": control["far_center_x"],
        "image_center_x": control["image_center_x"],
        "control_row_y": control["control_row_y"],
        "far_row_y": control["far_row_y"],
        "runway_width_px": control["runway_width_px"],
        "support_points": int(control["support_points"]),
        "reliable_support_points": int(control["reliable_support_points"]),
        "auxiliary_support_points": int(control["auxiliary_support_points"]),
        "runway_ratio": float(control["runway_ratio"]),
        "confidence": float(control["confidence"]),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_output_fps(capture_fps: float, requested_fps: float | None, frame_stride: int) -> float:
    if requested_fps is not None and requested_fps > 0:
        return requested_fps
    if capture_fps > 0:
        return max(capture_fps / max(frame_stride, 1), 1.0)
    return 20.0


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
        stabilized_trend[key] = blend_point(
            current_trend.get(key),
            previous_trend.get(key),
            near_alpha,
        )
    for key in ("ray_end", "projected_down_end"):
        stabilized_trend[key] = blend_point(
            current_trend.get(key),
            previous_trend.get(key),
            far_alpha,
        )

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


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    checkpoint_path = Path(args.checkpoint).resolve()
    video_path = Path(args.video).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = load_config_from_args(args, checkpoint)
    device = pick_device(args.device)
    threshold = float(args.threshold if args.threshold is not None else config["predict"]["threshold"])

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_video_path = Path(args.output_video).resolve() if args.output_video else default_output_video_path(
        config=config,
        video_path=video_path,
        output_dir_override=args.output_dir,
    )
    output_dir = ensure_dir(output_video_path.parent)
    progress_json_path = output_dir / f"{output_video_path.stem}_progress.json"
    control_csv_path = output_dir / f"{output_video_path.stem}_control.csv"
    frame_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    capture_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    processed_count = 0
    visited_count = 0
    writer: cv2.VideoWriter | None = None
    output_fps = resolve_output_fps(capture_fps, args.fps, args.frame_stride)
    start_time = time.time()
    previous_stabilized_geometry: dict[str, Any] | None = None

    try:
        with torch.no_grad():
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_index = args.start_frame + visited_count
                visited_count += 1

                if (visited_count - 1) % args.frame_stride != 0:
                    continue
                if args.max_frames is not None and processed_count >= args.max_frames:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess_frame(
                    frame_rgb=frame_rgb,
                    image_size=tuple(config["data"]["image_size"]),
                    mean=tuple(config["data"]["mean"]),
                    std=tuple(config["data"]["std"]),
                ).to(device)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits)
                pred_mask = (probs >= threshold).float()[0, 0].detach().cpu().numpy().astype(np.uint8)

                restored_mask = resize_binary_mask(pred_mask, (frame_rgb.shape[1], frame_rgb.shape[0]))
                geometry = extract_mask_geometry(restored_mask)
                if args.temporal_smoothing == "trend":
                    geometry = stabilize_video_geometry(geometry, previous_stabilized_geometry)
                previous_stabilized_geometry = copy.deepcopy(geometry)
                control = compute_visual_control_interface(
                    mask=restored_mask,
                    geometry=geometry,
                    frame_width=frame_rgb.shape[1],
                    frame_height=frame_rgb.shape[0],
                )
                overlay_image = draw_geometry_overlay(frame_rgb, restored_mask, geometry)

                source_fps = capture_fps if capture_fps > 0 else output_fps
                timestamp_sec = frame_index / source_fps
                visualization_frame = compose_visualization(
                    frame_rgb=frame_rgb,
                    overlay_image=overlay_image,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    geometry=geometry,
                    visualization=args.visualization,
                )

                if writer is None:
                    vis_height, vis_width = visualization_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_video_path),
                        cv2.VideoWriter_fourcc(*args.codec),
                        output_fps,
                        (vis_width, vis_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open VideoWriter for {output_video_path}. Try --codec avc1 or a different path."
                        )

                writer.write(cv2.cvtColor(visualization_frame, cv2.COLOR_RGB2BGR))
                frame_rows.append(build_frame_row(frame_index, timestamp_sec, geometry))
                control_rows.append(build_control_row(frame_index, timestamp_sec, control))
                processed_count += 1

                if args.progress_every > 0 and processed_count % args.progress_every == 0:
                    elapsed_sec = time.time() - start_time
                    fps_effective = processed_count / max(elapsed_sec, 1e-6)
                    remaining_frames = max(total_frames - (frame_index + 1), 0)
                    eta_sec = remaining_frames / max(fps_effective, 1e-6)
                    progress_payload = {
                        "input_video": str(video_path),
                        "output_video": str(output_video_path),
                        "processed_frames": processed_count,
                        "current_frame_index": frame_index,
                        "total_frames_in_video": total_frames,
                        "elapsed_sec": elapsed_sec,
                        "effective_fps": fps_effective,
                        "estimated_remaining_sec": eta_sec,
                        "visualization": args.visualization,
                    }
                    write_json(progress_payload, progress_json_path)
                    print(
                        f"[progress] processed={processed_count} current_frame={frame_index} "
                        f"elapsed={elapsed_sec:.1f}s fps={fps_effective:.2f} eta={eta_sec:.1f}s",
                        flush=True,
                    )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if processed_count == 0:
        raise RuntimeError("No frames were processed. Check --start-frame, --max-frames, and --frame-stride.")

    metrics_csv_path = output_dir / f"{output_video_path.stem}_frames.csv"
    summary_json_path = output_dir / f"{output_video_path.stem}_summary.json"
    write_csv(frame_rows, metrics_csv_path)
    write_csv(control_rows, control_csv_path)

    summary = {
        "input_video": str(video_path),
        "output_video": str(output_video_path),
        "checkpoint": str(checkpoint_path),
        "mode": checkpoint.get("mode", "loo"),
        "visualization": args.visualization,
        "threshold": threshold,
        "device": str(device),
        "total_frames_in_video": total_frames,
        "processed_frames": processed_count,
        "start_frame": args.start_frame,
        "frame_stride": args.frame_stride,
        "output_fps": output_fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "per_frame_csv": str(metrics_csv_path),
        "control_interface_csv": str(control_csv_path),
        "temporal_smoothing": args.temporal_smoothing,
    }
    write_json(summary, summary_json_path)

    print(summary)


if __name__ == "__main__":
    main()
