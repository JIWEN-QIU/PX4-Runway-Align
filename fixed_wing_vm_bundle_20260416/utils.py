from __future__ import annotations

import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = 1e-6

        if pos_weight is None:
            self.register_buffer("bce_pos_weight", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer(
                "bce_pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
                persistent=False,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pos_weight = self.bce_pos_weight if self.bce_pos_weight.numel() > 0 else None
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=dims) + targets.sum(dim=dims) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, {
            "loss": float(total_loss.detach().item()),
            "bce_loss": float(bce_loss.detach().item()),
            "dice_loss": float(dice_loss.detach().item()),
        }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    smooth = 1e-6
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)

    intersection = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    return image


def binary_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray(binary_mask_to_uint8(mask), mode="L")
    resized = mask_image.resize(size, Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
    return (np.asarray(resized) >= 128).astype(np.uint8)


def build_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: str = "#4ade80",
    alpha: float = 0.35,
) -> Image.Image:
    base = Image.fromarray(image, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgb = ImageColor.getrgb(color)

    mask_image = Image.fromarray(binary_mask_to_uint8(mask), mode="L")
    tint = Image.new("RGBA", base.size, rgb + (int(alpha * 255),))
    overlay = Image.composite(tint, overlay, mask_image)
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: int = 3,
) -> None:
    draw.line((float(start[0]), float(start[1]), float(end[0]), float(end[1])), fill=ImageColor.getrgb(color), width=width)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: int = 3,
    dash_length: int = 16,
    gap_length: int = 10,
) -> None:
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    total_length = math.hypot(x2 - x1, y2 - y1)
    if total_length < 1e-6:
        return

    direction_x = (x2 - x1) / total_length
    direction_y = (y2 - y1) / total_length
    step = dash_length + gap_length
    distance = 0.0

    while distance < total_length:
        dash_start = distance
        dash_end = min(distance + dash_length, total_length)
        start_point = (x1 + direction_x * dash_start, y1 + direction_y * dash_start)
        end_point = (x1 + direction_x * dash_end, y1 + direction_y * dash_end)
        draw_line(draw, start_point, end_point, color=color, width=width)
        distance += step


def draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: int,
    color: str,
    fill: bool = True,
) -> None:
    x, y = float(center[0]), float(center[1])
    bbox = (x - radius, y - radius, x + radius, y + radius)
    rgb = ImageColor.getrgb(color)
    draw.ellipse(bbox, outline=rgb, fill=rgb if fill else None, width=2)


def project_line_to_image_boundary(
    slope_x_per_y: float,
    intercept_x: float,
    start_y: float,
    image_shape: tuple[int, int],
    direction: str,
) -> dict[str, float]:
    height, width = image_shape
    y_min = 0.0
    y_max = float(height - 1)
    x_min = 0.0
    x_max = float(width - 1)

    candidates: list[tuple[float, float]] = []

    if direction == "down":
        bottom_x = slope_x_per_y * y_max + intercept_x
        if x_min <= bottom_x <= x_max:
            candidates.append((float(bottom_x), y_max))

        if abs(slope_x_per_y) > 1e-8:
            side_x = x_max if slope_x_per_y > 0 else x_min
            side_y = (side_x - intercept_x) / slope_x_per_y
            if start_y <= side_y <= y_max:
                candidates.append((float(side_x), float(side_y)))

        if candidates:
            end_x, end_y = min(candidates, key=lambda item: item[1])
            return {"x": end_x, "y": end_y}

        end_y = y_max
        end_x = float(np.clip(slope_x_per_y * end_y + intercept_x, x_min, x_max))
        return {"x": end_x, "y": end_y}

    top_x = slope_x_per_y * y_min + intercept_x
    if x_min <= top_x <= x_max:
        candidates.append((float(top_x), y_min))

    if abs(slope_x_per_y) > 1e-8:
        side_x = x_min if slope_x_per_y > 0 else x_max
        side_y = (side_x - intercept_x) / slope_x_per_y
        if y_min <= side_y <= start_y:
            candidates.append((float(side_x), float(side_y)))

    if candidates:
        end_x, end_y = max(candidates, key=lambda item: item[1])
        return {"x": end_x, "y": end_y}

    end_y = y_min
    end_x = float(np.clip(slope_x_per_y * end_y + intercept_x, x_min, x_max))
    return {"x": end_x, "y": end_y}


def draw_geometry_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    geometry: dict[str, Any],
    mask_color: str = "#22c55e",
    mask_alpha: float = 0.25,
) -> Image.Image:
    canvas = build_overlay(image, mask, color=mask_color, alpha=mask_alpha).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    bbox = geometry.get("bbox")
    if bbox:
        draw.rectangle(
            (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]),
            outline=ImageColor.getrgb("#fbbf24"),
            width=3,
        )

    centerline_analysis = geometry.get("centerline_analysis", {})
    reliable_points = centerline_analysis.get("reliable_points", [])
    rejected_points = centerline_analysis.get("rejected_points", [])

    for point in rejected_points:
        draw_circle(draw, (point["x"], point["y"]), radius=3, color="#94a3b8", fill=False)

    for point in reliable_points:
        draw_circle(draw, (point["x"], point["y"]), radius=4, color="#38bdf8")

    trusted_trend = geometry.get("trusted_trend", {})
    observed_start = trusted_trend.get("observed_start")
    observed_end = trusted_trend.get("observed_end")
    if observed_start and observed_end:
        draw_line(
            draw,
            (observed_start["x"], observed_start["y"]),
            (observed_end["x"], observed_end["y"]),
            color="#0ea5e9",
            width=3,
        )
    else:
        for point_a, point_b in zip(reliable_points[:-1], reliable_points[1:]):
            draw_line(draw, (point_a["x"], point_a["y"]), (point_b["x"], point_b["y"]), color="#0ea5e9", width=3)

    projected_down_start = trusted_trend.get("projected_down_start")
    projected_down_end = trusted_trend.get("projected_down_end")
    if projected_down_start and projected_down_end:
        draw_dashed_line(
            draw,
            (projected_down_start["x"], projected_down_start["y"]),
            (projected_down_end["x"], projected_down_end["y"]),
            color="#38bdf8",
            width=3,
            dash_length=18,
            gap_length=10,
        )

    ray_anchor = trusted_trend.get("ray_anchor")
    ray_end = trusted_trend.get("ray_end")
    trend_center = trusted_trend.get("center")
    if ray_anchor and ray_end:
        draw_line(draw, (ray_anchor["x"], ray_anchor["y"]), (ray_end["x"], ray_end["y"]), color="#ef4444", width=4)
    if trend_center:
        draw_circle(draw, (trend_center["x"], trend_center["y"]), radius=5, color="#ef4444")

    return canvas


def draw_mask_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = "",
) -> Image.Image:
    original = Image.fromarray(image, mode="RGB")
    gt_overlay = build_overlay(image, gt_mask, color="#60a5fa")
    pred_overlay = build_overlay(image, pred_mask, color="#f97316")

    panel = Image.new("RGB", (original.width * 3, original.height + 28), color=(20, 20, 20))
    panel.paste(original, (0, 28))
    panel.paste(gt_overlay, (original.width, 28))
    panel.paste(pred_overlay, (original.width * 2, 28))

    draw = ImageDraw.Draw(panel)
    draw.text((10, 6), f"{title} | original", fill=(255, 255, 255))
    draw.text((original.width + 10, 6), "ground truth", fill=(255, 255, 255))
    draw.text((original.width * 2 + 10, 6), "prediction", fill=(255, 255, 255))
    return panel


def save_sequence_gif(frames: list[Image.Image], path: str | Path, duration_ms: int) -> None:
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )


def save_image_grid(images: list[Image.Image], path: str | Path, columns: int = 3) -> None:
    if not images:
        return

    width = images[0].width
    height = images[0].height
    columns = max(1, columns)
    rows = math.ceil(len(images) / columns)

    grid = Image.new("RGB", (width * columns, height * rows), color=(16, 16, 16))
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        grid.paste(image, (col * width, row * height))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def extract_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=np.uint8)

    visited = np.zeros_like(mask, dtype=bool)
    best_component: list[tuple[int, int]] = []
    height, width = mask.shape

    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue

        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited[y, x] = True
        component: list[tuple[int, int]] = []

        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        if len(component) > len(best_component):
            best_component = component

    largest = np.zeros_like(mask, dtype=np.uint8)
    for y, x in best_component:
        largest[y, x] = 1
    return largest


def estimate_centerline(mask: np.ndarray, samples: int = 24) -> dict[str, Any]:
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return {"points": [], "slope_x_per_y": None}

    sample_rows = np.unique(np.linspace(ys.min(), ys.max(), num=min(samples, len(ys)), dtype=int))
    points: list[dict[str, float]] = []
    fit_x: list[float] = []
    fit_y: list[float] = []

    for y in sample_rows:
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue
        center_x = float((xs[0] + xs[-1]) / 2.0)
        points.append({"x": center_x, "y": float(y)})
        fit_x.append(center_x)
        fit_y.append(float(y))

    if len(points) < 2:
        return {"points": points, "slope_x_per_y": None}

    slope, intercept = np.polyfit(np.array(fit_y), np.array(fit_x), deg=1)
    return {
        "points": points,
        "slope_x_per_y": float(slope),
        "intercept_x": float(intercept),
    }


def analyze_centerline_observability(
    mask: np.ndarray,
    samples: int = 24,
    edge_margin_ratio: float = 0.03,
    fill_ratio_threshold: float = 0.92,
    span_ratio_threshold: float = 0.96,
    reliable_threshold: float = 0.55,
    max_resume_gap: int = 2,
    bootstrap_min_span_ratio: float = 0.02,
    bootstrap_min_span_px: int = 24,
    bootstrap_growth_factor: float = 3.0,
) -> dict[str, Any]:
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return {
            "samples": [],
            "reliable_points": [],
            "rejected_points": [],
            "reliable_count": 0,
        }

    sample_rows = np.unique(np.linspace(ys.min(), ys.max(), num=min(samples, len(ys)), dtype=int))
    image_width = mask.shape[1]
    edge_margin = max(2, int(round(image_width * edge_margin_ratio)))

    samples_info: list[dict[str, Any]] = []
    reliable_points: list[dict[str, float]] = []
    rejected_points: list[dict[str, float]] = []

    for y in sample_rows:
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue

        left = int(xs[0])
        right = int(xs[-1])
        span_width = int(right - left + 1)
        pixel_count = int(xs.size)
        center_x = float((left + right) / 2.0)
        fill_ratio = float(pixel_count / max(span_width, 1))
        span_ratio = float(span_width / image_width)
        touches_left = left <= edge_margin
        touches_right = right >= image_width - 1 - edge_margin

        confidence = 1.0
        if fill_ratio < fill_ratio_threshold:
            confidence *= max(0.2, fill_ratio / fill_ratio_threshold)
        if span_ratio > span_ratio_threshold:
            confidence *= 0.35
        if touches_left or touches_right:
            confidence *= 0.35
        if touches_left and touches_right:
            confidence *= 0.2

        normalized_y = float((y - ys.min()) / max(ys.max() - ys.min(), 1))
        support_weight = confidence * (1.15 - 0.35 * normalized_y)
        is_reliable = bool(confidence >= reliable_threshold)

        row_info = {
            "x": center_x,
            "y": float(y),
            "left_x": left,
            "right_x": right,
            "span_width": span_width,
            "pixel_count": pixel_count,
            "fill_ratio": fill_ratio,
            "span_ratio": span_ratio,
            "touches_left": touches_left,
            "touches_right": touches_right,
            "confidence": float(confidence),
            "support_weight": float(max(support_weight, 0.05)),
            "base_confidence": float(confidence),
            "base_support_weight": float(max(support_weight, 0.05)),
            "base_is_reliable": is_reliable,
            "is_reliable": is_reliable,
            "rejection_reason": "",
        }
        samples_info.append(row_info)

    accepted_rows: list[dict[str, Any]] = []
    gap_after_accept = 0
    overall_ys = [row["y"] for row in samples_info]
    lower_region_y = float(np.percentile(overall_ys, 70)) if overall_ys else 0.0
    bootstrap_span_threshold = max(bootstrap_min_span_px, int(round(image_width * bootstrap_min_span_ratio)))

    for idx, row in enumerate(samples_info):
        row["is_reliable"] = bool(row["base_is_reliable"])
        row["confidence"] = float(row["base_confidence"])
        row["support_weight"] = float(row["base_support_weight"])

        if not row["base_is_reliable"]:
            row["rejection_reason"] = "row_observability"
            if accepted_rows:
                gap_after_accept += 1
            continue

        if not accepted_rows:
            later_strong_rows = [
                later_row
                for later_row in samples_info[idx + 1 :]
                if later_row["base_is_reliable"]
                and later_row["span_width"] >= max(bootstrap_span_threshold, int(row["span_width"] * bootstrap_growth_factor))
            ]
            if row["span_width"] < bootstrap_span_threshold and later_strong_rows:
                row["is_reliable"] = False
                row["confidence"] = min(float(row["confidence"]), 0.20)
                row["support_weight"] = min(float(row["support_weight"]), 0.15)
                row["rejection_reason"] = "weak_anchor"
                continue

            accepted_rows.append(row)
            gap_after_accept = 0
            continue

        recent_rows = accepted_rows[-min(4, len(accepted_rows)) :]
        recent_xs = np.array([item["x"] for item in recent_rows], dtype=np.float64)
        recent_ys = np.array([item["y"] for item in recent_rows], dtype=np.float64)
        recent_spans = np.array([item["span_width"] for item in recent_rows], dtype=np.float64)
        expected_x = float(recent_xs[-1])
        if len(recent_rows) >= 2:
            local_slope, local_intercept = np.polyfit(recent_ys, recent_xs, deg=1)
            expected_x = float(local_slope * row["y"] + local_intercept)

        allowed_deviation = max(
            40.0,
            0.10 * max(float(row["span_width"]), float(recent_spans[-1])),
            0.035 * image_width,
        )
        if len(recent_rows) == 1:
            allowed_deviation = max(
                allowed_deviation,
                0.50 * float(row["span_width"]),
                0.12 * image_width,
            )
        x_deviation = abs(float(row["x"]) - expected_x)
        recent_span_median = float(np.median(recent_spans))
        span_ratio_to_recent = float(row["span_width"] / max(recent_span_median, 1.0))

        reject_reason = ""
        if gap_after_accept > max_resume_gap:
            reject_reason = "continuity_gap"
        elif x_deviation > allowed_deviation:
            reject_reason = "trend_deviation"
        elif row["y"] >= lower_region_y and span_ratio_to_recent < 0.35:
            reject_reason = "narrow_fragment"

        if reject_reason:
            row["is_reliable"] = False
            row["confidence"] = min(float(row["confidence"]), 0.25)
            row["support_weight"] = min(float(row["support_weight"]), 0.20)
            row["rejection_reason"] = reject_reason
            gap_after_accept += 1
            continue

        row["rejection_reason"] = ""
        accepted_rows.append(row)
        gap_after_accept = 0

    for row in samples_info:
        point = {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "confidence": float(row["confidence"]),
            "reason": row["rejection_reason"],
        }
        if row["is_reliable"]:
            reliable_points.append(point)
        else:
            rejected_points.append(point)

    return {
        "samples": samples_info,
        "reliable_points": reliable_points,
        "rejected_points": rejected_points,
        "reliable_count": len(reliable_points),
    }


def estimate_trusted_centerline_trend(
    analysis: dict[str, Any],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    rows = [row for row in analysis.get("samples", []) if row.get("is_reliable")]
    if len(rows) < 2:
        return {
            "available": False,
            "support_points": len(rows),
            "reliable_support_points": len(rows),
            "auxiliary_support_points": 0,
            "slope_x_per_y": None,
            "intercept_x": None,
            "start": None,
            "end": None,
            "observed_start": None,
            "observed_end": None,
            "auxiliary_points": [],
            "center": None,
            "ray_anchor": None,
            "ray_end": None,
            "projected_down_start": None,
            "projected_down_end": None,
        }

    ys = np.array([row["y"] for row in rows], dtype=np.float64)
    xs = np.array([row["x"] for row in rows], dtype=np.float64)
    weights = np.array([row["support_weight"] for row in rows], dtype=np.float64)
    slope, intercept = np.polyfit(ys, xs, deg=1, w=weights)

    recent_rows = rows[-min(4, len(rows)) :]
    recent_span_ys = np.array([row["y"] for row in recent_rows], dtype=np.float64)
    recent_spans = np.array([row["span_width"] for row in recent_rows], dtype=np.float64)
    if len(recent_rows) >= 2:
        span_slope, span_intercept = np.polyfit(recent_span_ys, recent_spans, deg=1)
    else:
        span_slope = 0.0
        span_intercept = float(recent_spans[-1])

    all_sample_rows = analysis.get("samples", [])
    reliable_max_y = float(ys.max())
    auxiliary_points: list[dict[str, float]] = []
    fit_xs = xs.tolist()
    fit_ys = ys.tolist()
    fit_weights = weights.tolist()

    for row in all_sample_rows:
        if row.get("is_reliable") or row.get("y", 0.0) <= reliable_max_y:
            continue
        if row.get("rejection_reason") != "row_observability":
            continue
        if not (row.get("touches_left") ^ row.get("touches_right")):
            continue
        if row.get("fill_ratio", 0.0) < 0.96:
            continue

        estimated_span = float(span_slope * row["y"] + span_intercept)
        estimated_span = float(
            np.clip(
                estimated_span,
                max(120.0, 0.8 * float(np.median(recent_spans))),
                0.95 * image_shape[1],
            )
        )
        if row.get("touches_right"):
            inferred_x = float(row["left_x"] + 0.5 * estimated_span)
        else:
            inferred_x = float(row["right_x"] - 0.5 * estimated_span)

        current_slope, current_intercept = np.polyfit(
            np.array(fit_ys, dtype=np.float64),
            np.array(fit_xs, dtype=np.float64),
            deg=1,
            w=np.array(fit_weights, dtype=np.float64),
        )
        expected_x = float(current_slope * row["y"] + current_intercept)
        allowed_deviation = max(70.0, 0.12 * estimated_span, 0.035 * image_shape[1])
        if abs(inferred_x - expected_x) > allowed_deviation:
            continue

        auxiliary_points.append({"x": inferred_x, "y": float(row["y"])})
        fit_xs.append(inferred_x)
        fit_ys.append(float(row["y"]))
        fit_weights.append(0.28)

    if auxiliary_points:
        ys = np.array(fit_ys, dtype=np.float64)
        xs = np.array(fit_xs, dtype=np.float64)
        weights = np.array(fit_weights, dtype=np.float64)
        slope, intercept = np.polyfit(ys, xs, deg=1, w=weights)

    height, width = image_shape
    start_y = float(ys.min())
    end_y = float(height - 1)
    start_x = float(np.clip(slope * start_y + intercept, 0.0, width - 1.0))
    end_x = float(np.clip(slope * end_y + intercept, 0.0, width - 1.0))
    observed_end_y = float(ys.max())
    observed_end_x = float(np.clip(slope * observed_end_y + intercept, 0.0, width - 1.0))
    center_y = float(np.average(ys, weights=weights))
    center_x = float(np.clip(slope * center_y + intercept, 0.0, width - 1.0))
    near_y = observed_end_y
    anchor_y = float(np.percentile(ys, 65))
    anchor_x = float(np.clip(slope * anchor_y + intercept, 0.0, width - 1.0))
    upward_boundary = project_line_to_image_boundary(slope, intercept, start_y, image_shape, direction="up")
    far_extension_y = float(max(upward_boundary["y"], start_y - 0.25 * max(near_y - start_y, 1.0)))
    far_extension_x = float(np.clip(slope * far_extension_y + intercept, 0.0, width - 1.0))
    lowest_sample_y = float(max((row["y"] for row in all_sample_rows), default=near_y))
    projected_down_start = None
    projected_down_end = None
    if lowest_sample_y > near_y:
        projected_down_start = {
            "x": float(np.clip(slope * near_y + intercept, 0.0, width - 1.0)),
            "y": near_y,
        }
        boundary_hit = project_line_to_image_boundary(slope, intercept, near_y, image_shape, direction="down")
        projected_down_end = boundary_hit if boundary_hit["y"] <= lowest_sample_y else {
            "x": float(slope * lowest_sample_y + intercept),
            "y": lowest_sample_y,
        }

    return {
        "available": True,
        "support_points": len(rows) + len(auxiliary_points),
        "reliable_support_points": len(rows),
        "auxiliary_support_points": len(auxiliary_points),
        "slope_x_per_y": float(slope),
        "intercept_x": float(intercept),
        "start": {"x": start_x, "y": start_y},
        "end": {"x": end_x, "y": end_y},
        "observed_start": {"x": start_x, "y": start_y},
        "observed_end": {"x": observed_end_x, "y": observed_end_y},
        "auxiliary_points": auxiliary_points,
        "center": {"x": center_x, "y": center_y},
        "ray_anchor": {"x": anchor_x, "y": anchor_y},
        "ray_end": {"x": far_extension_x, "y": far_extension_y},
        "projected_down_start": projected_down_start,
        "projected_down_end": projected_down_end,
        "source": "mask_centerline",
    }


def _empty_single_side_centerline_trend() -> dict[str, Any]:
    return {
        "available": False,
        "support_points": 0,
        "reliable_support_points": 0,
        "auxiliary_support_points": 0,
        "slope_x_per_y": None,
        "intercept_x": None,
        "start": None,
        "end": None,
        "observed_start": None,
        "observed_end": None,
        "auxiliary_points": [],
        "center": None,
        "ray_anchor": None,
        "ray_end": None,
        "projected_down_start": None,
        "projected_down_end": None,
        "source": "single_side_boundary",
        "single_side": None,
        "estimated_width_px": None,
    }


def estimate_single_side_boundary_centerline_trend(
    analysis: dict[str, Any],
    image_shape: tuple[int, int],
    min_support_points: int = 4,
    min_single_side_ratio: float = 0.72,
    max_two_side_rows: int = 0,
    max_unclipped_ratio: float = 0.20,
    offset_ratio: float = 0.50,
) -> dict[str, Any]:
    """Fallback centerline estimate for frames where only one runway side is visible.

    This is intentionally conservative: it only activates when nearly every sampled
    mask row is clipped by the same image side, which distinguishes "only one side
    ever visible" from the normal "both sides visible, then one side clips
    near-field" case handled by estimate_trusted_centerline_trend().
    """
    rows = [row for row in analysis.get("samples", []) if row.get("span_width", 0) > 0]
    if len(rows) < min_support_points:
        return _empty_single_side_centerline_trend()

    left_clipped = [row for row in rows if row.get("touches_left") and not row.get("touches_right")]
    right_clipped = [row for row in rows if row.get("touches_right") and not row.get("touches_left")]
    two_side_rows = [row for row in rows if row.get("touches_left") and row.get("touches_right")]
    unclipped_rows = [row for row in rows if not row.get("touches_left") and not row.get("touches_right")]

    if len(two_side_rows) > max_two_side_rows:
        return _empty_single_side_centerline_trend()
    if len(unclipped_rows) / max(len(rows), 1) > max_unclipped_ratio:
        return _empty_single_side_centerline_trend()

    if len(left_clipped) >= len(right_clipped):
        clipped_side = "left"
        side_rows = left_clipped
        visible_boundary_key = "right_x"
        center_sign = -1.0
    else:
        clipped_side = "right"
        side_rows = right_clipped
        visible_boundary_key = "left_x"
        center_sign = 1.0

    if len(side_rows) < min_support_points:
        return _empty_single_side_centerline_trend()
    if len(side_rows) / max(len(rows), 1) < min_single_side_ratio:
        return _empty_single_side_centerline_trend()

    height, width = image_shape
    side_rows = sorted(side_rows, key=lambda row: float(row["y"]))
    ys = np.array([row["y"] for row in side_rows], dtype=np.float64)
    boundary_xs = np.array([row[visible_boundary_key] for row in side_rows], dtype=np.float64)
    spans = np.array([row["span_width"] for row in side_rows], dtype=np.float64)

    if len(side_rows) >= 2:
        boundary_slope, boundary_intercept = np.polyfit(ys, boundary_xs, deg=1)
        span_slope, span_intercept = np.polyfit(ys, spans, deg=1)
    else:
        boundary_slope = 0.0
        boundary_intercept = float(boundary_xs[-1])
        span_slope = 0.0
        span_intercept = float(spans[-1])

    estimated_widths = np.clip(
        span_slope * ys + span_intercept,
        max(24.0, 0.04 * width),
        0.95 * width,
    )
    center_xs = boundary_xs + center_sign * offset_ratio * estimated_widths
    weights = np.linspace(0.75, 1.25, num=len(side_rows), dtype=np.float64)
    slope, intercept = np.polyfit(ys, center_xs, deg=1, w=weights)

    start_y = float(ys.min())
    observed_end_y = float(ys.max())
    end_y = float(height - 1)
    start_x = float(np.clip(slope * start_y + intercept, 0.0, width - 1.0))
    observed_end_x = float(np.clip(slope * observed_end_y + intercept, 0.0, width - 1.0))
    end_x = float(np.clip(slope * end_y + intercept, 0.0, width - 1.0))
    center_y = float(np.average(ys, weights=weights))
    center_x = float(np.clip(slope * center_y + intercept, 0.0, width - 1.0))
    anchor_y = float(np.percentile(ys, 65))
    anchor_x = float(np.clip(slope * anchor_y + intercept, 0.0, width - 1.0))
    upward_boundary = project_line_to_image_boundary(slope, intercept, start_y, image_shape, direction="up")
    far_extension_y = float(max(upward_boundary["y"], start_y - 0.25 * max(observed_end_y - start_y, 1.0)))
    far_extension_x = float(np.clip(slope * far_extension_y + intercept, 0.0, width - 1.0))

    projected_down_start = None
    projected_down_end = None
    lowest_sample_y = float(max((row["y"] for row in rows), default=observed_end_y))
    if lowest_sample_y > observed_end_y:
        projected_down_start = {"x": observed_end_x, "y": observed_end_y}
        boundary_hit = project_line_to_image_boundary(slope, intercept, observed_end_y, image_shape, direction="down")
        projected_down_end = boundary_hit if boundary_hit["y"] <= lowest_sample_y else {
            "x": float(slope * lowest_sample_y + intercept),
            "y": lowest_sample_y,
        }

    auxiliary_points = [
        {
            "x": float(np.clip(slope * row["y"] + intercept, 0.0, width - 1.0)),
            "y": float(row["y"]),
        }
        for row in side_rows
    ]

    return {
        "available": True,
        "support_points": len(side_rows),
        "reliable_support_points": 0,
        "auxiliary_support_points": len(side_rows),
        "slope_x_per_y": float(slope),
        "intercept_x": float(intercept),
        "start": {"x": start_x, "y": start_y},
        "end": {"x": end_x, "y": end_y},
        "observed_start": {"x": start_x, "y": start_y},
        "observed_end": {"x": observed_end_x, "y": observed_end_y},
        "auxiliary_points": auxiliary_points,
        "center": {"x": center_x, "y": center_y},
        "ray_anchor": {"x": anchor_x, "y": anchor_y},
        "ray_end": {"x": far_extension_x, "y": far_extension_y},
        "projected_down_start": projected_down_start,
        "projected_down_end": projected_down_end,
        "source": "single_side_boundary",
        "single_side": clipped_side,
        "estimated_width_px": float(np.median(estimated_widths)),
    }


def estimate_wide_single_side_centerline_trend(
    analysis: dict[str, Any],
    image_shape: tuple[int, int],
    min_support_points: int = 5,
    min_bottom_y_ratio: float = 0.75,
    min_median_span_ratio: float = 0.45,
) -> dict[str, Any]:
    """Fallback for one-visible-edge masks that are clipped by the bottom of frame.

    In this case row spans are often too wide/clipped to be considered reliable by
    the normal observability gate. Only sample rows with at least one visible lateral
    boundary are used for fitting; rows clipped on both sides are treated as near-field
    filler and are not allowed to pull the centerline.
    """
    rows = [row for row in analysis.get("samples", []) if row.get("span_width", 0) > 0]
    if len(rows) < min_support_points:
        return _empty_single_side_centerline_trend()

    height, width = image_shape
    rows = sorted(rows, key=lambda row: float(row["y"]))
    all_ys = np.array([row["y"] for row in rows], dtype=np.float64)
    all_xs = np.array([row["x"] for row in rows], dtype=np.float64)
    all_spans = np.array([row["span_width"] for row in rows], dtype=np.float64)

    if float(all_ys.max()) < float(height - 1) * min_bottom_y_ratio:
        return _empty_single_side_centerline_trend()
    if float(np.median(all_spans)) < float(width) * min_median_span_ratio:
        return _empty_single_side_centerline_trend()
    if float(np.ptp(all_xs)) < max(8.0, 0.015 * width):
        return _empty_single_side_centerline_trend()

    fit_rows = [
        row
        for row in rows
        if not (bool(row.get("touches_left")) and bool(row.get("touches_right")))
    ]
    if len(fit_rows) < min_support_points:
        return _empty_single_side_centerline_trend()

    rows = fit_rows
    ys = np.array([row["y"] for row in rows], dtype=np.float64)
    xs = np.array([row["x"] for row in rows], dtype=np.float64)
    spans = np.array([row["span_width"] for row in rows], dtype=np.float64)
    if float(np.ptp(xs)) < max(8.0, 0.015 * width):
        return _empty_single_side_centerline_trend()

    weights = np.linspace(0.75, 1.25, num=len(rows), dtype=np.float64)
    slope, intercept = np.polyfit(ys, xs, deg=1, w=weights)
    fitted_xs = slope * ys + intercept
    residuals = np.abs(xs - fitted_xs)

    inlier_threshold = max(90.0, 0.16 * width)
    inlier_mask = residuals <= inlier_threshold
    if int(inlier_mask.sum()) >= min_support_points and float(inlier_mask.mean()) >= 0.45:
        ys = ys[inlier_mask]
        xs = xs[inlier_mask]
        spans = spans[inlier_mask]
        rows = [row for row, keep in zip(rows, inlier_mask.tolist()) if keep]
        weights = np.linspace(0.75, 1.25, num=len(rows), dtype=np.float64)
        slope, intercept = np.polyfit(ys, xs, deg=1, w=weights)
        fitted_xs = slope * ys + intercept
        residuals = np.abs(xs - fitted_xs)

    if float(np.median(residuals)) > max(80.0, 0.13 * width):
        return _empty_single_side_centerline_trend()
    if float(np.percentile(residuals, 80)) > max(140.0, 0.22 * width):
        return _empty_single_side_centerline_trend()

    start_y = float(ys.min())
    observed_end_y = float(ys.max())
    end_y = float(height - 1)
    start_x = float(np.clip(slope * start_y + intercept, 0.0, width - 1.0))
    observed_end_x = float(np.clip(slope * observed_end_y + intercept, 0.0, width - 1.0))
    end_x = float(np.clip(slope * end_y + intercept, 0.0, width - 1.0))
    center_y = float(np.average(ys, weights=weights))
    center_x = float(np.clip(slope * center_y + intercept, 0.0, width - 1.0))
    anchor_y = float(np.percentile(ys, 65))
    anchor_x = float(np.clip(slope * anchor_y + intercept, 0.0, width - 1.0))
    upward_boundary = project_line_to_image_boundary(slope, intercept, start_y, image_shape, direction="up")
    far_extension_y = float(max(upward_boundary["y"], start_y - 0.25 * max(observed_end_y - start_y, 1.0)))
    far_extension_x = float(np.clip(slope * far_extension_y + intercept, 0.0, width - 1.0))

    projected_down_start = None
    projected_down_end = None
    if end_y > observed_end_y:
        projected_down_start = {"x": observed_end_x, "y": observed_end_y}
        projected_down_end = project_line_to_image_boundary(slope, intercept, observed_end_y, image_shape, direction="down")

    auxiliary_points = [
        {
            "x": float(np.clip(slope * row["y"] + intercept, 0.0, width - 1.0)),
            "y": float(row["y"]),
        }
        for row in rows
    ]

    return {
        "available": True,
        "support_points": len(rows),
        "reliable_support_points": 0,
        "auxiliary_support_points": len(rows),
        "slope_x_per_y": float(slope),
        "intercept_x": float(intercept),
        "start": {"x": start_x, "y": start_y},
        "end": {"x": end_x, "y": end_y},
        "observed_start": {"x": start_x, "y": start_y},
        "observed_end": {"x": observed_end_x, "y": observed_end_y},
        "auxiliary_points": auxiliary_points,
        "center": {"x": center_x, "y": center_y},
        "ray_anchor": {"x": anchor_x, "y": anchor_y},
        "ray_end": {"x": far_extension_x, "y": far_extension_y},
        "projected_down_start": projected_down_start,
        "projected_down_end": projected_down_end,
        "source": "wide_single_side_centerline",
        "single_side": "visible_edge",
        "estimated_width_px": float(np.median(all_spans)),
        "fit_region": "visible_boundary_rows",
    }


def has_sufficient_two_side_boundary_support(
    analysis: dict[str, Any],
    min_rows: int = 4,
    max_span_ratio: float = 0.92,
) -> bool:
    rows = [row for row in analysis.get("samples", []) if row.get("span_width", 0) > 0]
    two_side_rows = [
        row
        for row in rows
        if not bool(row.get("touches_left"))
        and not bool(row.get("touches_right"))
        and float(row.get("span_ratio", 1.0)) <= max_span_ratio
        and float(row.get("fill_ratio", 0.0)) >= 0.90
    ]
    reliable_two_side_rows = [row for row in two_side_rows if row.get("is_reliable")]
    return len(two_side_rows) >= min_rows or len(reliable_two_side_rows) >= max(2, min_rows - 1)


def estimate_principal_direction(mask: np.ndarray) -> dict[str, Any]:
    coords = np.argwhere(mask > 0)
    if coords.shape[0] < 2:
        return {"angle_deg": None, "direction_vector": None}

    centered = coords.astype(np.float64) - coords.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    dy = float(principal[0])
    dx = float(principal[1])
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    norm = math.sqrt(dx * dx + dy * dy) + 1e-8

    return {
        "angle_deg": angle_deg,
        "direction_vector": [dx / norm, dy / norm],
        "centroid_x": float(coords[:, 1].mean()),
        "centroid_y": float(coords[:, 0].mean()),
    }


def extract_mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    component = extract_largest_connected_component(mask)
    ys, xs = np.where(component > 0)
    centerline_analysis = analyze_centerline_observability(component)
    trusted_trend = estimate_trusted_centerline_trend(centerline_analysis, component.shape)
    single_side_trend = estimate_single_side_boundary_centerline_trend(centerline_analysis, component.shape)
    wide_single_side_trend = estimate_wide_single_side_centerline_trend(centerline_analysis, component.shape)
    has_two_side_support = has_sufficient_two_side_boundary_support(centerline_analysis)
    if single_side_trend.get("available", False):
        trusted_trend = single_side_trend
    elif wide_single_side_trend.get("available", False) and not has_two_side_support:
        trusted_trend = wide_single_side_trend

    if xs.size == 0:
        return {
            "foreground_pixels": 0,
            "runway_ratio": 0.0,
            "bbox": None,
            "principal_direction": estimate_principal_direction(component),
            "centerline": estimate_centerline(component),
            "centerline_analysis": centerline_analysis,
            "trusted_trend": trusted_trend,
        }

    bbox = {
        "xmin": int(xs.min()),
        "ymin": int(ys.min()),
        "xmax": int(xs.max()),
        "ymax": int(ys.max()),
    }

    return {
        "foreground_pixels": int(component.sum()),
        "runway_ratio": float(component.mean()),
        "bbox": bbox,
        "principal_direction": estimate_principal_direction(component),
        "centerline": estimate_centerline(component),
        "centerline_analysis": centerline_analysis,
        "trusted_trend": trusted_trend,
    }
