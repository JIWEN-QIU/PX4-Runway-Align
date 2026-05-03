from __future__ import annotations

from typing import Any

import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _sample_runway_span(
    mask: np.ndarray,
    target_y: int,
    search_radius: int = 16,
) -> dict[str, float] | None:
    height, _ = mask.shape
    for offset in range(search_radius + 1):
        for candidate_y in {target_y - offset, target_y + offset}:
            if not (0 <= candidate_y < height):
                continue
            xs = np.where(mask[candidate_y] > 0)[0]
            if xs.size == 0:
                continue
            left_x = float(xs[0])
            right_x = float(xs[-1])
            width_px = float(right_x - left_x + 1.0)
            return {
                "sample_y": float(candidate_y),
                "left_x": left_x,
                "right_x": right_x,
                "width_px": width_px,
                "center_x": float((left_x + right_x) * 0.5),
            }
    return None


def _compute_heading_error_deg(geometry: dict[str, Any]) -> float | None:
    trusted_trend = geometry.get("trusted_trend", {})
    near_point = trusted_trend.get("ray_anchor") or trusted_trend.get("observed_end")
    far_point = trusted_trend.get("ray_end") or trusted_trend.get("observed_start")
    if near_point is None or far_point is None:
        return None

    dx = float(far_point["x"] - near_point["x"])
    forward_dy = float(near_point["y"] - far_point["y"])
    if abs(forward_dy) < 1e-6 and abs(dx) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(dx, max(forward_dy, 1e-6))))


def _trend_x_at_y(trusted_trend: dict[str, Any], y: float, frame_width: int) -> float | None:
    slope = trusted_trend.get("slope_x_per_y")
    intercept = trusted_trend.get("intercept_x")
    if slope is None or intercept is None:
        return None
    return float(np.clip(float(slope) * float(y) + float(intercept), 0.0, max(frame_width - 1.0, 0.0)))


def _should_use_observed_near_span_for_lateral(
    near_span: dict[str, float] | None,
    trusted_trend: dict[str, Any],
    frame_width: int,
    prefer_observed_span_for_lateral: bool,
) -> tuple[bool, str]:
    if near_span is None:
        return False, "image_center_fallback"

    source = trusted_trend.get("source")
    if source not in {"single_side_boundary", "wide_single_side_centerline"}:
        return True, "observed_near_span"
    if not prefer_observed_span_for_lateral:
        return False, "trend_extrapolated_preference_off"

    border_margin_px = 4.0
    left_x = float(near_span["left_x"])
    right_x = float(near_span["right_x"])
    width_px = float(near_span["width_px"])
    touches_border = left_x <= border_margin_px or right_x >= float(frame_width - 1) - border_margin_px
    if touches_border:
        return False, "trend_extrapolated_border_clipped"

    estimated_width_px = trusted_trend.get("estimated_width_px")
    if estimated_width_px is None:
        return False, "trend_extrapolated_single_side"

    visible_width_ratio = width_px / max(float(estimated_width_px), 1.0)
    if visible_width_ratio < 0.75:
        return False, "trend_extrapolated_narrow_span"

    return True, "observed_near_span"


def _pick_far_row_y(
    geometry: dict[str, Any],
    control_row_y: float,
    frame_height: int,
) -> float:
    bbox = geometry.get("bbox") or {}
    bbox_ymin = float(bbox.get("ymin", 0.20 * frame_height))
    observed_start = geometry.get("trusted_trend", {}).get("observed_start")
    if observed_start is not None:
        bbox_ymin = min(bbox_ymin, float(observed_start["y"]))

    separation = max(140.0, 0.18 * frame_height)
    candidate = max(bbox_ymin + 30.0, control_row_y - separation)
    if candidate >= control_row_y:
        candidate = max(0.0, control_row_y - 120.0)
    return float(np.clip(candidate, 0.0, max(control_row_y - 1.0, 0.0)))


def _compute_local_heading_error_deg(
    near_span: dict[str, float] | None,
    far_span: dict[str, float] | None,
    geometry: dict[str, Any],
) -> float | None:
    if near_span is not None and far_span is not None:
        dx = float(far_span["center_x"] - near_span["center_x"])
        forward_dy = float(near_span["sample_y"] - far_span["sample_y"])
        if abs(forward_dy) < 1e-6 and abs(dx) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(dx, max(forward_dy, 1e-6))))
    return _compute_heading_error_deg(geometry)


def compute_visual_control_interface(
    mask: np.ndarray,
    geometry: dict[str, Any],
    frame_width: int,
    frame_height: int,
    control_row_ratio: float = 0.82,
    search_radius: int = 16,
    prefer_observed_span_for_lateral: bool = True,
) -> dict[str, Any]:
    image_center_x = float((frame_width - 1) * 0.5)
    target_row_y = int(round((frame_height - 1) * control_row_ratio))
    trusted_trend = geometry.get("trusted_trend", {})
    observed_end = trusted_trend.get("observed_end")
    if observed_end is not None:
        target_row_y = max(target_row_y, int(round(float(observed_end["y"]))))
    target_row_y = int(np.clip(target_row_y, 0, frame_height - 1))

    near_span = _sample_runway_span(mask, target_row_y, search_radius=search_radius)
    sample_row_y = float(near_span["sample_y"]) if near_span is not None else float(target_row_y)
    far_row_y = _pick_far_row_y(geometry, sample_row_y, frame_height)
    far_span = _sample_runway_span(mask, int(round(far_row_y)), search_radius=max(search_radius, 24))
    use_single_side_trend = trusted_trend.get("source") in {
        "single_side_boundary",
        "wide_single_side_centerline",
    }

    trend_centerline_x = _trend_x_at_y(trusted_trend, sample_row_y, frame_width)
    trend_far_center_x = _trend_x_at_y(trusted_trend, far_row_y, frame_width)
    use_observed_near_span, lateral_source = _should_use_observed_near_span_for_lateral(
        near_span=near_span,
        trusted_trend=trusted_trend,
        frame_width=frame_width,
        prefer_observed_span_for_lateral=prefer_observed_span_for_lateral,
    )
    centerline_x = near_span["center_x"] if use_observed_near_span and near_span is not None else image_center_x
    if use_single_side_trend and trend_centerline_x is not None and not use_observed_near_span:
        centerline_x = trend_centerline_x
    elif near_span is None:
        lateral_source = "image_center_fallback"
    heading_error_deg = (
        _compute_heading_error_deg(geometry)
        if use_single_side_trend
        else _compute_local_heading_error_deg(near_span, far_span, geometry)
    )

    lateral_error_px = None
    lateral_error_norm = None
    runway_width_px = None
    lateral_error_runway_half_width = None
    if centerline_x is not None:
        lateral_error_px = float(centerline_x - image_center_x)
        lateral_error_norm = float(lateral_error_px / max(frame_width * 0.5, 1.0))
    if near_span is not None:
        runway_width_px = float(near_span["width_px"])
    if use_single_side_trend and trusted_trend.get("estimated_width_px") is not None:
        runway_width_px = float(trusted_trend["estimated_width_px"])
    if runway_width_px is not None:
        if lateral_error_px is not None:
            lateral_error_runway_half_width = float(lateral_error_px / max(runway_width_px * 0.5, 1.0))

    support_points = int(trusted_trend.get("support_points", 0) or 0)
    reliable_support_points = int(trusted_trend.get("reliable_support_points", 0) or 0)
    auxiliary_support_points = int(trusted_trend.get("auxiliary_support_points", 0) or 0)
    runway_ratio = float(geometry.get("runway_ratio", 0.0))

    support_conf = _clip01((reliable_support_points - 1.0) / 7.0)
    width_conf = 0.0 if runway_width_px is None else _clip01(runway_width_px / max(frame_width * 0.22, 1.0))
    far_conf = 0.0 if far_span is None else 1.0
    ratio_conf = _clip01(runway_ratio / 0.10)
    confidence = 0.40 * support_conf + 0.25 * width_conf + 0.20 * ratio_conf + 0.15 * far_conf
    if not trusted_trend.get("available", False):
        confidence *= 0.25

    bbox = geometry.get("bbox") or {}
    bbox_ymax = float(bbox.get("ymax", -1.0))
    mask_bottom_ratio = 0.0 if bbox_ymax < 0.0 else float((bbox_ymax + 1.0) / max(frame_height, 1))
    near_field_present = near_span is not None and float(near_span["sample_y"]) >= float(frame_height * 0.60)
    minimum_width_px = max(24.0, frame_width * 0.08)
    sufficient_width = runway_width_px is not None and float(runway_width_px) >= minimum_width_px
    sufficient_ratio = runway_ratio >= 0.01
    sufficient_bottom_extent = mask_bottom_ratio >= 0.60

    gate_trusted_trend = bool(trusted_trend.get("available", False))
    gate_near_span = bool(near_span is not None)
    gate_heading = bool(heading_error_deg is not None)
    gate_near_field = bool(near_field_present)
    gate_width = bool(sufficient_width)
    gate_ratio = bool(sufficient_ratio)
    gate_bottom_extent = bool(sufficient_bottom_extent)

    interface_valid = bool(
        gate_trusted_trend
        and gate_near_span
        and gate_heading
        and gate_near_field
        and gate_width
        and gate_ratio
        and gate_bottom_extent
    )

    return {
        "interface_valid": interface_valid,
        "target_direction_assumption": "runway_centerline",
        "heading_error_deg": heading_error_deg,
        "lateral_error_px": lateral_error_px,
        "lateral_error_norm": lateral_error_norm,
        "lateral_error_runway_half_width": lateral_error_runway_half_width,
        "centerline_x_at_control_row": centerline_x,
        "near_center_x": trend_centerline_x if use_single_side_trend else (None if near_span is None else float(near_span["center_x"])),
        "far_center_x": trend_far_center_x if use_single_side_trend else (None if far_span is None else float(far_span["center_x"])),
        "image_center_x": image_center_x,
        "control_row_y": sample_row_y,
        "far_row_y": None if far_span is None else float(far_span["sample_y"]),
        "runway_width_px": runway_width_px,
        "support_points": support_points,
        "reliable_support_points": reliable_support_points,
        "auxiliary_support_points": auxiliary_support_points,
        "runway_ratio": runway_ratio,
        "mask_bottom_ratio": mask_bottom_ratio,
        "gate_trusted_trend": gate_trusted_trend,
        "gate_near_span": gate_near_span,
        "gate_heading": gate_heading,
        "gate_near_field": gate_near_field,
        "gate_width": gate_width,
        "gate_ratio": gate_ratio,
        "gate_bottom_extent": gate_bottom_extent,
        "geometry_source": trusted_trend.get("source"),
        "lateral_source": lateral_source,
        "single_side": trusted_trend.get("single_side"),
        "confidence": float(confidence),
    }
