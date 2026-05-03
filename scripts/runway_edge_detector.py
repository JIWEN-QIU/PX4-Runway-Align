#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


class RunwayEdgeDetector:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/front_camera/image_raw")
        self.roi_top_ratio = float(rospy.get_param("~roi_top_ratio", 0.0))
        self.roi_bottom_ratio = float(rospy.get_param("~roi_bottom_ratio", 0.5))
        self.center_exclusion_half_width_ratio = float(
            rospy.get_param("~center_exclusion_half_width_ratio", 0.12)
        )
        self.trapezoid_top_y_ratio = float(rospy.get_param("~trapezoid_top_y_ratio", 0.35))
        self.side_band_top_inner_ratio = float(rospy.get_param("~side_band_top_inner_ratio", 0.42))
        self.side_band_bottom_inner_ratio = float(rospy.get_param("~side_band_bottom_inner_ratio", 0.28))
        self.use_side_band_mask = bool(rospy.get_param("~use_side_band_mask", False))
        self.white_threshold = int(rospy.get_param("~white_threshold", 185))
        self.white_hsv_s_max = int(rospy.get_param("~white_hsv_s_max", 60))
        self.white_hsv_v_min = int(rospy.get_param("~white_hsv_v_min", 170))
        self.white_lab_l_min = int(rospy.get_param("~white_lab_l_min", 170))
        self.white_lab_ab_dev_max = int(rospy.get_param("~white_lab_ab_dev_max", 20))
        self.use_edge_map = bool(rospy.get_param("~use_edge_map", False))
        self.white_scan_min_pixels = int(rospy.get_param("~white_scan_min_pixels", 6))
        self.white_scan_row_step = int(rospy.get_param("~white_scan_row_step", 8))
        self.canny_low = int(rospy.get_param("~canny_low", 50))
        self.canny_high = int(rospy.get_param("~canny_high", 150))
        self.hough_threshold = int(rospy.get_param("~hough_threshold", 25))
        self.min_line_length = int(rospy.get_param("~min_line_length", 40))
        self.max_line_gap = int(rospy.get_param("~max_line_gap", 20))
        self.min_abs_slope = float(rospy.get_param("~min_abs_slope", 0.3))
        self.max_abs_slope = float(rospy.get_param("~max_abs_slope", 8.0))
        self.reference_y_ratio = float(rospy.get_param("~reference_y_ratio", 0.92))
        self.lookahead_y_ratio = float(rospy.get_param("~lookahead_y_ratio", 0.55))
        self.publish_debug_mask = bool(rospy.get_param("~publish_debug_mask", True))

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.overlay_pub = rospy.Publisher("~overlay_image", Image, queue_size=1)
        self.runway_visible_pub = rospy.Publisher("~runway_visible", Bool, queue_size=1)
        self.e_lat_pub = rospy.Publisher("~e_lat_img_px", Float32, queue_size=1)
        self.e_yaw_pub = rospy.Publisher("~e_yaw_img_rad", Float32, queue_size=1)
        self.confidence_pub = rospy.Publisher("~confidence_geom", Float32, queue_size=1)
        self.lookahead_pub = rospy.Publisher("~lookahead_point", PointStamped, queue_size=1)
        self.mask_pub = None
        if self.publish_debug_mask:
            self.mask_pub = rospy.Publisher("~binary_mask", Image, queue_size=1)

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo("runway_edge_detector listening on %s", self.image_topic)

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge conversion failed: %s", exc)
            return

        height, width = image.shape[:2]
        roi_top = max(0, min(height - 1, int(height * self.roi_top_ratio)))
        roi_bottom = max(roi_top + 1, min(height, int(height * self.roi_bottom_ratio)))
        roi = image[roi_top:roi_bottom, :]

        preprocessed_roi = self.preprocess_roi(roi)
        white_mask = self.extract_white_mask(preprocessed_roi)
        edge_map = self.extract_edge_map(preprocessed_roi)
        mask, roi_mask, left_segments, right_segments = self.estimate_runway_edges(
            white_mask, edge_map, roi.shape[1], roi.shape[0]
        )
        left_model = self.fit_edge_model(left_segments, "left")
        right_model = self.fit_edge_model(right_segments, "right")

        overlay = image.copy()
        cv2.rectangle(overlay, (0, roi_top), (width - 1, roi_bottom - 1), (120, 120, 120), 1)
        exclusion_half_width = int(width * self.center_exclusion_half_width_ratio)
        if exclusion_half_width > 0:
            exclusion_left = max(0, int(width * 0.5) - exclusion_half_width)
            exclusion_right = min(width - 1, int(width * 0.5) + exclusion_half_width)
            cv2.rectangle(
                overlay,
                (exclusion_left, roi_top),
                (exclusion_right, roi_bottom - 1),
                (60, 60, 180),
                1,
            )
        self.draw_roi_mask_outline(overlay, roi_mask, roi_top)

        self.draw_segments(overlay, left_segments, roi_top, (80, 220, 80))
        self.draw_segments(overlay, right_segments, roi_top, (80, 160, 255))

        guideline = self.build_guideline(width, left_model, right_model)
        result = self.compute_control_features(width, roi.shape[0], left_model, right_model, guideline)

        runway_visible = result is not None
        confidence = result["confidence"] if runway_visible else 0.0
        e_lat = result["e_lat_px"] if runway_visible else float("nan")
        e_yaw = result["e_yaw_rad"] if runway_visible else float("nan")

        if runway_visible:
            self.draw_edge_polyline(overlay, left_model, roi_top, (0, 255, 0))
            self.draw_edge_polyline(overlay, right_model, roi_top, (0, 191, 255))
            self.draw_guide(overlay, result, roi_top)
            self.publish_lookahead(msg, result["lookahead_point"], roi_top)
        else:
            self.publish_lookahead(msg, None, roi_top)
            cv2.putText(
                overlay,
                "runway not reliable",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        self.draw_metrics(overlay, runway_visible, confidence, e_lat, e_yaw)
        self.publish_scalars(runway_visible, e_lat, e_yaw, confidence)
        self.publish_images(msg, overlay, mask, roi_top)

    def preprocess_roi(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = self.clahe.apply(gray)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        return {
            "gray": gray,
            "hsv": hsv,
            "lab": lab,
        }

    def extract_white_mask(self, preprocessed_roi):
        gray = preprocessed_roi["gray"]
        hsv = preprocessed_roi["hsv"]
        lab = preprocessed_roi["lab"]

        _, gray_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)

        hsv_lower = np.array([0, 0, self.white_hsv_v_min], dtype=np.uint8)
        hsv_upper = np.array([180, self.white_hsv_s_max, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        lab_l = lab[:, :, 0]
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]
        lab_l_mask = cv2.inRange(lab_l, self.white_lab_l_min, 255)
        lab_a_delta = cv2.absdiff(lab_a, np.full_like(lab_a, 128))
        lab_b_delta = cv2.absdiff(lab_b, np.full_like(lab_b, 128))
        lab_a_mask = cv2.inRange(lab_a_delta, 0, self.white_lab_ab_dev_max)
        lab_b_mask = cv2.inRange(lab_b_delta, 0, self.white_lab_ab_dev_max)
        lab_mask = cv2.bitwise_and(lab_l_mask, cv2.bitwise_and(lab_a_mask, lab_b_mask))

        white_mask = cv2.bitwise_or(gray_mask, cv2.bitwise_and(hsv_mask, lab_mask))
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1
        )
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1
        )
        return white_mask

    def extract_edge_map(self, preprocessed_roi):
        return cv2.Canny(preprocessed_roi["gray"], self.canny_low, self.canny_high)

    def estimate_runway_edges(self, white_mask, edge_map, width, height):
        if self.use_edge_map:
            combined = cv2.bitwise_or(white_mask, edge_map)
        else:
            combined = white_mask.copy()

        left_segments = []
        right_segments = []
        center_x = width * 0.5
        exclusion_half_width = width * self.center_exclusion_half_width_ratio
        exclusion_left = center_x - exclusion_half_width
        exclusion_right = center_x + exclusion_half_width
        bottom_y = height - 1

        roi_mask = self.build_side_band_mask(width, height)
        combined = cv2.bitwise_and(combined, roi_mask)
        if exclusion_half_width > 0:
            combined[:, int(max(0, exclusion_left)):int(min(width, exclusion_right))] = 0

        left_points, right_points = self.scan_white_edge_points(white_mask, roi_mask)
        if len(left_points) >= 3:
            left_segments = self.points_to_segments(left_points)
        if len(right_points) >= 3:
            right_segments = self.points_to_segments(right_points)

        if left_segments and right_segments:
            return combined, roi_mask, left_segments, right_segments

        lines = cv2.HoughLinesP(
            combined,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return combined, roi_mask, left_segments, right_segments

        for line in lines:
            x1, y1, x2, y2 = [int(value) for value in line[0]]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < self.min_line_length or abs(dx) < 1e-3:
                continue

            slope = dy / dx
            abs_slope = abs(slope)
            if abs_slope < self.min_abs_slope or abs_slope > self.max_abs_slope:
                continue

            x_mid = 0.5 * (x1 + x2)
            if exclusion_half_width > 0 and exclusion_left <= x_mid <= exclusion_right:
                continue

            x_bottom = self.project_x_at_y((x1, y1, x2, y2), bottom_y)
            if x_bottom is None:
                continue

            segment = (x1, y1, x2, y2)
            if slope < 0.0 and x_bottom < center_x:
                left_segments.append(segment)
            elif slope > 0.0 and x_bottom > center_x:
                right_segments.append(segment)

        return combined, roi_mask, left_segments, right_segments

    def scan_white_edge_points(self, white_mask, roi_mask):
        height, width = white_mask.shape
        center_x = width * 0.5
        left_points = []
        right_points = []

        row_start = max(0, int(height * self.trapezoid_top_y_ratio))
        row_stop = height - 1
        row_step = max(1, self.white_scan_row_step)

        for y in range(row_start, row_stop, row_step):
            row_mask = roi_mask[y, :]
            row_white = white_mask[y, :]
            valid = np.where((row_mask > 0) & (row_white > 0))[0]
            if valid.size == 0:
                continue

            left_candidates = valid[valid < center_x]
            right_candidates = valid[valid > center_x]

            left_run = self.select_longest_contiguous_run(left_candidates)
            right_run = self.select_longest_contiguous_run(right_candidates)

            if left_run is not None and left_run[2] >= self.white_scan_min_pixels:
                left_points.append((float(left_run[1]), float(y)))

            if right_run is not None and right_run[2] >= self.white_scan_min_pixels:
                right_points.append((float(right_run[0]), float(y)))

        return left_points, right_points

    @staticmethod
    def select_longest_contiguous_run(indices):
        if indices.size == 0:
            return None

        best = None
        run_start = int(indices[0])
        prev = int(indices[0])
        best_length = 1

        for value in indices[1:]:
            value = int(value)
            if value == prev + 1:
                prev = value
                continue

            current_length = prev - run_start + 1
            if current_length > best_length:
                best = (run_start, prev, current_length)
                best_length = current_length

            run_start = value
            prev = value

        current_length = prev - run_start + 1
        if current_length > best_length or best is None:
            best = (run_start, prev, current_length)

        return best

    @staticmethod
    def points_to_segments(points):
        segments = []
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            segments.append((int(x1), int(y1), int(x2), int(y2)))
        return segments

    def build_side_band_mask(self, width, height):
        if not self.use_side_band_mask:
            return np.full((height, width), 255, dtype=np.uint8)

        mask = np.zeros((height, width), dtype=np.uint8)

        top_y = int(height * self.trapezoid_top_y_ratio)
        top_y = max(0, min(height - 1, top_y))
        left_top_inner = int(width * self.side_band_top_inner_ratio)
        left_bottom_inner = int(width * self.side_band_bottom_inner_ratio)
        right_top_inner = width - left_top_inner
        right_bottom_inner = width - left_bottom_inner

        left_polygon = np.array(
            [
                [0, height - 1],
                [0, top_y],
                [left_top_inner, top_y],
                [left_bottom_inner, height - 1],
            ],
            dtype=np.int32,
        )
        right_polygon = np.array(
            [
                [width - 1, height - 1],
                [width - 1, top_y],
                [right_top_inner, top_y],
                [right_bottom_inner, height - 1],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, [left_polygon, right_polygon], 255)
        return mask

    def fit_edge_model(self, segments, side):
        if len(segments) < 1:
            return None

        points = []
        for x1, y1, x2, y2 in segments:
            points.append((x1, y1))
            points.append((x2, y2))

        unique_points = np.unique(np.array(points, dtype=np.float32), axis=0)
        if unique_points.shape[0] < 2:
            return None

        sort_index = np.argsort(unique_points[:, 1])
        unique_points = unique_points[sort_index]

        filtered_points = self.filter_edge_points(unique_points, side)
        if filtered_points.shape[0] < 3:
            return None

        filtered_points = self.condense_edge_points(filtered_points, side)
        if filtered_points.shape[0] < 3:
            return None

        xs = filtered_points[:, 0]
        ys = filtered_points[:, 1]
        if np.max(ys) - np.min(ys) < 1.0:
            return None

        slope_xy, intercept_xy = np.polyfit(ys, xs, 1)
        if side == "left" and slope_xy > -0.01:
            return None
        if side == "right" and slope_xy < 0.01:
            return None

        return {
            "x_of_y_slope": float(slope_xy),
            "x_of_y_intercept": float(intercept_xy),
            "points": [tuple(point) for point in filtered_points.tolist()],
            "y_min": float(np.min(ys)),
            "y_max": float(np.max(ys)),
        }

    @staticmethod
    def filter_edge_points(points, side):
        if points.shape[0] < 3:
            return points

        filtered = [points[0]]
        tolerance = 10.0

        for point in points[1:]:
            prev_x = filtered[-1][0]
            current_x = point[0]
            if side == "left":
                if current_x <= prev_x + tolerance:
                    filtered.append(point)
            else:
                if current_x >= prev_x - tolerance:
                    filtered.append(point)

        filtered_points = np.array(filtered, dtype=np.float32)
        if filtered_points.shape[0] < 3:
            return points

        xs = filtered_points[:, 0]
        ys = filtered_points[:, 1]
        slope_xy, intercept_xy = np.polyfit(ys, xs, 1)
        predicted_xs = slope_xy * ys + intercept_xy
        residuals = np.abs(xs - predicted_xs)
        inlier_mask = residuals < 15.0
        if np.count_nonzero(inlier_mask) >= 3:
            filtered_points = filtered_points[inlier_mask]

        return filtered_points

    @staticmethod
    def condense_edge_points(points, side):
        if points.shape[0] < 2:
            return points

        condensed = []
        unique_ys = np.unique(points[:, 1])
        for y_value in unique_ys:
            same_row = points[np.isclose(points[:, 1], y_value)]
            if same_row.shape[0] == 0:
                continue
            if side == "left":
                x_value = np.max(same_row[:, 0])
            else:
                x_value = np.min(same_row[:, 0])
            condensed.append((float(x_value), float(y_value)))

        condensed_points = np.array(condensed, dtype=np.float32)
        if condensed_points.shape[0] < 2:
            return points

        sort_index = np.argsort(condensed_points[:, 1])
        return condensed_points[sort_index]

    def build_guideline(self, width, left_model, right_model):
        if left_model is None or right_model is None:
            return None

        left_points = sorted(left_model["points"], key=lambda point: point[1])
        right_points = sorted(right_model["points"], key=lambda point: point[1])
        if len(left_points) < 3 or len(right_points) < 3:
            return None

        overlap_y_min = max(left_points[0][1], right_points[0][1])
        overlap_y_max = min(left_points[-1][1], right_points[-1][1])
        if overlap_y_max - overlap_y_min < 24.0:
            return None

        sample_ys = np.linspace(overlap_y_min, overlap_y_max, 12)
        guide_midpoints = []
        widths = []

        for sample_y in sample_ys:
            left_x = self.interpolate_x_for_y(left_points, sample_y)
            right_x = self.interpolate_x_for_y(right_points, sample_y)
            if left_x is None or right_x is None:
                continue
            if right_x <= left_x:
                continue
            if left_x < 0 or right_x > (width - 1):
                continue

            widths.append(right_x - left_x)
            guide_midpoints.append((0.5 * (left_x + right_x), float(sample_y)))

        if len(guide_midpoints) < 3:
            return None

        return {
            "guide_midpoints": guide_midpoints,
            "widths": widths,
            "overlap_y_min": float(overlap_y_min),
            "overlap_y_max": float(overlap_y_max),
        }

    def compute_control_features(self, width, roi_height, left_model, right_model, guideline):
        if left_model is None or right_model is None or guideline is None:
            return None

        guide_midpoints = guideline["guide_midpoints"]
        widths = guideline["widths"]
        overlap_y_min = guideline["overlap_y_min"]
        overlap_y_max = guideline["overlap_y_max"]

        ref_y = overlap_y_min + 0.85 * (overlap_y_max - overlap_y_min)
        lookahead_y = overlap_y_min + 0.35 * (overlap_y_max - overlap_y_min)
        guide_x_ref = self.interpolate_x_for_y(guide_midpoints, ref_y)
        guide_x_lookahead = self.interpolate_x_for_y(guide_midpoints, lookahead_y)

        if guide_x_ref is None or guide_x_lookahead is None:
            return None

        e_lat_px = guide_x_ref - (width * 0.5)
        dy = float(ref_y - lookahead_y)
        dx = float(guide_x_lookahead - guide_x_ref)
        e_yaw_rad = math.atan2(dx, max(dy, 1.0))

        count_score = min(1.0, min(len(left_model["points"]), len(right_model["points"])) / 10.0)
        span_score = min(1.0, len(guide_midpoints) / 10.0)
        overlap_score = min(1.0, (overlap_y_max - overlap_y_min) / (0.35 * roi_height))
        width_mean = float(np.mean(widths))
        width_std = float(np.std(widths))
        width_stability = 0.0 if width_mean <= 1.0 else max(0.0, 1.0 - (width_std / width_mean) * 2.5)
        confidence = float(
            np.clip(0.25 * count_score + 0.20 * span_score + 0.25 * overlap_score + 0.30 * width_stability, 0.0, 1.0)
        )

        return {
            "guide_midpoints": guide_midpoints,
            "reference_point": (guide_x_ref, float(ref_y)),
            "lookahead_point": (guide_x_lookahead, float(lookahead_y)),
            "e_lat_px": float(e_lat_px),
            "e_yaw_rad": float(e_yaw_rad),
            "confidence": confidence,
        }

    @staticmethod
    def project_x_at_y(segment, target_y):
        x1, y1, x2, y2 = segment
        dy = float(y2 - y1)
        if abs(dy) < 1e-5:
            return None
        scale = (target_y - y1) / dy
        return x1 + scale * (x2 - x1)

    @staticmethod
    def model_x_at_y(model, sample_y):
        return model["x_of_y_slope"] * float(sample_y) + model["x_of_y_intercept"]

    @staticmethod
    def interpolate_x_for_y(points, sample_y):
        sorted_points = sorted(points, key=lambda point: point[1])
        ys = [point[1] for point in sorted_points]
        xs = [point[0] for point in sorted_points]
        if sample_y < ys[0] or sample_y > ys[-1]:
            return None
        return float(np.interp(sample_y, ys, xs))

    @staticmethod
    def draw_segments(image, segments, roi_top, color):
        for x1, y1, x2, y2 in segments:
            cv2.line(image, (x1, y1 + roi_top), (x2, y2 + roi_top), color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_roi_mask_outline(image, roi_mask, roi_top):
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            shifted = contour.copy()
            shifted[:, 0, 1] += roi_top
            cv2.polylines(image, [shifted], True, (180, 80, 80), 1, cv2.LINE_AA)

    def draw_edge_polyline(self, image, model, roi_top, color):
        if model is None or len(model["points"]) < 2:
            return
        points = sorted(model["points"], key=lambda point: point[1])
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            cv2.line(
                image,
                (int(x1), int(y1 + roi_top)),
                (int(x2), int(y2 + roi_top)),
                color,
                3,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_guide(image, result, roi_top):
        guide_points = result["guide_midpoints"]
        for index in range(len(guide_points) - 1):
            x1, y1 = guide_points[index]
            x2, y2 = guide_points[index + 1]
            cv2.line(
                image,
                (int(x1), int(y1 + roi_top)),
                (int(x2), int(y2 + roi_top)),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        ref_x, ref_y = result["reference_point"]
        lookahead_x, lookahead_y = result["lookahead_point"]
        cv2.circle(image, (int(ref_x), int(ref_y + roi_top)), 5, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (int(lookahead_x), int(lookahead_y + roi_top)), 6, (0, 0, 255), -1, cv2.LINE_AA)

    @staticmethod
    def draw_metrics(image, runway_visible, confidence, e_lat, e_yaw):
        metrics = [
            "visible: {}".format("yes" if runway_visible else "no"),
            "confidence: {:.2f}".format(confidence),
            "e_lat_px: {:.1f}".format(e_lat) if not math.isnan(e_lat) else "e_lat_px: nan",
            "e_yaw_rad: {:.3f}".format(e_yaw) if not math.isnan(e_yaw) else "e_yaw_rad: nan",
        ]

        for index, text in enumerate(metrics):
            cv2.putText(
                image,
                text,
                (20, 35 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def publish_scalars(self, runway_visible, e_lat, e_yaw, confidence):
        self.runway_visible_pub.publish(Bool(data=runway_visible))
        self.e_lat_pub.publish(Float32(data=float(e_lat)))
        self.e_yaw_pub.publish(Float32(data=float(e_yaw)))
        self.confidence_pub.publish(Float32(data=float(confidence)))

    def publish_lookahead(self, image_msg, lookahead_point, roi_top):
        point_msg = PointStamped()
        point_msg.header = image_msg.header
        if lookahead_point is not None:
            lookahead_x, lookahead_y = lookahead_point
            point_msg.point.x = float(lookahead_x)
            point_msg.point.y = float(lookahead_y + roi_top)
            point_msg.point.z = 0.0
        else:
            point_msg.point.x = float("nan")
            point_msg.point.y = float("nan")
            point_msg.point.z = float("nan")
        self.lookahead_pub.publish(point_msg)

    def publish_images(self, image_msg, overlay, mask, roi_top):
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = image_msg.header
            self.overlay_pub.publish(overlay_msg)

            if self.mask_pub is not None:
                mask_full = np.zeros((overlay.shape[0], overlay.shape[1]), dtype=np.uint8)
                mask_full[roi_top:roi_top + mask.shape[0], :] = mask
                mask_msg = self.bridge.cv2_to_imgmsg(mask_full, encoding="mono8")
                mask_msg.header = image_msg.header
                self.mask_pub.publish(mask_msg)
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge publish failed: %s", exc)


def main():
    rospy.init_node("runway_edge_detector")
    RunwayEdgeDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
