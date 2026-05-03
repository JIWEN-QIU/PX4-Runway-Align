#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rospy
from std_msgs.msg import Bool, Float32, String


def get_param_with_alias(name: str, legacy_name: str | None, default):
    if rospy.has_param(name):
        return rospy.get_param(name)
    if legacy_name and rospy.has_param(legacy_name):
        return rospy.get_param(legacy_name)
    return default


class RunwayGroundAlignController:
    def __init__(self):
        self.dry_run = bool(rospy.get_param("~dry_run", True))
        self.heading_sign = float(rospy.get_param("~heading_sign", 1.0))
        self.lateral_sign = float(rospy.get_param("~lateral_sign", 1.0))
        self.k_heading = float(rospy.get_param("~k_heading", 0.03))
        self.k_lateral = float(rospy.get_param("~k_lateral", 0.5))
        self.k_lateral_i = float(rospy.get_param("~k_lateral_i", 0.08))
        self.max_heading_error_deg = float(rospy.get_param("~max_heading_error_deg", 10.0))
        self.max_lateral_error = float(rospy.get_param("~max_lateral_error", 1.0))
        self.lateral_integral_limit = float(rospy.get_param("~lateral_integral_limit", 0.35))
        self.lateral_integral_leak = float(rospy.get_param("~lateral_integral_leak", 0.92))
        self.integral_heading_gate_deg = float(rospy.get_param("~integral_heading_gate_deg", 12.0))
        self.integral_zone_lateral_error = float(rospy.get_param("~integral_zone_lateral_error", 0.35))
        self.conf_min = float(rospy.get_param("~conf_min", 0.45))
        self.low_conf_scale = float(rospy.get_param("~low_conf_scale", 0.5))
        self.max_steer = float(rospy.get_param("~max_steer", 0.25))
        self.filter_alpha = float(rospy.get_param("~filter_alpha", 0.8))
        self.max_delta = float(rospy.get_param("~max_delta", 0.03))
        self.heading_deadband_deg = float(rospy.get_param("~heading_deadband_deg", 0.0))
        self.lateral_deadband = float(rospy.get_param("~lateral_deadband", 0.0))
        self.lateral_target_offset = float(rospy.get_param("~lateral_target_offset", 0.0))
        self.command_scale = float(rospy.get_param("~command_scale", 1.0))
        self.min_effective_steer = float(rospy.get_param("~min_effective_steer", 0.0))
        self.center_hold_heading_deg = float(rospy.get_param("~center_hold_heading_deg", 4.0))
        self.center_hold_lateral_error = float(rospy.get_param("~center_hold_lateral_error", 0.10))
        self.near_center_command_scale = float(rospy.get_param("~near_center_command_scale", 0.45))
        self.min_effective_heading_gate_deg = float(rospy.get_param("~min_effective_heading_gate_deg", 8.0))
        self.min_effective_lateral_gate = float(rospy.get_param("~min_effective_lateral_gate", 0.20))
        self.center_release_heading_deg = float(rospy.get_param("~center_release_heading_deg", 7.0))
        self.center_release_lateral_error = float(rospy.get_param("~center_release_lateral_error", 0.22))
        self.center_hold_zero_cmd = bool(rospy.get_param("~center_hold_zero_cmd", True))
        self.straight_hold_heading_deg = float(rospy.get_param("~straight_hold_heading_deg", 2.0))
        self.straight_hold_lateral_error = float(rospy.get_param("~straight_hold_lateral_error", 0.04))
        self.straight_release_heading_deg = float(rospy.get_param("~straight_release_heading_deg", 3.5))
        self.straight_release_lateral_error = float(rospy.get_param("~straight_release_lateral_error", 0.08))
        self.straight_hold_require_startup_boost_inactive = bool(
            rospy.get_param("~straight_hold_require_startup_boost_inactive", True)
        )
        self.straight_hold_require_center_hold = bool(
            rospy.get_param("~straight_hold_require_center_hold", True)
        )
        self.straight_hold_lateral_priority = bool(
            rospy.get_param("~straight_hold_lateral_priority", True)
        )
        self.straight_hold_heading_keep_scale = float(
            rospy.get_param("~straight_hold_heading_keep_scale", 0.35)
        )
        self.straight_hold_heading_keep_max = float(
            rospy.get_param("~straight_hold_heading_keep_max", 0.08)
        )
        self.straight_hold_heading_deadband_deg = float(
            rospy.get_param("~straight_hold_heading_deadband_deg", 0.0)
        )
        self.straight_hold_allow_during_startup_near_center = bool(
            rospy.get_param("~straight_hold_allow_during_startup_near_center", False)
        )
        self.straight_hold_startup_heading_deg = float(
            rospy.get_param("~straight_hold_startup_heading_deg", 12.0)
        )
        self.straight_hold_startup_lateral_error = float(
            rospy.get_param("~straight_hold_startup_lateral_error", self.straight_hold_lateral_error)
        )
        startup_lateral_sources = str(
            rospy.get_param("~straight_hold_startup_lateral_sources", "")
        )
        self.straight_hold_startup_lateral_sources = {
            source.strip()
            for source in startup_lateral_sources.split(",")
            if source.strip()
        }
        self.sign_flip_heading_gate_deg = float(rospy.get_param("~sign_flip_heading_gate_deg", 6.0))
        self.sign_flip_lateral_gate = float(rospy.get_param("~sign_flip_lateral_gate", 0.15))
        self.far_lateral_error = float(rospy.get_param("~far_lateral_error", 0.30))
        self.far_lateral_gain_scale = float(rospy.get_param("~far_lateral_gain_scale", 1.35))
        self.near_heading_error_deg = float(rospy.get_param("~near_heading_error_deg", 8.0))
        self.near_heading_gain_scale = float(rospy.get_param("~near_heading_gain_scale", 0.45))
        self.large_error_lateral_threshold = float(
            get_param_with_alias(
                "~large_error_lateral_threshold",
                "~startup_lateral_error",
                0.12,
            )
        )
        self.large_error_lateral_gain_scale = float(
            get_param_with_alias(
                "~large_error_lateral_gain_scale",
                "~startup_lateral_gain_scale",
                1.35,
            )
        )
        self.large_error_heading_gain_scale = float(
            get_param_with_alias(
                "~large_error_heading_gain_scale",
                "~startup_heading_gain_scale",
                0.55,
            )
        )
        self.convergence_protect_lateral_threshold = float(
            get_param_with_alias(
                "~convergence_protect_lateral_threshold",
                "~large_error_lateral_threshold",
                0.12,
            )
        )
        self.convergence_opposing_heading_ratio = float(
            rospy.get_param("~convergence_opposing_heading_ratio", 0.5)
        )
        self.convergence_opposing_heading_ratio_strong = float(
            rospy.get_param("~convergence_opposing_heading_ratio_strong", 0.25)
        )
        self.convergence_strong_lateral_threshold = float(
            rospy.get_param("~convergence_strong_lateral_threshold", self.far_lateral_error)
        )
        self.heading_priority_heading_deg = float(
            rospy.get_param("~heading_priority_heading_deg", 45.0)
        )
        self.heading_priority_lateral_ratio = float(
            rospy.get_param("~heading_priority_lateral_ratio", 0.5)
        )
        self.heading_priority_untrusted_lateral_ratio = float(
            rospy.get_param("~heading_priority_untrusted_lateral_ratio", self.heading_priority_lateral_ratio)
        )
        self.heading_priority_untrusted_min_lateral_error = float(
            rospy.get_param("~heading_priority_untrusted_min_lateral_error", self.far_lateral_error)
        )
        untrusted_sources = str(
            rospy.get_param("~heading_priority_untrusted_lateral_sources", "")
        )
        self.heading_priority_untrusted_lateral_sources = {
            source.strip()
            for source in untrusted_sources.split(",")
            if source.strip()
        }
        self.vision_timeout_sec = float(rospy.get_param("~vision_timeout_sec", 2.0))
        self.control_rate_hz = float(rospy.get_param("~control_rate_hz", 20.0))
        self.command_topic = rospy.get_param("~command_topic", "~vehicle_steer_cmd")

        self.interface_valid = False
        self.heading_error_deg: float | None = None
        self.lateral_error: float | None = None
        self.confidence = 0.0
        self.lateral_source = ""
        self.startup_boost_active = False
        self.last_vision_time: float | None = None
        self.last_cmd = 0.0
        self.center_hold_latched = False
        self.straight_hold_latched = False
        self.lateral_integral = 0.0
        self.last_compute_time: float | None = None
        self._warned_unimplemented = False

        rospy.Subscriber("/runway_vision/interface_valid", Bool, self.valid_cb, queue_size=1)
        rospy.Subscriber("/runway_vision/heading_error_deg", Float32, self.heading_cb, queue_size=1)
        rospy.Subscriber(
            "/runway_vision/lateral_error_runway_half_width",
            Float32,
            self.lateral_cb,
            queue_size=1,
        )
        rospy.Subscriber("/runway_vision/confidence", Float32, self.confidence_cb, queue_size=1)
        rospy.Subscriber("/runway_vision/lateral_source", String, self.lateral_source_cb, queue_size=1)
        rospy.Subscriber(
            "/runway_ground_motion_assist/startup_boost_active",
            Bool,
            self.startup_boost_cb,
            queue_size=1,
        )

        self.raw_cmd_pub = rospy.Publisher("~steer_cmd_raw", Float32, queue_size=1)
        self.heading_term_pub = rospy.Publisher("~heading_term", Float32, queue_size=1)
        self.heading_term_raw_pub = rospy.Publisher("~heading_term_raw", Float32, queue_size=1)
        self.lateral_term_pub = rospy.Publisher("~lateral_term", Float32, queue_size=1)
        self.lateral_integral_term_pub = rospy.Publisher("~lateral_integral_term", Float32, queue_size=1)
        self.lateral_integral_state_pub = rospy.Publisher("~lateral_integral_state", Float32, queue_size=1)
        self.heading_error_used_pub = rospy.Publisher("~heading_error_used_deg", Float32, queue_size=1)
        self.lateral_error_used_pub = rospy.Publisher("~lateral_error_used", Float32, queue_size=1)
        self.lateral_error_control_pub = rospy.Publisher("~lateral_error_control", Float32, queue_size=1)
        self.heading_gain_scale_pub = rospy.Publisher("~heading_gain_scale", Float32, queue_size=1)
        self.lateral_gain_scale_pub = rospy.Publisher("~lateral_gain_scale", Float32, queue_size=1)
        self.startup_balance_active_pub = rospy.Publisher("~startup_balance_active", Bool, queue_size=1)
        self.large_error_mode_active_pub = rospy.Publisher("~large_error_mode_active", Bool, queue_size=1)
        self.convergence_protect_active_pub = rospy.Publisher("~convergence_protect_active", Bool, queue_size=1)
        self.convergence_allowed_heading_ratio_pub = rospy.Publisher(
            "~convergence_allowed_heading_ratio",
            Float32,
            queue_size=1,
        )
        self.heading_priority_active_pub = rospy.Publisher("~heading_priority_active", Bool, queue_size=1)
        self.confidence_scale_pub = rospy.Publisher("~confidence_scale", Float32, queue_size=1)
        self.center_hold_active_pub = rospy.Publisher("~center_hold_active", Bool, queue_size=1)
        self.straight_hold_active_pub = rospy.Publisher("~straight_hold_active", Bool, queue_size=1)
        self.straight_hold_ready_pub = rospy.Publisher("~straight_hold_ready", Bool, queue_size=1)
        self.straight_hold_lateral_ready_pub = rospy.Publisher("~straight_hold_lateral_ready", Bool, queue_size=1)
        self.straight_hold_heading_ready_pub = rospy.Publisher("~straight_hold_heading_ready", Bool, queue_size=1)
        self.straight_hold_blocked_startup_pub = rospy.Publisher("~straight_hold_blocked_startup", Bool, queue_size=1)
        self.straight_hold_blocked_center_pub = rospy.Publisher("~straight_hold_blocked_center", Bool, queue_size=1)
        self.straight_hold_heading_cmd_pub = rospy.Publisher("~straight_hold_heading_cmd", Float32, queue_size=1)
        self.sign_flip_blocked_pub = rospy.Publisher("~sign_flip_blocked", Bool, queue_size=1)
        self.near_center_scale_pub = rospy.Publisher("~near_center_scale", Float32, queue_size=1)
        self.scaled_cmd_pub = rospy.Publisher("~steer_cmd_scaled", Float32, queue_size=1)
        self.limited_cmd_pub = rospy.Publisher("~steer_cmd_limited", Float32, queue_size=1)
        self.filtered_cmd_pub = rospy.Publisher("~steer_cmd_filtered", Float32, queue_size=1)
        self.cmd_pub = rospy.Publisher("~steer_cmd", Float32, queue_size=1)
        self.vision_age_pub = rospy.Publisher("~vision_age_sec", Float32, queue_size=1)
        self.vision_timed_out_pub = rospy.Publisher("~vision_timed_out", Bool, queue_size=1)
        self.controller_valid_pub = rospy.Publisher("~controller_valid", Bool, queue_size=1)
        self.saturated_pub = rospy.Publisher("~saturated", Bool, queue_size=1)
        self.rate_limited_pub = rospy.Publisher("~rate_limited", Bool, queue_size=1)
        self.vehicle_cmd_pub = rospy.Publisher(self.command_topic, Float32, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.control_rate_hz, 1e-3)), self.control_timer_cb)
        rospy.loginfo("runway_ground_align_controller running, dry_run=%s", self.dry_run)

    def valid_cb(self, msg: Bool) -> None:
        self.interface_valid = bool(msg.data)
        self.last_vision_time = time.time()

    def heading_cb(self, msg: Float32) -> None:
        self.heading_error_deg = None if math.isnan(msg.data) else float(msg.data)
        self.last_vision_time = time.time()

    def lateral_cb(self, msg: Float32) -> None:
        self.lateral_error = None if math.isnan(msg.data) else float(msg.data)
        self.last_vision_time = time.time()

    def confidence_cb(self, msg: Float32) -> None:
        self.confidence = float(msg.data)
        self.last_vision_time = time.time()

    def lateral_source_cb(self, msg: String) -> None:
        self.lateral_source = str(msg.data)

    def startup_boost_cb(self, msg: Bool) -> None:
        self.startup_boost_active = bool(msg.data)

    def control_timer_cb(self, _event: rospy.timer.TimerEvent) -> None:
        cmd = self.compute_command()
        self.cmd_pub.publish(Float32(cmd))
        self.publish_vehicle_command(cmd)
        self.last_cmd = cmd

    def compute_command(self) -> float:
        raw = 0.0
        heading_error_used = 0.0
        heading_error_for_straight_hold = 0.0
        lateral_error_used = 0.0
        lateral_error_control = 0.0
        heading_term = 0.0
        heading_term_raw = 0.0
        lateral_term = 0.0
        lateral_integral_term = 0.0
        heading_gain_scale = 1.0
        lateral_gain_scale = 1.0
        startup_balance_active = False
        conf_scale = 1.0
        center_hold_active = False
        straight_hold_active = False
        straight_hold_ready = False
        straight_hold_lateral_ready = False
        straight_hold_heading_ready = False
        straight_hold_blocked_startup = False
        straight_hold_blocked_center = False
        straight_hold_heading_cmd = 0.0
        near_center_scale = 1.0
        convergence_protect_active = False
        convergence_allowed_heading_ratio = self.convergence_opposing_heading_ratio
        heading_priority_active = False
        sign_flip_blocked = False
        timed_out = self._vision_timed_out()
        dt = self._compute_dt()

        if not timed_out and self.interface_valid:
            if self.heading_error_deg is not None and self.lateral_error is not None:
                heading_error = self._apply_deadband(self.heading_error_deg, self.heading_deadband_deg)
                lateral_error = self._apply_deadband(self.lateral_error, self.lateral_deadband)
                heading_error_for_straight_hold = heading_error
                heading_error_used = self._clip(heading_error, -self.max_heading_error_deg, self.max_heading_error_deg)
                lateral_error_used = self._clip(lateral_error, -self.max_lateral_error, self.max_lateral_error)
                lateral_error_control = self._clip(
                    lateral_error_used - self.lateral_target_offset,
                    -self.max_lateral_error,
                    self.max_lateral_error,
                )
                center_hold_active = self._update_center_hold(
                    heading_error_used=heading_error_used,
                    lateral_error_control=lateral_error_control,
                )
                straight_hold_state = self._update_straight_hold(
                    heading_error_used=heading_error_used,
                    heading_error_raw=heading_error,
                    lateral_error_control=lateral_error_control,
                    lateral_source=self.lateral_source,
                    center_hold_active=center_hold_active,
                )
                straight_hold_active = straight_hold_state["active"]
                straight_hold_ready = straight_hold_state["ready"]
                straight_hold_lateral_ready = straight_hold_state["lateral_ready"]
                straight_hold_heading_ready = straight_hold_state["heading_ready"]
                straight_hold_blocked_startup = straight_hold_state["blocked_startup"]
                straight_hold_blocked_center = straight_hold_state["blocked_center"]
                if center_hold_active:
                    near_center_scale = self.near_center_command_scale
                (
                    heading_gain_scale,
                    lateral_gain_scale,
                    startup_balance_active,
                ) = self._compute_gain_scales(
                    heading_error_used=heading_error_used,
                    lateral_error_control=lateral_error_control,
                    straight_hold_active=straight_hold_active,
                )
                heading_term_raw = self.heading_sign * self.k_heading * heading_gain_scale * heading_error_used
                heading_term = heading_term_raw
                lateral_term = self.lateral_sign * self.k_lateral * lateral_gain_scale * lateral_error_control
                (
                    lateral_term,
                    heading_priority_active,
                ) = self._apply_heading_priority(
                    heading_error_raw=heading_error,
                    lateral_error_control=lateral_error_control,
                    lateral_source=self.lateral_source,
                    heading_term=heading_term,
                    lateral_term=lateral_term,
                )
                if not heading_priority_active:
                    (
                        heading_term,
                        convergence_protect_active,
                        convergence_allowed_heading_ratio,
                    ) = self._apply_convergence_protection(
                        heading_term=heading_term,
                        lateral_term=lateral_term,
                        lateral_error_control=lateral_error_control,
                    )
                lateral_integral_term = self._update_lateral_integral(
                    lateral_error_used=lateral_error_control,
                    heading_error_used=heading_error_used,
                    dt=dt,
                )
                raw = heading_term + lateral_term
                raw += lateral_integral_term

        self.raw_cmd_pub.publish(Float32(raw))
        self.heading_error_used_pub.publish(Float32(heading_error_used))
        self.lateral_error_used_pub.publish(Float32(lateral_error_used))
        self.lateral_error_control_pub.publish(Float32(lateral_error_control))
        self.heading_gain_scale_pub.publish(Float32(heading_gain_scale))
        self.lateral_gain_scale_pub.publish(Float32(lateral_gain_scale))
        self.startup_balance_active_pub.publish(Bool(startup_balance_active))
        self.large_error_mode_active_pub.publish(Bool(startup_balance_active))
        self.convergence_protect_active_pub.publish(Bool(convergence_protect_active))
        self.convergence_allowed_heading_ratio_pub.publish(Float32(convergence_allowed_heading_ratio))
        self.heading_priority_active_pub.publish(Bool(heading_priority_active))
        self.heading_term_raw_pub.publish(Float32(heading_term_raw))
        self.heading_term_pub.publish(Float32(heading_term))
        self.lateral_term_pub.publish(Float32(lateral_term))
        self.lateral_integral_term_pub.publish(Float32(lateral_integral_term))
        self.lateral_integral_state_pub.publish(Float32(self.lateral_integral))
        self.center_hold_active_pub.publish(Bool(center_hold_active))
        self.straight_hold_active_pub.publish(Bool(straight_hold_active))
        self.straight_hold_ready_pub.publish(Bool(straight_hold_ready))
        self.straight_hold_lateral_ready_pub.publish(Bool(straight_hold_lateral_ready))
        self.straight_hold_heading_ready_pub.publish(Bool(straight_hold_heading_ready))
        self.straight_hold_blocked_startup_pub.publish(Bool(straight_hold_blocked_startup))
        self.straight_hold_blocked_center_pub.publish(Bool(straight_hold_blocked_center))
        self.vision_timed_out_pub.publish(Bool(timed_out))
        self.controller_valid_pub.publish(
            Bool(
                (not timed_out)
                and self.interface_valid
                and self.heading_error_deg is not None
                and self.lateral_error is not None
            )
        )
        self.vision_age_pub.publish(Float32(self._vision_age_sec()))

        if timed_out or not self.interface_valid:
            self.straight_hold_latched = False
            self._decay_lateral_integral(reset=True)
            self.confidence_scale_pub.publish(Float32(0.0))
            self.near_center_scale_pub.publish(Float32(0.0))
            self.heading_gain_scale_pub.publish(Float32(0.0))
            self.lateral_gain_scale_pub.publish(Float32(0.0))
            self.scaled_cmd_pub.publish(Float32(0.0))
            self.limited_cmd_pub.publish(Float32(0.0))
            self.filtered_cmd_pub.publish(Float32(0.0))
            self.straight_hold_heading_cmd_pub.publish(Float32(0.0))
            self.saturated_pub.publish(Bool(False))
            self.rate_limited_pub.publish(Bool(False))
            self.sign_flip_blocked_pub.publish(Bool(False))
            self.heading_priority_active_pub.publish(Bool(False))
            return 0.0

        if self.heading_error_deg is None or self.lateral_error is None:
            self.straight_hold_latched = False
            self._decay_lateral_integral(reset=True)
            self.confidence_scale_pub.publish(Float32(0.0))
            self.near_center_scale_pub.publish(Float32(0.0))
            self.heading_gain_scale_pub.publish(Float32(0.0))
            self.lateral_gain_scale_pub.publish(Float32(0.0))
            self.scaled_cmd_pub.publish(Float32(0.0))
            self.limited_cmd_pub.publish(Float32(0.0))
            self.filtered_cmd_pub.publish(Float32(0.0))
            self.straight_hold_heading_cmd_pub.publish(Float32(0.0))
            self.saturated_pub.publish(Bool(False))
            self.rate_limited_pub.publish(Bool(False))
            self.sign_flip_blocked_pub.publish(Bool(False))
            self.heading_priority_active_pub.publish(Bool(False))
            return 0.0

        if self.confidence < self.conf_min:
            conf_scale = self.low_conf_scale
            raw *= conf_scale

        scaled = raw * self.command_scale * near_center_scale
        if straight_hold_active:
            straight_hold_heading_error = self._apply_deadband(
                heading_error_for_straight_hold,
                self.straight_hold_heading_deadband_deg,
            )
            straight_hold_heading_error = self._clip(
                straight_hold_heading_error,
                -self.max_heading_error_deg,
                self.max_heading_error_deg,
            )
            straight_hold_heading_cmd = self._clip(
                self.heading_sign * self.k_heading * straight_hold_heading_error * self.straight_hold_heading_keep_scale,
                -self.straight_hold_heading_keep_max,
                self.straight_hold_heading_keep_max,
            )
            scaled = straight_hold_heading_cmd
        if center_hold_active and self.center_hold_zero_cmd:
            scaled = 0.0

        proposed_sign = self._signum(scaled)
        previous_sign = self._signum(self.last_cmd)
        if (
            proposed_sign != 0
            and previous_sign != 0
            and proposed_sign != previous_sign
            and abs(heading_error_used) <= self.sign_flip_heading_gate_deg
            and abs(lateral_error_control) <= self.sign_flip_lateral_gate
        ):
            scaled = 0.0
            sign_flip_blocked = True

        limited = self._clip(scaled, -self.max_steer, self.max_steer)
        limited = self._apply_min_effective_steer(
            limited,
            heading_error_used=heading_error_used,
            lateral_error_used=lateral_error_used,
        )
        filtered = self.filter_alpha * self.last_cmd + (1.0 - self.filter_alpha) * limited
        cmd = float(self._rate_limit(filtered, self.last_cmd, self.max_delta))
        saturated = abs(scaled) > (self.max_steer + 1e-6)
        rate_limited = abs(cmd - filtered) > 1e-6
        self.confidence_scale_pub.publish(Float32(conf_scale))
        self.near_center_scale_pub.publish(Float32(near_center_scale))
        self.scaled_cmd_pub.publish(Float32(scaled))
        self.limited_cmd_pub.publish(Float32(limited))
        self.filtered_cmd_pub.publish(Float32(filtered))
        self.straight_hold_heading_cmd_pub.publish(Float32(straight_hold_heading_cmd))
        self.saturated_pub.publish(Bool(saturated))
        self.rate_limited_pub.publish(Bool(rate_limited))
        self.sign_flip_blocked_pub.publish(Bool(sign_flip_blocked))
        rospy.loginfo_throttle(
            1.0,
            "runway_ground_align_controller: heading=%.3f lateral=%.3f lat_ctrl=%.3f lat_i=%.3f raw=%.3f scale=%.2f center_scale=%.2f h_scale=%.2f l_scale=%.2f scaled=%.3f limited=%.3f filtered=%.3f cmd=%.3f conf=%.2f conf_scale=%.2f center_hold=%s straight_ready=%s straight_hold=%s straight_lat_ready=%s straight_head_ready=%s straight_blk_start=%s straight_blk_center=%s straight_head_cmd=%.3f flip_blocked=%s",
            heading_term,
            lateral_term,
            lateral_error_control,
            lateral_integral_term,
            raw,
            self.command_scale,
            near_center_scale,
            heading_gain_scale,
            lateral_gain_scale,
            scaled,
            limited,
            filtered,
            cmd,
            self.confidence,
            conf_scale,
            center_hold_active,
            straight_hold_ready,
            straight_hold_active,
            straight_hold_lateral_ready,
            straight_hold_heading_ready,
            straight_hold_blocked_startup,
            straight_hold_blocked_center,
            straight_hold_heading_cmd,
            sign_flip_blocked,
        )
        return cmd

    def _apply_heading_priority(
        self,
        heading_error_raw: float,
        lateral_error_control: float,
        lateral_source: str,
        heading_term: float,
        lateral_term: float,
    ) -> tuple[float, bool]:
        if abs(heading_error_raw) < self.heading_priority_heading_deg:
            return lateral_term, False
        if abs(heading_term) < 1e-6 or heading_term * lateral_term >= 0.0:
            return lateral_term, False

        untrusted_large_lateral = (
            lateral_source in self.heading_priority_untrusted_lateral_sources
            and abs(lateral_error_control) >= self.heading_priority_untrusted_min_lateral_error
        )
        if abs(lateral_error_control) > self.large_error_lateral_threshold and not untrusted_large_lateral:
            return lateral_term, False

        lateral_ratio = (
            self.heading_priority_untrusted_lateral_ratio
            if untrusted_large_lateral
            else self.heading_priority_lateral_ratio
        )
        lateral_limit = abs(heading_term) * max(0.0, lateral_ratio)
        protected_lateral = math.copysign(min(abs(lateral_term), lateral_limit), lateral_term)
        return protected_lateral, True

    def _update_center_hold(self, heading_error_used: float, lateral_error_control: float) -> bool:
        inside_hold = (
            abs(heading_error_used) <= self.center_hold_heading_deg
            and abs(lateral_error_control) <= self.center_hold_lateral_error
        )
        outside_release = (
            abs(heading_error_used) >= self.center_release_heading_deg
            or abs(lateral_error_control) >= self.center_release_lateral_error
        )
        if inside_hold:
            self.center_hold_latched = True
        elif outside_release:
            self.center_hold_latched = False
        return self.center_hold_latched

    def _update_straight_hold(
        self,
        heading_error_used: float,
        heading_error_raw: float,
        lateral_error_control: float,
        lateral_source: str,
        center_hold_active: bool,
    ) -> dict[str, bool]:
        lateral_ready = abs(lateral_error_control) <= self.straight_hold_lateral_error
        heading_ready = abs(heading_error_used) <= self.straight_hold_heading_deg
        if self.straight_hold_lateral_priority:
            inside_hold = lateral_ready
            outside_release = abs(lateral_error_control) >= self.straight_release_lateral_error
        else:
            inside_hold = heading_ready and lateral_ready
            outside_release = (
                abs(heading_error_used) >= self.straight_release_heading_deg
                or abs(lateral_error_control) >= self.straight_release_lateral_error
            )

        startup_allows = (
            (not self.straight_hold_require_startup_boost_inactive)
            or (not self.startup_boost_active)
        )
        if not startup_allows:
            startup_allows = self._startup_near_center_allows_straight_hold(
                heading_error_raw=heading_error_raw,
                lateral_error_control=lateral_error_control,
                lateral_source=lateral_source,
            )
        center_allows = (
            (not self.straight_hold_require_center_hold)
            or center_hold_active
        )
        blocked_startup = inside_hold and not startup_allows
        blocked_center = inside_hold and startup_allows and not center_allows
        ready = inside_hold and startup_allows and center_allows

        if ready:
            self.straight_hold_latched = True
        elif outside_release or not startup_allows or not center_allows:
            self.straight_hold_latched = False

        return {
            "active": self.straight_hold_latched,
            "ready": ready,
            "lateral_ready": lateral_ready,
            "heading_ready": heading_ready,
            "blocked_startup": blocked_startup,
            "blocked_center": blocked_center,
        }

    def _startup_near_center_allows_straight_hold(
        self,
        heading_error_raw: float,
        lateral_error_control: float,
        lateral_source: str,
    ) -> bool:
        if not self.straight_hold_allow_during_startup_near_center:
            return False
        if not self.startup_boost_active:
            return False
        if abs(heading_error_raw) > self.straight_hold_startup_heading_deg:
            return False
        if abs(lateral_error_control) > self.straight_hold_startup_lateral_error:
            return False
        if (
            self.straight_hold_startup_lateral_sources
            and lateral_source not in self.straight_hold_startup_lateral_sources
        ):
            return False
        return True

    def _compute_gain_scales(
        self,
        heading_error_used: float,
        lateral_error_control: float,
        straight_hold_active: bool,
    ) -> tuple[float, float, bool]:
        heading_gain_scale = 1.0
        lateral_gain_scale = 1.0
        large_error_mode_active = False

        if abs(lateral_error_control) >= self.far_lateral_error:
            lateral_gain_scale = self.far_lateral_gain_scale
        if abs(heading_error_used) <= self.near_heading_error_deg:
            heading_gain_scale = self.near_heading_gain_scale
        if (
            abs(lateral_error_control) >= self.large_error_lateral_threshold
            and not straight_hold_active
        ):
            large_error_mode_active = True
            lateral_gain_scale = max(lateral_gain_scale, self.large_error_lateral_gain_scale)
            heading_gain_scale = min(heading_gain_scale, self.large_error_heading_gain_scale)

        return heading_gain_scale, lateral_gain_scale, large_error_mode_active

    def _apply_convergence_protection(
        self,
        heading_term: float,
        lateral_term: float,
        lateral_error_control: float,
    ) -> tuple[float, bool, float]:
        allowed_ratio = self._convergence_allowed_heading_ratio(lateral_error_control)
        if abs(lateral_error_control) < self.convergence_protect_lateral_threshold:
            return heading_term, False, allowed_ratio
        if abs(lateral_term) < 1e-6 or heading_term * lateral_term >= 0.0:
            return heading_term, False, allowed_ratio

        protected_heading = math.copysign(
            min(abs(heading_term), abs(lateral_term) * allowed_ratio),
            heading_term,
        )
        return protected_heading, True, allowed_ratio

    def _convergence_allowed_heading_ratio(self, lateral_error_control: float) -> float:
        base_ratio = max(0.0, self.convergence_opposing_heading_ratio)
        strong_ratio = max(0.0, min(base_ratio, self.convergence_opposing_heading_ratio_strong))
        strong_threshold = max(
            self.convergence_strong_lateral_threshold,
            self.convergence_protect_lateral_threshold,
        )
        if strong_threshold <= self.convergence_protect_lateral_threshold + 1e-6:
            return strong_ratio

        progress = (
            abs(lateral_error_control) - self.convergence_protect_lateral_threshold
        ) / (strong_threshold - self.convergence_protect_lateral_threshold)
        progress = self._clip(progress, 0.0, 1.0)
        return base_ratio + (strong_ratio - base_ratio) * progress

    def publish_vehicle_command(self, steer_cmd: float) -> None:
        if self.dry_run:
            return

        if not self._warned_unimplemented:
            rospy.logwarn(
                "runway_ground_align_controller is publishing Float32 debug commands to %s. "
                "Replace publish_vehicle_command() with your final MAVROS/PX4 actuator bridge.",
                self.command_topic,
            )
            self._warned_unimplemented = True

        self.vehicle_cmd_pub.publish(Float32(steer_cmd))

    def _vision_timed_out(self) -> bool:
        if self.last_vision_time is None:
            return True
        return (time.time() - self.last_vision_time) > self.vision_timeout_sec

    def _vision_age_sec(self) -> float:
        if self.last_vision_time is None:
            return float("inf")
        return float(time.time() - self.last_vision_time)

    def _compute_dt(self) -> float:
        now = time.time()
        if self.last_compute_time is None:
            self.last_compute_time = now
            return 0.0
        dt = max(0.0, min(0.2, now - self.last_compute_time))
        self.last_compute_time = now
        return dt

    def _clip(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _signum(self, value: float) -> int:
        if value > 1e-6:
            return 1
        if value < -1e-6:
            return -1
        return 0

    def _apply_deadband(self, value: float, deadband: float) -> float:
        if deadband <= 0.0:
            return value
        if abs(value) <= deadband:
            return 0.0
        return math.copysign(abs(value) - deadband, value)

    def _decay_lateral_integral(self, reset: bool = False) -> None:
        if reset:
            self.lateral_integral = 0.0
            return
        self.lateral_integral *= self.lateral_integral_leak

    def _update_lateral_integral(self, lateral_error_used: float, heading_error_used: float, dt: float) -> float:
        if self.k_lateral_i <= 0.0 or dt <= 0.0:
            return self.k_lateral_i * self.lateral_integral

        if (
            abs(heading_error_used) <= self.integral_heading_gate_deg
            and abs(lateral_error_used) <= self.integral_zone_lateral_error
        ):
            self.lateral_integral += lateral_error_used * dt
            self.lateral_integral = self._clip(
                self.lateral_integral,
                -self.lateral_integral_limit,
                self.lateral_integral_limit,
            )
        else:
            self._decay_lateral_integral(reset=False)

        return self.k_lateral_i * self.lateral_integral

    def _apply_min_effective_steer(
        self,
        value: float,
        heading_error_used: float,
        lateral_error_used: float,
    ) -> float:
        minimum = abs(self.min_effective_steer)
        if minimum <= 0.0 or abs(value) < 1e-6:
            return value
        allow_minimum = (
            abs(heading_error_used) >= self.min_effective_heading_gate_deg
            or abs(lateral_error_used) >= self.min_effective_lateral_gate
        )
        if not allow_minimum:
            return value
        return math.copysign(max(abs(value), minimum), value)

    def _rate_limit(self, value: float, previous: float, max_delta: float) -> float:
        delta = value - previous
        delta = self._clip(delta, -max_delta, max_delta)
        return previous + delta


def main() -> None:
    rospy.init_node("runway_ground_align_controller")
    RunwayGroundAlignController()
    rospy.spin()


if __name__ == "__main__":
    main()
