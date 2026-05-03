#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rospy
from std_msgs.msg import Bool, Float32, String


class TopicValue:
    def __init__(self, name: str, value_type: str) -> None:
        self.name = name
        self.value_type = value_type
        self.value: float | bool | str | None = None
        self.timestamp: float | None = None

    def update(self, value: float | bool | str) -> None:
        self.value = value
        self.timestamp = time.time()

    def format(self, stale_timeout_sec: float) -> str:
        if self.value is None or self.timestamp is None:
            return f"{self.name}=--"

        age = time.time() - self.timestamp
        stale_marker = "!" if age > stale_timeout_sec else ""

        if self.value_type == "bool":
            text = "1" if bool(self.value) else "0"
        elif self.value_type == "string":
            text = str(self.value) if str(self.value) else "--"
        else:
            numeric = float(self.value)
            if math.isnan(numeric):
                text = "nan"
            else:
                text = f"{numeric:+.3f}"
        return f"{self.name}={text}{stale_marker}"


class RunwayClosedLoopMonitor:
    def __init__(self) -> None:
        self.rate_hz = float(rospy.get_param("~rate_hz", 5.0))
        self.stale_timeout_sec = float(rospy.get_param("~stale_timeout_sec", 1.0))

        self.values = {
            "vision_valid": TopicValue("vision_valid", "bool"),
            "ctrl_valid": TopicValue("ctrl_valid", "bool"),
            "timed_out": TopicValue("timed_out", "bool"),
            "assist_active": TopicValue("assist_active", "bool"),
            "align_hold": TopicValue("align_hold", "bool"),
            "startup_boost": TopicValue("startup_boost", "bool"),
            "startup_armed": TopicValue("startup_armed", "bool"),
            "straight_hold": TopicValue("straight_hold", "bool"),
            "straight_ready": TopicValue("straight_ready", "bool"),
            "straight_lat_ready": TopicValue("straight_lat_ready", "bool"),
            "straight_head_ready": TopicValue("straight_head_ready", "bool"),
            "straight_blk_start": TopicValue("straight_blk_start", "bool"),
            "straight_blk_center": TopicValue("straight_blk_center", "bool"),
            "saturated": TopicValue("saturated", "bool"),
            "rate_limited": TopicValue("rate_limited", "bool"),
            "center_hold": TopicValue("center_hold", "bool"),
            "startup_bal": TopicValue("startup_bal", "bool"),
            "large_err": TopicValue("large_err", "bool"),
            "conv_protect": TopicValue("conv_protect", "bool"),
            "conv_ratio": TopicValue("conv_ratio", "float"),
            "head_prio": TopicValue("head_prio", "bool"),
            "flip_blocked": TopicValue("flip_blocked", "bool"),
            "head_deg": TopicValue("head_deg", "float"),
            "lat_hw": TopicValue("lat_hw", "float"),
            "conf": TopicValue("conf", "float"),
            "geom_src": TopicValue("geom_src", "string"),
            "lat_src": TopicValue("lat_src", "string"),
            "infer_ms": TopicValue("infer_ms", "float"),
            "img_age_ms": TopicValue("img_age_ms", "float"),
            "pipe_ms": TopicValue("pipe_ms", "float"),
            "center_scale": TopicValue("center_scale", "float"),
            "h_gain": TopicValue("h_gain", "float"),
            "l_gain": TopicValue("l_gain", "float"),
            "lat_i_term": TopicValue("lat_i_term", "float"),
            "lat_i_state": TopicValue("lat_i_state", "float"),
            "head_used": TopicValue("head_used", "float"),
            "lat_used": TopicValue("lat_used", "float"),
            "lat_ctrl": TopicValue("lat_ctrl", "float"),
            "head_term_raw": TopicValue("head_term_raw", "float"),
            "head_term": TopicValue("head_term", "float"),
            "lat_term": TopicValue("lat_term", "float"),
            "raw": TopicValue("raw", "float"),
            "scaled": TopicValue("scaled", "float"),
            "limited": TopicValue("limited", "float"),
            "filtered": TopicValue("filtered", "float"),
            "cmd": TopicValue("cmd", "float"),
            "straight_hdg_cmd": TopicValue("straight_hdg_cmd", "float"),
            "steer_used": TopicValue("steer_used", "float"),
            "yaw_deg_s": TopicValue("yaw_deg_s", "float"),
            "speed_mps": TopicValue("speed_mps", "float"),
            "speed_scale": TopicValue("speed_scale", "float"),
            "boost_scale": TopicValue("boost_scale", "float"),
        }

        self._sub_bool("/runway_vision/interface_valid", "vision_valid")
        self._sub_bool("/runway_ground_align_controller/controller_valid", "ctrl_valid")
        self._sub_bool("/runway_ground_align_controller/vision_timed_out", "timed_out")
        self._sub_bool("/runway_ground_motion_assist/active", "assist_active")
        self._sub_bool("/runway_ground_motion_assist/align_hold_active", "align_hold")
        self._sub_bool("/runway_ground_motion_assist/startup_boost_active", "startup_boost")
        self._sub_bool("/runway_ground_motion_assist/startup_boost_armed", "startup_armed")
        self._sub_bool("/runway_ground_align_controller/straight_hold_active", "straight_hold")
        self._sub_bool("/runway_ground_align_controller/straight_hold_ready", "straight_ready")
        self._sub_bool("/runway_ground_align_controller/straight_hold_lateral_ready", "straight_lat_ready")
        self._sub_bool("/runway_ground_align_controller/straight_hold_heading_ready", "straight_head_ready")
        self._sub_bool("/runway_ground_align_controller/straight_hold_blocked_startup", "straight_blk_start")
        self._sub_bool("/runway_ground_align_controller/straight_hold_blocked_center", "straight_blk_center")
        self._sub_bool("/runway_ground_align_controller/saturated", "saturated")
        self._sub_bool("/runway_ground_align_controller/rate_limited", "rate_limited")
        self._sub_bool("/runway_ground_align_controller/center_hold_active", "center_hold")
        self._sub_bool("/runway_ground_align_controller/startup_balance_active", "startup_bal")
        self._sub_bool("/runway_ground_align_controller/large_error_mode_active", "large_err")
        self._sub_bool("/runway_ground_align_controller/convergence_protect_active", "conv_protect")
        self._sub_float("/runway_ground_align_controller/convergence_allowed_heading_ratio", "conv_ratio")
        self._sub_bool("/runway_ground_align_controller/heading_priority_active", "head_prio")
        self._sub_bool("/runway_ground_align_controller/sign_flip_blocked", "flip_blocked")

        self._sub_float("/runway_vision/heading_error_deg", "head_deg")
        self._sub_float("/runway_vision/lateral_error_runway_half_width", "lat_hw")
        self._sub_float("/runway_vision/confidence", "conf")
        self._sub_string("/runway_vision/geometry_source", "geom_src")
        self._sub_string("/runway_vision/lateral_source", "lat_src")
        self._sub_float("/runway_vision/inference_ms", "infer_ms")
        self._sub_float("/runway_vision/image_age_ms", "img_age_ms")
        self._sub_float("/runway_vision/pipeline_latency_ms", "pipe_ms")
        self._sub_float("/runway_ground_align_controller/near_center_scale", "center_scale")
        self._sub_float("/runway_ground_align_controller/heading_gain_scale", "h_gain")
        self._sub_float("/runway_ground_align_controller/lateral_gain_scale", "l_gain")
        self._sub_float("/runway_ground_align_controller/lateral_integral_term", "lat_i_term")
        self._sub_float("/runway_ground_align_controller/lateral_integral_state", "lat_i_state")
        self._sub_float("/runway_ground_align_controller/heading_error_used_deg", "head_used")
        self._sub_float("/runway_ground_align_controller/lateral_error_used", "lat_used")
        self._sub_float("/runway_ground_align_controller/lateral_error_control", "lat_ctrl")
        self._sub_float("/runway_ground_align_controller/heading_term_raw", "head_term_raw")
        self._sub_float("/runway_ground_align_controller/heading_term", "head_term")
        self._sub_float("/runway_ground_align_controller/lateral_term", "lat_term")
        self._sub_float("/runway_ground_align_controller/steer_cmd_raw", "raw")
        self._sub_float("/runway_ground_align_controller/steer_cmd_scaled", "scaled")
        self._sub_float("/runway_ground_align_controller/steer_cmd_limited", "limited")
        self._sub_float("/runway_ground_align_controller/steer_cmd_filtered", "filtered")
        self._sub_float("/runway_ground_align_controller/steer_cmd", "cmd")
        self._sub_float("/runway_ground_align_controller/straight_hold_heading_cmd", "straight_hdg_cmd")
        self._sub_float("/runway_ground_motion_assist/steer_used", "steer_used")
        self._sub_float("/runway_ground_motion_assist/yaw_rate_cmd_deg_s", "yaw_deg_s")
        self._sub_float("/runway_ground_motion_assist/speed_cmd_mps", "speed_mps")
        self._sub_float("/runway_ground_motion_assist/speed_scale", "speed_scale")
        self._sub_float("/runway_ground_motion_assist/startup_boost_scale", "boost_scale")

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1e-3)), self.timer_cb)
        rospy.loginfo("runway_closed_loop_monitor running at %.1f Hz", self.rate_hz)

    def _sub_float(self, topic: str, key: str) -> None:
        rospy.Subscriber(topic, Float32, self._float_cb, callback_args=key, queue_size=1)

    def _sub_bool(self, topic: str, key: str) -> None:
        rospy.Subscriber(topic, Bool, self._bool_cb, callback_args=key, queue_size=1)

    def _sub_string(self, topic: str, key: str) -> None:
        rospy.Subscriber(topic, String, self._string_cb, callback_args=key, queue_size=1)

    def _float_cb(self, msg: Float32, key: str) -> None:
        self.values[key].update(float(msg.data))

    def _bool_cb(self, msg: Bool, key: str) -> None:
        self.values[key].update(bool(msg.data))

    def _string_cb(self, msg: String, key: str) -> None:
        self.values[key].update(str(msg.data))

    def timer_cb(self, _event: rospy.timer.TimerEvent) -> None:
        groups = [
            ("state", ["vision_valid", "ctrl_valid", "timed_out", "assist_active", "align_hold", "startup_boost", "startup_armed", "startup_bal", "large_err", "conv_protect", "conv_ratio", "head_prio", "straight_ready", "straight_hold", "straight_lat_ready", "straight_head_ready", "straight_blk_start", "straight_blk_center", "saturated", "rate_limited", "center_hold", "flip_blocked"]),
            ("vision", ["head_deg", "lat_hw", "conf", "geom_src", "lat_src", "infer_ms", "img_age_ms", "pipe_ms"]),
            ("ctrl", ["center_scale", "h_gain", "l_gain", "lat_i_term", "lat_i_state", "head_used", "lat_used", "lat_ctrl", "head_term_raw", "head_term", "lat_term", "raw", "scaled", "limited", "filtered", "cmd", "straight_hdg_cmd"]),
            ("assist", ["steer_used", "yaw_deg_s", "speed_mps", "speed_scale", "boost_scale"]),
        ]

        chunks = []
        for label, keys in groups:
            text = " ".join(self.values[key].format(self.stale_timeout_sec) for key in keys)
            chunks.append(f"{label}: {text}")

        rospy.loginfo_throttle_identical(0.15, " | ".join(chunks))


def main() -> None:
    rospy.init_node("runway_closed_loop_monitor")
    RunwayClosedLoopMonitor()
    rospy.spin()


if __name__ == "__main__":
    main()
