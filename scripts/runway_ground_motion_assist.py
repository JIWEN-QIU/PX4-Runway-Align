#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import ManualControl, State
from std_msgs.msg import Bool, Float32


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


class RunwayGroundMotionAssist:
    """Simple kinematic ground-motion helper for validating visual alignment.

    The stock PX4 plane model is not built as a realistic steerable-wheel
    ground vehicle. This node leaves the SDF intact and only applies a bounded
    planar motion model while the vehicle is armed and throttle is present.
    """

    def __init__(self) -> None:
        self.model_name = rospy.get_param("~model_name", "plane")
        self.reference_frame = rospy.get_param("~reference_frame", "world")
        self.steer_topic = rospy.get_param("~steer_topic", "/runway_ground_align_controller/steer_cmd")
        self.manual_topic = rospy.get_param("~manual_topic", "/mavros/manual_control/send")
        self.state_topic = rospy.get_param("~state_topic", "/mavros/state")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 0.9))
        self.min_throttle = float(rospy.get_param("~min_throttle", 40.0))
        self.speed_override_mps = float(rospy.get_param("~speed_override_mps", -1.0))
        self.max_yaw_rate_deg_s = float(rospy.get_param("~max_yaw_rate_deg_s", 35.0))
        self.steer_sign = float(rospy.get_param("~steer_sign", 1.0))
        self.steer_override = float(rospy.get_param("~steer_override", "nan"))
        self.command_timeout_sec = float(rospy.get_param("~command_timeout_sec", 0.5))
        self.hold_z = bool(rospy.get_param("~hold_z", True))
        self.enable_startup_steer_boost = bool(rospy.get_param("~enable_startup_steer_boost", False))
        self.startup_boost_heading_deg = float(rospy.get_param("~startup_boost_heading_deg", 5.0))
        self.startup_boost_lateral_error = float(rospy.get_param("~startup_boost_lateral_error", 0.12))
        self.startup_boost_release_heading_deg = float(
            rospy.get_param("~startup_boost_release_heading_deg", 2.5)
        )
        self.startup_boost_release_lateral_error = float(
            rospy.get_param("~startup_boost_release_lateral_error", 0.05)
        )
        self.startup_boost_gain = float(rospy.get_param("~startup_boost_gain", 1.45))
        self.startup_boost_max_steer = float(rospy.get_param("~startup_boost_max_steer", 0.75))
        self.startup_boost_one_shot = bool(rospy.get_param("~startup_boost_one_shot", True))
        self.enable_align_hold = bool(rospy.get_param("~enable_align_hold", False))
        self.align_hold_heading_deg = float(rospy.get_param("~align_hold_heading_deg", 6.0))
        self.align_hold_lateral_error = float(rospy.get_param("~align_hold_lateral_error", 0.18))
        self.align_release_heading_deg = float(rospy.get_param("~align_release_heading_deg", 3.0))
        self.align_release_lateral_error = float(rospy.get_param("~align_release_lateral_error", 0.08))
        self.align_min_speed_scale = float(rospy.get_param("~align_min_speed_scale", 0.15))

        self.last_steer = 0.0
        self.last_steer_time: float | None = None
        self.manual_z = 0.0
        self.last_manual_time: float | None = None
        self.armed = False
        self.last_update_time: float | None = None
        self.fixed_z: float | None = None
        self.heading_error_deg: float | None = None
        self.lateral_error: float | None = None
        self.align_hold_active = True
        self.startup_boost_active = False
        self.startup_boost_armed = True

        rospy.Subscriber(self.steer_topic, Float32, self.steer_cb, queue_size=1)
        rospy.Subscriber(self.manual_topic, ManualControl, self.manual_cb, queue_size=1)
        rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        rospy.Subscriber("/runway_vision/heading_error_deg", Float32, self.heading_cb, queue_size=1)
        rospy.Subscriber(
            "/runway_vision/lateral_error_runway_half_width",
            Float32,
            self.lateral_cb,
            queue_size=1,
        )

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.speed_pub = rospy.Publisher("~speed_cmd_mps", Float32, queue_size=1)
        self.steer_pub = rospy.Publisher("~steer_used", Float32, queue_size=1)
        self.yaw_rate_pub = rospy.Publisher("~yaw_rate_cmd_deg_s", Float32, queue_size=1)
        self.align_hold_pub = rospy.Publisher("~align_hold_active", Bool, queue_size=1)
        self.speed_scale_pub = rospy.Publisher("~speed_scale", Float32, queue_size=1)
        self.startup_boost_pub = rospy.Publisher("~startup_boost_active", Bool, queue_size=1)
        self.startup_boost_armed_pub = rospy.Publisher("~startup_boost_armed", Bool, queue_size=1)
        self.startup_boost_scale_pub = rospy.Publisher("~startup_boost_scale", Float32, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1e-3)), self.timer_cb)
        rospy.loginfo(
            "runway_ground_motion_assist running: model=%s max_speed=%.2f max_yaw_rate=%.1f deg/s",
            self.model_name,
            self.max_speed_mps,
            self.max_yaw_rate_deg_s,
        )

    def steer_cb(self, msg: Float32) -> None:
        if math.isnan(msg.data):
            return
        self.last_steer = max(-1.0, min(1.0, float(msg.data)))
        self.last_steer_time = time.time()

    def manual_cb(self, msg: ManualControl) -> None:
        self.manual_z = float(msg.z)
        self.last_manual_time = time.time()

    def state_cb(self, msg: State) -> None:
        self.armed = bool(msg.armed)

    def heading_cb(self, msg: Float32) -> None:
        self.heading_error_deg = None if math.isnan(msg.data) else float(msg.data)

    def lateral_cb(self, msg: Float32) -> None:
        self.lateral_error = None if math.isnan(msg.data) else float(msg.data)

    def timer_cb(self, _event: rospy.timer.TimerEvent) -> None:
        now = time.time()
        if self.last_update_time is None:
            self.last_update_time = now
            self.active_pub.publish(Bool(False))
            return

        dt = max(0.0, min(0.2, now - self.last_update_time))
        self.last_update_time = now

        use_steer_override = math.isfinite(self.steer_override)
        steer_fresh = use_steer_override or (
            self.last_steer_time is not None and (now - self.last_steer_time) <= self.command_timeout_sec
        )
        manual_fresh = self.last_manual_time is not None and (now - self.last_manual_time) <= self.command_timeout_sec
        throttle = self.manual_z if manual_fresh else 0.0
        use_speed_override = self.speed_override_mps >= 0.0
        active = self.armed and steer_fresh and (use_speed_override or throttle > self.min_throttle)
        self.active_pub.publish(Bool(active))

        if not active:
            self.startup_boost_active = False
            self.startup_boost_armed = True
            self.speed_pub.publish(Float32(0.0))
            self.yaw_rate_pub.publish(Float32(0.0))
            self.startup_boost_pub.publish(Bool(False))
            self.startup_boost_armed_pub.publish(Bool(True))
            self.startup_boost_scale_pub.publish(Float32(1.0))
            return

        if use_speed_override:
            speed = max(0.0, min(self.max_speed_mps, self.speed_override_mps))
        else:
            speed = self.max_speed_mps * max(0.0, min(1.0, throttle / 1000.0))
        steer = self.steer_override if use_steer_override else self.last_steer
        steer = max(-1.0, min(1.0, steer))
        startup_boost_scale, startup_boost_active = self._startup_boost_scale()
        steer = math.copysign(
            min(abs(steer) * startup_boost_scale, self.startup_boost_max_steer),
            steer,
        )
        speed_scale = self._alignment_speed_scale()
        speed *= speed_scale
        yaw_rate = math.radians(self.max_yaw_rate_deg_s) * self.steer_sign * steer
        self.align_hold_pub.publish(Bool(self.align_hold_active))
        self.speed_scale_pub.publish(Float32(speed_scale))
        self.startup_boost_pub.publish(Bool(startup_boost_active))
        self.startup_boost_armed_pub.publish(Bool(self.startup_boost_armed))
        self.startup_boost_scale_pub.publish(Float32(startup_boost_scale))
        self.speed_pub.publish(Float32(speed))
        self.steer_pub.publish(Float32(steer))
        self.yaw_rate_pub.publish(Float32(math.degrees(yaw_rate)))

        try:
            state = self.get_model_state(self.model_name, self.reference_frame)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "get_model_state failed: %s", exc)
            return

        if not state.success:
            rospy.logwarn_throttle(2.0, "get_model_state unsuccessful: %s", state.status_message)
            return

        pose = state.pose
        yaw = yaw_from_quaternion(pose.orientation)
        next_yaw = yaw + yaw_rate * dt
        pose.position.x += speed * math.cos(next_yaw) * dt
        pose.position.y += speed * math.sin(next_yaw) * dt

        if self.hold_z:
            if self.fixed_z is None:
                self.fixed_z = pose.position.z
            pose.position.z = self.fixed_z

        pose.orientation = quaternion_from_yaw(next_yaw)

        msg = ModelState()
        msg.model_name = self.model_name
        msg.pose = pose
        msg.reference_frame = self.reference_frame
        msg.twist.linear.x = speed * math.cos(next_yaw)
        msg.twist.linear.y = speed * math.sin(next_yaw)
        msg.twist.angular.z = yaw_rate

        try:
            self.set_model_state(msg)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "set_model_state failed: %s", exc)

    def _alignment_speed_scale(self) -> float:
        if not self.enable_align_hold:
            self.align_hold_active = False
            return 1.0

        heading = abs(self.heading_error_deg) if self.heading_error_deg is not None else None
        lateral = abs(self.lateral_error) if self.lateral_error is not None else None

        if heading is None or lateral is None:
            self.align_hold_active = False
            return 1.0

        inside_release = heading <= self.align_release_heading_deg and lateral <= self.align_release_lateral_error
        outside_hold = heading >= self.align_hold_heading_deg or lateral >= self.align_hold_lateral_error

        if outside_hold:
            self.align_hold_active = True
        elif inside_release:
            self.align_hold_active = False

        if not self.align_hold_active:
            return 1.0

        heading_scale = self._scaled_alignment_component(
            value=heading,
            release_threshold=self.align_release_heading_deg,
            hold_threshold=self.align_hold_heading_deg,
        )
        lateral_scale = self._scaled_alignment_component(
            value=lateral,
            release_threshold=self.align_release_lateral_error,
            hold_threshold=self.align_hold_lateral_error,
        )
        severity = max(heading_scale, lateral_scale)
        return float(max(self.align_min_speed_scale, 1.0 - severity))

    def _scaled_alignment_component(
        self,
        value: float,
        release_threshold: float,
        hold_threshold: float,
    ) -> float:
        if hold_threshold <= release_threshold:
            return 1.0 if value > release_threshold else 0.0
        normalized = (value - release_threshold) / max(hold_threshold - release_threshold, 1e-6)
        return max(0.0, min(1.0, normalized))

    def _startup_boost_scale(self) -> tuple[float, bool]:
        if not self.enable_startup_steer_boost:
            self.startup_boost_armed = True
            return 1.0, False
        if self.heading_error_deg is None or self.lateral_error is None:
            self.startup_boost_active = False
            return 1.0, False

        heading = abs(self.heading_error_deg)
        lateral = abs(self.lateral_error)

        activate = (
            heading >= self.startup_boost_heading_deg
            or lateral >= self.startup_boost_lateral_error
        )
        release = (
            heading <= self.startup_boost_release_heading_deg
            and lateral <= self.startup_boost_release_lateral_error
        )

        if release and self.startup_boost_one_shot:
            self.startup_boost_armed = False
            self.startup_boost_active = False
        elif activate and (self.startup_boost_armed or not self.startup_boost_one_shot):
            self.startup_boost_active = True
        elif release:
            self.startup_boost_active = False

        if not self.startup_boost_active:
            return 1.0, False

        heading_scale = self._scaled_alignment_component(
            value=heading,
            release_threshold=self.startup_boost_release_heading_deg,
            hold_threshold=self.startup_boost_heading_deg,
        )
        lateral_scale = self._scaled_alignment_component(
            value=lateral,
            release_threshold=self.startup_boost_release_lateral_error,
            hold_threshold=self.startup_boost_lateral_error,
        )
        severity = max(heading_scale, lateral_scale)
        scale = 1.0 + (max(self.startup_boost_gain, 1.0) - 1.0) * severity
        return float(scale), True


def main() -> None:
    rospy.init_node("runway_ground_motion_assist")
    RunwayGroundMotionAssist()
    rospy.spin()


if __name__ == "__main__":
    main()
