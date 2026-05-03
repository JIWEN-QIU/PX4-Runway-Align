#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rospy
from mavros_msgs.msg import ManualControl, OverrideRCIn
from std_msgs.msg import Bool, Float32, Int16, UInt16


class RunwayRCOverrideBridge:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/runway_ground_align_controller/steer_cmd")
        self.mode = str(rospy.get_param("~mode", "manual_control"))
        self.manual_control_topic = rospy.get_param("~manual_control_topic", "/mavros/manual_control/send")
        self.rc_override_topic = rospy.get_param("~rc_override_topic", "/mavros/rc/override")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.command_timeout_sec = float(rospy.get_param("~command_timeout_sec", 0.5))
        self.release_on_timeout = bool(rospy.get_param("~release_on_timeout", True))

        # MAVROS OverrideRCIn channels are 1-based in user-facing params.
        self.steer_channel = int(rospy.get_param("~steer_channel", 1))
        self.reverse = bool(rospy.get_param("~reverse", False))
        self.pwm_center = int(rospy.get_param("~pwm_center", 1500))
        self.pwm_range = int(rospy.get_param("~pwm_range", 400))
        self.pwm_min = int(rospy.get_param("~pwm_min", 1100))
        self.pwm_max = int(rospy.get_param("~pwm_max", 1900))
        self.channel_count = int(rospy.get_param("~channel_count", 18))

        self.manual_axis = str(rospy.get_param("~manual_axis", "r"))
        self.manual_scale = float(rospy.get_param("~manual_scale", 1000.0))
        self.manual_x = float(rospy.get_param("~manual_x", 0.0))
        self.manual_y = float(rospy.get_param("~manual_y", 0.0))
        self.manual_z = float(rospy.get_param("~manual_z", 0.0))
        self.manual_r = float(rospy.get_param("~manual_r", 0.0))
        self.manual_buttons = int(rospy.get_param("~manual_buttons", 0))

        if self.mode not in {"manual_control", "rc_override"}:
            raise ValueError(f"mode must be 'manual_control' or 'rc_override', got {self.mode!r}")
        if self.mode == "rc_override" and not (1 <= self.steer_channel <= self.channel_count):
            raise ValueError(f"steer_channel must be in [1, {self.channel_count}], got {self.steer_channel}")
        if self.manual_axis not in {"x", "y", "z", "r"}:
            raise ValueError(f"manual_axis must be one of x/y/z/r, got {self.manual_axis!r}")
        self.output_topic = self.manual_control_topic if self.mode == "manual_control" else self.rc_override_topic

        self.last_cmd = 0.0
        self.last_cmd_time: float | None = None
        self.active = False

        rospy.Subscriber(self.input_topic, Float32, self.cmd_cb, queue_size=1)
        if self.mode == "manual_control":
            self.command_pub = rospy.Publisher(self.output_topic, ManualControl, queue_size=1)
        else:
            self.command_pub = rospy.Publisher(self.output_topic, OverrideRCIn, queue_size=1)
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.pwm_pub = rospy.Publisher("~steer_pwm", UInt16, queue_size=1)
        self.manual_pub = rospy.Publisher("~manual_steer", Int16, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.rate_hz, 1e-3)), self.timer_cb)
        rospy.loginfo(
            "runway_rc_override_bridge running: mode=%s, %s -> %s, steer_channel=%d, manual_axis=%s, reverse=%s",
            self.mode,
            self.input_topic,
            self.output_topic,
            self.steer_channel,
            self.manual_axis,
            self.reverse,
        )

    def cmd_cb(self, msg: Float32) -> None:
        if math.isnan(msg.data):
            return
        self.last_cmd = float(max(-1.0, min(1.0, msg.data)))
        self.last_cmd_time = time.time()

    def timer_cb(self, _event: rospy.timer.TimerEvent) -> None:
        timed_out = self.last_cmd_time is None or (time.time() - self.last_cmd_time) > self.command_timeout_sec
        if timed_out and self.release_on_timeout:
            self.publish_release()
            self.active = False
            self.active_pub.publish(Bool(False))
            return

        cmd = 0.0 if timed_out else self.last_cmd
        if self.mode == "manual_control":
            manual_steer = self.command_to_manual(cmd)
            msg = self.build_manual_control_msg(manual_steer)
            self.command_pub.publish(msg)
            self.manual_pub.publish(Int16(manual_steer))
            self.active = not timed_out
            self.active_pub.publish(Bool(self.active))
            return

        pwm = self.command_to_pwm(cmd)
        msg = self.build_override_msg(pwm)
        self.command_pub.publish(msg)
        self.pwm_pub.publish(UInt16(pwm))
        self.active = not timed_out
        self.active_pub.publish(Bool(self.active))

    def command_to_pwm(self, cmd: float) -> int:
        signed_cmd = -cmd if self.reverse else cmd
        pwm = int(round(self.pwm_center + signed_cmd * self.pwm_range))
        return int(max(self.pwm_min, min(self.pwm_max, pwm)))

    def command_to_manual(self, cmd: float) -> int:
        signed_cmd = -cmd if self.reverse else cmd
        value = int(round(signed_cmd * self.manual_scale))
        return int(max(-1000, min(1000, value)))

    def build_manual_control_msg(self, steer_value: int) -> ManualControl:
        msg = ManualControl()
        msg.header.stamp = rospy.Time.now()
        msg.x = float(self.manual_x)
        msg.y = float(self.manual_y)
        msg.z = float(self.manual_z)
        msg.r = float(self.manual_r)
        setattr(msg, self.manual_axis, float(steer_value))
        msg.buttons = int(self.manual_buttons)
        return msg

    def build_override_msg(self, steer_pwm: int) -> OverrideRCIn:
        msg = OverrideRCIn()
        msg.channels = [OverrideRCIn.CHAN_NOCHANGE] * self.channel_count
        msg.channels[self.steer_channel - 1] = int(steer_pwm)
        return msg

    def publish_release(self) -> None:
        if self.mode == "manual_control":
            msg = self.build_manual_control_msg(0)
            self.command_pub.publish(msg)
            self.manual_pub.publish(Int16(0))
            return

        msg = OverrideRCIn()
        msg.channels = [OverrideRCIn.CHAN_RELEASE] * self.channel_count
        self.command_pub.publish(msg)


def main() -> None:
    rospy.init_node("runway_rc_override_bridge")
    RunwayRCOverrideBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
