#!/usr/bin/env python3
from __future__ import annotations

import threading
import time

from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

from runway_inference import RunwayInferenceConfig, RunwayInferenceEngine


class RunwayVisionNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/front_camera/image_raw")
        self.publish_overlay = bool(rospy.get_param("~publish_overlay", True))
        self.publish_mask = bool(rospy.get_param("~publish_mask", True))
        self.max_fps = float(rospy.get_param("~max_fps", 15.0))

        cfg = RunwayInferenceConfig(
            checkpoint_path=rospy.get_param("~checkpoint"),
            config_path=rospy.get_param("~config", None),
            device=rospy.get_param("~device", None),
            threshold=self._optional_float_param("~threshold"),
            mask_cleaning=str(rospy.get_param("~mask_cleaning", "bottom_anchor")),
            mask_anchor_row_ratio=float(rospy.get_param("~mask_anchor_row_ratio", 0.65)),
            mask_bottom_row_ratio=float(rospy.get_param("~mask_bottom_row_ratio", 0.98)),
            temporal_smoothing=bool(rospy.get_param("~temporal_smoothing", True)),
            control_row_ratio=float(rospy.get_param("~control_row_ratio", 0.82)),
            search_radius=int(rospy.get_param("~search_radius", 16)),
            prefer_observed_span_for_lateral=bool(
                rospy.get_param("~prefer_observed_span_for_lateral", True)
            ),
        )
        self.engine = RunwayInferenceEngine(cfg)
        self.last_process_time = 0.0
        self.pending_lock = threading.Lock()
        self.pending_msg: Image | None = None
        self.pending_frame_bgr = None
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.running = True

        self.valid_pub = rospy.Publisher("~interface_valid", Bool, queue_size=1)
        self.heading_pub = rospy.Publisher("~heading_error_deg", Float32, queue_size=1)
        self.lateral_pub = rospy.Publisher("~lateral_error_runway_half_width", Float32, queue_size=1)
        self.confidence_pub = rospy.Publisher("~confidence", Float32, queue_size=1)
        self.lateral_px_pub = rospy.Publisher("~lateral_error_px", Float32, queue_size=1)
        self.width_pub = rospy.Publisher("~runway_width_px", Float32, queue_size=1)
        self.inference_ms_pub = rospy.Publisher("~inference_ms", Float32, queue_size=1)
        self.image_age_ms_pub = rospy.Publisher("~image_age_ms", Float32, queue_size=1)
        self.pipeline_latency_ms_pub = rospy.Publisher("~pipeline_latency_ms", Float32, queue_size=1)
        self.geometry_source_pub = rospy.Publisher("~geometry_source", String, queue_size=1)
        self.lateral_source_pub = rospy.Publisher("~lateral_source", String, queue_size=1)

        self.overlay_pub = None
        self.mask_pub = None
        if self.publish_overlay:
            self.overlay_pub = rospy.Publisher("~overlay_image", Image, queue_size=1)
        if self.publish_mask:
            self.mask_pub = rospy.Publisher("~debug_mask", Image, queue_size=1)

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
            tcp_nodelay=True,
        )
        self.worker.start()
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("runway_vision listening on %s", self.image_topic)

    def image_callback(self, msg: Image) -> None:
        if not self._should_process():
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge conversion failed: %s", exc)
            return

        with self.pending_lock:
            self.pending_msg = msg
            self.pending_frame_bgr = frame_bgr

    def _worker_loop(self) -> None:
        while self.running and not rospy.is_shutdown():
            msg = None
            frame_bgr = None
            with self.pending_lock:
                if self.pending_msg is not None and self.pending_frame_bgr is not None:
                    msg = self.pending_msg
                    frame_bgr = self.pending_frame_bgr
                    self.pending_msg = None
                    self.pending_frame_bgr = None

            if msg is None or frame_bgr is None:
                time.sleep(0.001)
                continue

            callback_start = time.perf_counter()
            image_age_ms = self._message_age_ms(msg)

            try:
                result = self.engine.infer(frame_bgr)
            except Exception as exc:  # pragma: no cover - runtime guard for ROS loops
                rospy.logerr_throttle(2.0, "runway inference failed: %s", exc)
                self.engine.reset_temporal_state()
                continue

            self._publish_control(result["control"])
            self._publish_debug(msg, result)
            self.image_age_ms_pub.publish(Float32(image_age_ms))
            total_latency_ms = image_age_ms + 1000.0 * (time.perf_counter() - callback_start)
            self.pipeline_latency_ms_pub.publish(Float32(float(total_latency_ms)))

    def _shutdown(self) -> None:
        self.running = False

    def _should_process(self) -> bool:
        if self.max_fps <= 0:
            return True
        now = time.time()
        min_dt = 1.0 / self.max_fps
        if now - self.last_process_time < min_dt:
            return False
        self.last_process_time = now
        return True

    def _publish_control(self, control: dict[str, float | bool | None]) -> None:
        self.valid_pub.publish(Bool(bool(control["interface_valid"])))
        self.heading_pub.publish(Float32(self._to_float32(control["heading_error_deg"])))
        self.lateral_pub.publish(Float32(self._to_float32(control["lateral_error_runway_half_width"])))
        self.confidence_pub.publish(Float32(float(control["confidence"])))
        self.lateral_px_pub.publish(Float32(self._to_float32(control["lateral_error_px"])))
        self.width_pub.publish(Float32(self._to_float32(control["runway_width_px"])))
        self.geometry_source_pub.publish(String(str(control.get("geometry_source") or "")))
        self.lateral_source_pub.publish(String(str(control.get("lateral_source") or "")))

    def _publish_debug(self, msg: Image, result: dict[str, object]) -> None:
        timing = result["timing"]
        self.inference_ms_pub.publish(Float32(float(timing["inference_ms"])))

        if self.overlay_pub is not None and result["overlay_bgr"] is not None:
            overlay_msg = self.bridge.cv2_to_imgmsg(result["overlay_bgr"], encoding="bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

        if self.mask_pub is not None and result["mask"] is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(result["mask"], encoding="mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

    def _optional_float_param(self, name: str) -> float | None:
        if not rospy.has_param(name):
            return None
        value = rospy.get_param(name)
        if value in (None, ""):
            return None
        return float(value)

    def _to_float32(self, value: float | None) -> float:
        if value is None:
            return float("nan")
        return float(value)

    def _message_age_ms(self, msg: Image) -> float:
        stamp = msg.header.stamp
        if stamp is None or stamp.to_sec() <= 0.0:
            return float("nan")
        age = rospy.Time.now() - stamp
        return max(0.0, age.to_sec() * 1000.0)


def main() -> None:
    rospy.init_node("runway_vision")
    RunwayVisionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
