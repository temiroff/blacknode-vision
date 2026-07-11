from __future__ import annotations

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class UsbCamera(Node):
    def __init__(self) -> None:
        super().__init__("blacknode_usb_camera")
        self.declare_parameter("device", 0)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("hz", 30.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("rotation", 0)
        self.declare_parameter("backend", "v4l2")
        self.declare_parameter("warmup_frames", 5)

        device = self.get_parameter("device").value
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        backend = str(self.get_parameter("backend").value).strip().lower()
        capture_backend = cv2.CAP_V4L2 if backend in {"", "v4l2"} else cv2.CAP_ANY

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.rotation = int(self.get_parameter("rotation").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        hz = max(0.1, float(self.get_parameter("hz").value))
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)

        self.cap = cv2.VideoCapture(device, capture_backend)
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device!r}")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.image_topic, 10)
        self.last_warn = 0.0
        self._warmup(max(0, int(self.get_parameter("warmup_frames").value)))
        self.create_timer(1.0 / hz, self.tick)
        self.get_logger().info(
            f"publishing {device!r} to {self.image_topic} at {hz:g} Hz "
            f"({width}x{height}, rotation={self.rotation})"
        )

    def _warmup(self, frames: int) -> None:
        for _ in range(frames):
            self.cap.read()

    def tick(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            now = time.monotonic()
            if now - self.last_warn > 2.0:
                self.get_logger().warn("camera frame read failed")
                self.last_warn = now
            return
        frame = rotate_frame(frame, self.rotation)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self) -> bool:
        self.cap.release()
        return super().destroy_node()


def rotate_frame(frame, rotation: int):
    rotation = rotation % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def main() -> None:
    rclpy.init()
    node = UsbCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
