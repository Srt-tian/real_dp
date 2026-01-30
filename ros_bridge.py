#!/usr/bin/env python3
import json
import socket
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry 


def now_us() -> int:
    return int(time.time() * 1e6)


def quat_wxyz_to_rotvec(w, x, y, z):
    """Hamilton(wxyz) -> rotvec(3)."""
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q = q / n
    w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    axis = np.array([x, y, z], dtype=np.float64) / s
    return (axis * angle).astype(np.float32)


class RosBridge(Node):
    def __init__(
        self,
        listen_topic="/fam1/fmu/in/vehicle_visual_odometry",
        action_topic="/vla/action_7d",
        udp_state_port=15001,
        udp_action_port=15000,
        udp_ip="127.0.0.1",
    ):
        super().__init__("ros_bridge")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._print_cnt = 0
        # Subscribe VehicleOdometry
        self._latest_rotvec = np.zeros(3, dtype=np.float32)
        self._lock = threading.Lock()
        self.create_subscription(VehicleOdometry, listen_topic, self._odom_cb, qos)

        # Publish action to "upper computer node"
        self.action_pub = self.create_publisher(Float32MultiArray, action_topic, qos)

        # UDP: push state -> infer
        self.udp_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_state_addr = (udp_ip, udp_state_port)

        # UDP: receive action <- infer
        self.udp_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_rx.bind((udp_ip, udp_action_port))
        self.udp_rx.setblocking(False)

        self.get_logger().info(f"Sub  : {listen_topic} (VehicleOdometry)")
        self.get_logger().info(f"Pub  : {action_topic} (Float32MultiArray len=7)")
        self.get_logger().info(f"UDP-> : {udp_ip}:{udp_state_port} (state rotvec)")
        self.get_logger().info(f"UDP<- : {udp_ip}:{udp_action_port} (action 7d)")

        # timers
        self.create_timer(0.02, self._push_state_udp)   # 50Hz state push
        self.create_timer(0.005, self._poll_actions)    # 200Hz poll actions

    def _odom_cb(self, msg: VehicleOdometry):
        # msg.q is (w,x,y,z) per your msg definition
        w, x, y, z = msg.q
        rotvec = quat_wxyz_to_rotvec(w, x, y, z)
        with self._lock:
            self._latest_rotvec = rotvec

        self._print_cnt += 1
        if self._print_cnt % 50 == 0:
            self.get_logger().info(
                f"quat(wxyz) = [{w: .4f}, {x: .4f}, {y: .4f}, {z: .4f}] | "
                f"rotvec = [{rotvec[0]: .3f}, {rotvec[1]: .3f}, {rotvec[2]: .3f}]"
            )
    def _push_state_udp(self):
        with self._lock:
            r = self._latest_rotvec.copy()

        payload = {"type": "state", "t_us": now_us(), "rotvec": r.tolist()}
        try:
            self.udp_tx.sendto(json.dumps(payload).encode("utf-8"), self.udp_state_addr)
        except Exception:
            pass

    def _poll_actions(self):
        # read all queued UDP packets, keep last action
        last = None
        while True:
            try:
                data, _ = self.udp_rx.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                obj = json.loads(data.decode("utf-8"))
                if obj.get("type") == "action":
                    last = obj
            except Exception:
                continue

        if last is None:
            return

        a = last.get("a", None)
        if not (isinstance(a, list) and len(a) == 7):
            self.get_logger().warn("Bad action format")
            return

        msg = Float32MultiArray()
        msg.data = [float(x) for x in a]
        self.action_pub.publish(msg)


def main():
    rclpy.init()
    node = RosBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
