from infer_real_robot_min import load_policy_from_ckpt, build_obs_dict
from diffusion_policy.real_world.real_env import _USBCamera
from px4_msgs.msg import VehicleOdometry
from scan_feetech import control_open_state
from scipy.spatial.transform import Rotation as R
import time
import cv2
import numpy as np
import torch
import json, socket, time
import numpy as np

class UdpClient:
    def __init__(self, udp_ip="127.0.0.1", state_port=15001, action_port=15000):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((udp_ip, state_port))
        self.rx.setblocking(False)

        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.action_addr = (udp_ip, action_port)

        self._latest_rotvec = np.zeros(3, dtype=np.float32)
        self._latest_quat_wxyz = np.array([1,0,0,0], dtype=np.float32)

    def _drain_latest_state(self):
        last = None
        while True:
            try:
                data, _ = self.rx.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                obj = json.loads(data.decode("utf-8"))
                if obj.get("type") == "state":
                    last = obj
            except Exception:
                continue

        if last is None:
            return

        if "rotvec" in last:
            self._latest_rotvec = np.asarray(last["rotvec"], dtype=np.float32).reshape(3)

        if "quat_wxyz" in last:
            q = np.asarray(last["quat_wxyz"], dtype=np.float64).reshape(4)
            self._latest_quat_wxyz = quat_normalize_wxyz(q).astype(np.float32)

        if "pos" in last:
            pos = np.asarray(last["pos"], dtype=np.float32).reshape(3)
            self._latest_pos = pos

    def get_latest_rotvec(self):
        self._drain_latest_state()
        return self._latest_rotvec.copy()

    def get_latest_quat(self):
        self._drain_latest_state()
        return self._latest_quat_wxyz.copy()
    
    def get_latest_pos(self):
        self._drain_latest_state()
        return self._latest_pos.copy()
    
    def send_action(self, action7):
        a = np.asarray(action7, dtype=np.float32).reshape(8).tolist()
        payload = {"type": "action", "t_us": int(time.time() * 1e6), "a": a}
        self.tx.sendto(json.dumps(payload).encode("utf-8"), self.action_addr)

def quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / n
    if q[0] < 0:
        q = -q
    return q

def quat_mul_wxyz(q1, q2):
    # Hamilton product: q = q1 ⊗ q2, both wxyz
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

class UAVEnv:
    def __init__(self):
        self.cam = _USBCamera(
            cam_id=0,
            resolution=(1280, 720),
            fps=30,
        )
        self._last_grip = 0.0  # gripper state
        self.udp_client = UdpClient()

    #TODO
    def get_motion_capture(self):
        print("Getting motion capture data via UDP...")
        return self.udp_client.get_latest_rotvec()
    
    def get_motion_capture_quat(self):
        print("Getting motion capture quaternion via UDP...")
        return self.udp_client.get_latest_quat()
    
    def visualize_obs(self, obs):
        img = obs["camera_0"][-1, :, :, ::-1].copy()
        scale = 4
        vis_big = cv2.resize(
            img,
            (img.shape[1] * scale, img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST  # 保持像素块，不模糊
        )
        cv2.imshow("UAV Camera", vis_big)
        cv2.waitKey(1)

    def get_uav_obs(self):
        # camera
        img = self.cam.read_rgb()  # 原始分辨率
        # resize to 128x128
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        imgs = np.repeat(img[None, ...], 1, axis=0)
        # state
        rot = self.get_motion_capture()
        uav_state = {
            "robot_eef_rot": np.array(rot, dtype=np.float32),
        }   
        gripper_state = {"gripper_open_state": np.array([self._last_grip], dtype=np.float32)}
        obs = {"camera_0": imgs}
        obs.update(uav_state)
        obs.update(gripper_state)
        return obs
    
    # TODO
    def publish_action(self, action):
        self.udp_client.send_action(action)
        print(f"Published action: {action}")






def main(ckpt, device,k):
    policy = load_policy_from_ckpt(ckpt, device=device)
    env = UAVEnv()
    while True:
        # Get observation
        obs = env.get_uav_obs()

        # visualize
        env.visualize_obs(obs)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        # build obs_dict
        obs_dict = build_obs_dict(
            obs,
            device=device,
            image_key="camera_0",
            rot_key="robot_eef_rot",
            grip_key="gripper_open_state",
        )

        # get action
        with torch.no_grad():
            out = policy.predict_action(obs_dict)
            action_all = out["action"][0].detach().cpu().numpy()  # (H,7)
        action_seq = action_all[:k].astype(np.float64)  # (k,7)
        action_seq_list = action_seq.tolist()

        # publish action
        dt = 0.2
        for action in action_seq_list:
            action = np.array(action, dtype=np.float64)
            p8 = np.zeros(8, dtype=np.float64)

            # dx,dy,dz = action[:3]
            dx,dy,dz = [0.,0.,0.]  # 位置增量由视觉伺服控制
            pos_cur = env.udp_client.get_latest_pos()
            # import pdb; pdb.set_trace()
            pos_still = np.array([ 2.7629836 , -0.61798394,  0.980616  ], dtype=np.float32)
            # import pdb; pdb.set_trace()
            # p8[0:3] = pos_cur + np.array([dx,dy,dz], dtype=np.float64)
            p8[0:3] = pos_still + np.array([dx,dy,dz], dtype=np.float64)
            drot = action[3:6]

            # 每个增量都是以当前最新姿态为基础进行增量旋转计算，而不是以推理时间点的姿态为基础（需要验证是否可行）
            # 1) 当前绝对姿态（wxyz）
            q_cur = quat_normalize_wxyz(env.udp_client.get_latest_quat())
            # import pdb; pdb.set_trace()
            q_still = np.array([0.99991416, 0.00665369, 0.00167531, 0.01116195],dtype=np.float64)
            # 2) 增量旋转 dq（wxyz）
            dq_xyzw = R.from_rotvec(drot).as_quat()  # xyzw
            dq_wxyz = np.array([dq_xyzw[3], dq_xyzw[0], dq_xyzw[1], dq_xyzw[2]], dtype=np.float64)
            dq_wxyz = quat_normalize_wxyz(dq_wxyz)

            # 3) 合成绝对四元数：q_abs = dq ⊗ q_cur
            q_abs = quat_mul_wxyz(dq_wxyz, q_cur)
            q_abs = quat_normalize_wxyz(q_abs)

            # p8[3:7] = q_abs.astype(np.float64)
            p8[3:7] = q_still.astype(np.float64)
            p8[7] = action[6]
            t_cycle_end = time.time() + dt
            env.publish_action(p8)
            g = 1 if action[6] > 0.5 else 0
            if g != env._last_grip:
                # control_open_state(g)
                env._last_grip = g
            sleep_time = t_cycle_end - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    ckpt = "/home/tian/diffusion_policy/data/outputs_delta/2026.01.24/19.14.34_train_diffusion_unet_real_h5_image_real_h5_image/checkpoints/epoch=0100-train_loss=0.023.ckpt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(ckpt, device, k=8)