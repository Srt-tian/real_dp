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

    def get_latest_rotvec(self):
        # drain buffer, keep newest
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
        if last is not None:
            rv = np.asarray(last["rotvec"], dtype=np.float32).reshape(3)
            self._latest_rotvec = rv
        return self._latest_rotvec.copy()

    def send_action(self, action7):
        a = np.asarray(action7, dtype=np.float32).reshape(7).tolist()
        payload = {"type": "action", "t_us": int(time.time() * 1e6), "a": a}
        self.tx.sendto(json.dumps(payload).encode("utf-8"), self.action_addr)

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
            p8[:3] = action[:3]
            drot = action[3:6]
            dq_xyzw = R.from_rotvec(drot).as_quat() 
            dq_xyzw = dq_xyzw / (np.linalg.norm(dq_xyzw) + 1e-12)
            dq_wxyz = np.array([dq_xyzw[3], dq_xyzw[0], dq_xyzw[1], dq_xyzw[2]])
            p8[3:7] = dq_wxyz
            p8[7] = action[6]
            t_cycle_end = time.time() + dt
            env.publish_action(p8)
            g = 1 if action[6] > 0.5 else 0
            if g != env._last_grip:
                control_open_state(g)
                env._last_grip = g
            sleep_time = t_cycle_end - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    ckpt = "/home/tian/diffusion_policy/data/outputs_delta/2026.01.24/19.14.34_train_diffusion_unet_real_h5_image_real_h5_image/checkpoints/epoch=0100-train_loss=0.023.ckpt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(ckpt, device, k=8)