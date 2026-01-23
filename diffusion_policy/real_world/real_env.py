from typing import Optional, Dict, Any, List
import pathlib
import time
import numpy as np
import cv2
import h5py
from multiprocessing.managers import SharedMemoryManager

from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from scan_feetech import control_open_state

DEFAULT_OBS_KEY_MAP = {
    # robot
    "ActualTCPPose": "robot_eef_pose",
}


class _USBCamera:
    """Minimal USB camera wrapper (cv2.VideoCapture). Returns RGB uint8."""
    def __init__(self, cam_id: int, resolution=(640, 480), fps: int = 30):
        self.cam_id = int(cam_id)
        self.resolution = (int(resolution[0]), int(resolution[1]))  # (W,H)
        self.fps = int(fps)

        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera id={self.cam_id}")

        w, h = self.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            print("[WARN] Failed to set CAP_PROP_BUFFERSIZE")

    def read_rgb(self) -> np.ndarray:
        # 尽量清空缓冲：限时 + 限次，避免极端情况卡住
        t0 = time.time()
        n = 0
        while (time.time() - t0) < 0.01:
            self.cap.grab()
            n += 1

        ok, frame_bgr = self.cap.retrieve()

        # 兜底：retrieve失败时，尝试直接read一次
        if (not ok) or (frame_bgr is None):
            ok, frame_bgr = self.cap.read()

        w, h = self.resolution
        if (not ok) or (frame_bgr is None):
            return np.zeros((h, w, 3), dtype=np.uint8)

        if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

        return frame_bgr[..., ::-1].copy()

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


class RealEnv:
    """
    Minimal RealEnv for:
      - UR robot teleop via RTDEInterpolationController
      - USB camera (no RealSense)
      - dataset recording to HDF5 per episode

    Interface kept to satisfy demo_real_robot.py usage:
      - start(), stop()
      - get_robot_state()
      - get_obs()
      - exec_actions(actions, timestamps)
      - start_episode(start_time), end_episode()
      - context manager (__enter__/__exit__)
    """

    def __init__(
        self,
        output_dir,
        robot_ip,
        # env params
        frequency=10,
        n_obs_steps=1,
        # obs
        obs_image_resolution=(640, 480),  # (W,H)
        obs_key_map=DEFAULT_OBS_KEY_MAP,
        obs_float32=False,
        # video flags (kept for signature compatibility; not used in minimal)
        record_raw_video=True,
        # shared memory (kept; not required here)
        shm_manager: Optional[SharedMemoryManager] = None,
        # usb cam
        usb_cam_id: int = 2,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.robot_ip = robot_ip

        self.frequency = int(frequency)
        self.dt = 1.0 / max(self.frequency, 1)
        self.n_obs_steps = int(n_obs_steps)

        self.obs_image_resolution = (int(obs_image_resolution[0]), int(obs_image_resolution[1]))  # (W,H)
        self.obs_key_map = dict(obs_key_map)
        self.obs_float32 = bool(obs_float32)

        self.usb_cam_id = int(usb_cam_id)

        # episode bookkeeping
        self._episode_idx = 0
        self._recording = False
        self._episode_start_time = None
        self._episode_dir = None

        # buffers
        self._buf_images: List[np.ndarray] = []
        self._buf_robot: List[Dict[str, Any]] = []
        self._buf_actions: List[np.ndarray] = []
        self._buf_timestamps: List[float] = []
        self._buf_robot_eef_rot = []
        self._buf_open_state = []
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        self.shm_manager = shm_manager
        # init robot + camera
        self.robot = RTDEInterpolationController(
            robot_ip=self.robot_ip,
            shm_manager=self.shm_manager,
        )
        self.camera = _USBCamera(
            cam_id=self.usb_cam_id,
            resolution=self.obs_image_resolution,
            fps=30,
        )
        self._last_grip = 0

    # ---------------- lifecycle ----------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self):
        # start robot controller
        self.robot.start(wait=False)
        if hasattr(self.robot, "start_wait"):
            self.robot.start_wait()
        time.sleep(0.2)

    def stop(self):
        # stop recording if needed
        if self._recording:
            try:
                self.end_episode()
            except Exception:
                pass

        # stop robot
        try:
            self.robot.stop(wait=False)
            if hasattr(self.robot, "stop_wait"):
                self.robot.stop_wait()
        except Exception:
            pass

        # close camera
        try:
            self.camera.close()
        except Exception:
            pass

    # ---------------- robot / obs ----------------
    def get_robot_state(self) -> Dict[str, Any]:
        # Must at least provide TargetTCPPose for demo_real_robot.py
        return self.robot.get_state()

    def _map_robot_obs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for src_k, dst_k in self.obs_key_map.items():
            if src_k not in state:
                continue

            v = np.asarray(state[src_k])

            # controller 历史窗口 (N,6) → 取最后一帧
            if v.ndim == 2:
                v = v[-1]

            # 单帧 (6,) → 直接用
            elif v.ndim == 1:
                pass

            else:
                raise ValueError(
                    f"{src_k} has unexpected shape {v.shape}"
                )

            # 真正的语义约束
            assert v.shape == (6,), f"{dst_k} shape {v.shape}, expected (6,)"
            out[dst_k] = v.astype(np.float32 if self.obs_float32 else None)
            if "robot_eef_pose" in out:
                pose6 = out["robot_eef_pose"]  # (6,)
                out["robot_eef_rot"] = pose6[3:6].astype(np.float32, copy=True)

        return out
    # def get_vis_obs(self) -> Dict[str, Any]:
    #     """
    #     Return:
    #       - camera_0: (H, W, 3) uint8 RGB
    #     """
    #     # camera
    #     img = self.camera.read_rgb()  # 原始分辨率
    #     raw_img = img.copy()
    #     raw_imgs = raw_img[None, ...]  # (1, H, W, 3)
    #     raw_imgs = {"raw_camera_0": raw_imgs}
    #     return raw_imgs
    
    def get_obs(self) -> Dict[str, Any]:
        """
        Return:
          - camera_0: (n_obs_steps, H, W, 3) uint8 RGB
          - robot_* mapped keys (float32)
        """
        # camera
        img = self.camera.read_rgb()  # 原始分辨率
        img_raw = img.copy()
        # resize to 84x84 (W,H) -> (84,84)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)

        imgs = np.repeat(img[None, ...], self.n_obs_steps, axis=0)
        imgs_raw = np.repeat(img_raw[None, ...], self.n_obs_steps, axis=0)
        # robot
        state_all = self.robot.get_all_state() if hasattr(self.robot, "get_all_state") else self.robot.get_state()
        robot_obs = self._map_robot_obs(state_all)
        gripper_state = {"gripper_open_state": np.array([self._last_grip], dtype=np.float32)}
        obs = {"camera_0": imgs}
        obs.update(robot_obs)
        obs.update(gripper_state)
        obs.update({"raw_camera_0": imgs_raw})
        return obs

    # ---------------- control + recording ----------------
    def exec_actions(self, obs, actions: List[np.ndarray], timestamps: List[float]):
        """
        actions: list of target poses (e.g., [x,y,z,rx,ry,rz] or [x,y,z,rotvec])
        timestamps: list of wall-clock unix timestamps (time.time()) when to execute
        """

        for a, ts in zip(actions, timestamps):
            pose = np.asarray(a, dtype=np.float64).copy()
            self.robot.schedule_waypoint(pose[0:6], ts)
            g = 1 if pose[6] > 0.5 else 0
            if g != self._last_grip:
                control_open_state(g)   # 例如：1=close,0=open（按你的约定）
                print(f"[GRIP] set to {g} at {ts:.3f}")
                self._last_grip = g
            if self._recording:

                # store last frame only for dataset (T,H,W,3)
                self._buf_images.append(obs["camera_0"][-1].astype(np.uint8))
                self._buf_robot_eef_rot.append(np.asarray(obs["robot_eef_pose"], dtype=np.float32)[3:])
                self._buf_open_state.append(float(self._last_grip))
                # record robot state (use raw keys for maximal info)
                state_all = self.robot.get_all_state() if hasattr(self.robot, "get_all_state") else self.robot.get_state()
                self._buf_robot.append(state_all)

                self._buf_actions.append(pose.astype(np.float32))
                self._buf_timestamps.append(float(ts))

    # ---------------- episode I/O ----------------
    def start_episode(self, start_time: Optional[float] = None):
        if start_time is None:
            start_time = time.time()
        self._episode_start_time = float(start_time)
        self._recording = True

        # clear buffers
        self._buf_images.clear()
        self._buf_robot.clear()
        self._buf_actions.clear()
        self._buf_timestamps.clear()
        self._buf_robot_eef_rot.clear()
        self._buf_open_state.clear()
        # create episode dir
        self._episode_dir = self.output_dir / f"episode_{self._episode_idx:04d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        # return first obs (handy)
        return self.get_obs()

    def end_episode(self):
        if not self._recording:
            return

        self._recording = False

        T = len(self._buf_images)
        if T == 0:
            # still bump idx; create an empty marker file
            (self._episode_dir / "EMPTY").write_text("no frames recorded\n", encoding="utf-8")
            self._episode_idx += 1
            return

        # stack arrays
        images = np.stack(self._buf_images, axis=0)  # (T,H,W,3) RGB uint8
        actions = np.stack(self._buf_actions, axis=0)  # (T, A)
        timestamps = np.asarray(self._buf_timestamps, dtype=np.float64)
        eef = np.stack(self._buf_robot_eef_rot, axis=0)  # (T,3)
        gripper_open_state = np.asarray(self._buf_open_state, dtype=np.float32)[:, None]  # (T,1)

        # align checks (avoid silent mismatch / crash later)
        assert len(self._buf_actions) == T, f"actions len {len(self._buf_actions)} != T {T}"
        assert len(self._buf_timestamps) == T, f"timestamps len {len(self._buf_timestamps)} != T {T}"
        assert len(self._buf_robot_eef_rot) == T, f"eef len {len(self._buf_robot_eef_rot)} != T {T}"
        assert len(self._buf_open_state) == T, f"open_state len {len(self._buf_open_state)} != T {T}"


        # save HDF5
        h5_path = self._episode_dir / "episode.hdf5"
        with h5py.File(h5_path, "w") as f:
            g_obs = f.create_group("obs")
            g_act = f.create_group("action")

            g_obs.create_dataset("camera_0", data=images, dtype=np.uint8, compression="gzip", compression_opts=4)
            g_obs.create_dataset("robot_eef_rot", data=eef, dtype=np.float32)
            g_obs.create_dataset("gripper_open_state",data=gripper_open_state,dtype=np.float32)
            g_act.create_dataset("target_pose", data=actions, dtype=np.float32)
            f.create_dataset("timestamp", data=timestamps, dtype=np.float64)

            # # store robot raw dict as a best-effort (numbers only)
            # g_robot = g_obs.create_group("robot_raw")
            # # collect numeric keys
            # sample = self._buf_robot[0]
            # for k in sample.keys():
            #     try:
            #         arr = np.stack([np.asarray(x.get(k)) for x in self._buf_robot], axis=0)
            #         if np.issubdtype(arr.dtype, np.number):
            #             g_robot.create_dataset(k, data=arr.astype(np.float32))
            #     except Exception:
            #         # skip non-stackable / non-numeric
            #         pass

            # meta
            f.attrs["episode_start_time"] = self._episode_start_time
            f.attrs["frequency"] = self.frequency
            f.attrs["obs_image_resolution_w"] = self.obs_image_resolution[0]
            f.attrs["obs_image_resolution_h"] = self.obs_image_resolution[1]
            f.attrs["usb_cam_id"] = self.usb_cam_id

        self._episode_idx += 1
    def discard_episode(self, reason: str = "user_discard"):
        """Discard current episode without saving HDF5."""
        if not self._recording:
            return
        self._recording = False

        # clear buffers
        self._buf_images.clear()
        self._buf_robot.clear()
        self._buf_actions.clear()
        self._buf_timestamps.clear()

        # mark on disk (optional but useful for debugging)
        try:
            if self._episode_dir is not None:
                (self._episode_dir / "DISCARDED").write_text(f"{reason}\n", encoding="utf-8")
        except Exception:
            pass

        self._episode_idx += 1