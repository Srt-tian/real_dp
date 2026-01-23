import time
import click
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
import hydra
from omegaconf import OmegaConf
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)

def last_frame(x):
    """
    兼容两类输入：
      - (T, D) / (n_obs_steps, D): 取最后一帧
      - (D,): 直接返回
    """
    x = np.asarray(x)
    return x[-1] if x.ndim >= 2 else x

def build_obs_dict(
    obs,
    device,
    image_key="camera_0",
    rot_key="robot_eef_rot",
    grip_key="gripper_open_state",
    img_hw=(84, 84),
):
    """
    输出匹配你训练时的 shape_meta:
      obs.image: (B, To, C, H, W) = (1,1,3,84,84), float32 [0,1]
      obs.state: (B, To, D)       = (1,1,4), float32
    """
    # image: HWC RGB uint8 -> CHW float32 [0,1]
    img = obs[image_key][-1]  # HWC RGB
    if (img.shape[0], img.shape[1]) != img_hw:
        img = cv2.resize(img, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
    img = (img - mean) / std
    # state: rot(3) + gripper(1)
    rot3 = np.asarray(last_frame(obs[rot_key]), dtype=np.float32).reshape(3)
    grip = np.asarray(last_frame(obs[grip_key]), dtype=np.float32).reshape(-1)
    grip = np.array([float(grip[0]) if grip.size > 0 else 0.0], dtype=np.float32)
    state = np.concatenate([rot3, grip], axis=0).astype(np.float32)  # (4,)

    obs_dict = {
        "image": to_torch(img[None, None, ...], device),    # (1,1,3,H,W)
        "state": to_torch(state[None, None, ...], device),  # (1,1,4)
    }
    return obs_dict


def load_policy_from_ckpt(ckpt_path, device="cuda:0"):
    """
    兼容你当前 ckpt 格式:
      payload = {
        'cfg': <OmegaConf dict>,
        'state_dicts': {'model':..., 'ema_model':..., 'optimizer':...},
        'pickles': ...
      }
    """
    payload = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(payload, dict), "unexpected ckpt type"
    assert "cfg" in payload and "state_dicts" in payload, "ckpt missing cfg/state_dicts"

    cfg = payload["cfg"]
    state_dicts = payload["state_dicts"]

    # 1) instantiate policy / model
    # Diffusion Policy 的 workspace 通常是 cfg.policy 或 cfg.model / cfg.workspace.policy
    policy_cfg = None
    for path in ["policy", "workspace.policy", "model", "workspace.model"]:
        cand = OmegaConf.select(cfg, path)
        if isinstance(cand, dict) or (cand is not None and "_target_" in cand):
            if cand is not None and "_target_" in cand:
                policy_cfg = cand
                break

    if policy_cfg is None:
        # 兜底：很多仓库里 policy 是 workspace 内部通过 cfg.policy 生成；
        # 如果你这里没找到，说明你的 cfg 结构稍不同，把 list(cfg.keys()) 打印出来我再对齐。
        raise RuntimeError(
            "在 ckpt['cfg'] 里找不到 policy 的 _target_（尝试 policy/workspace.policy/model/workspace.model）。"
        )

    policy = hydra.utils.instantiate(policy_cfg)

    # 2) load weights (EMA 优先)
    sd = None
    if isinstance(state_dicts, dict) and "ema_model" in state_dicts:
        sd = state_dicts["ema_model"]
        print("[LOAD] using state_dicts['ema_model']")
    elif isinstance(state_dicts, dict) and "model" in state_dicts:
        sd = state_dicts["model"]
        print("[LOAD] using state_dicts['model']")
    else:
        raise RuntimeError("ckpt['state_dicts'] 里没有 'ema_model' 或 'model'")

    # 3) 剥可能的前缀（常见：'policy.' 'model.' 'module.'）
    def strip_prefix(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    policy_sd_keys = set(policy.state_dict().keys())
    if set(sd.keys()) != policy_sd_keys:
        for pref in ["policy.", "model.", "ema_model.", "module."]:
            stripped = strip_prefix(sd, pref)
            if stripped and set(stripped.keys()).issubset(policy_sd_keys):
                sd = stripped
                print(f"[LOAD] stripped prefix: {pref}")
                break

    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[LOAD] missing examples:", missing[:10])
    if len(unexpected) > 0:
        print("[LOAD] unexpected examples:", unexpected[:10])

    policy.to(device)
    policy.eval()
    return policy


def action_seq_to_target_poses(
    action_seq,               # (k,7)
    action_mode,              # "delta" or "absolute"
    base_target_pose6,        # (6,) absolute rotvec pose: [x,y,z,rx,ry,rz]
):
    """
    统一输出要发送给 exec_actions 的 pose7 列表：[[x,y,z,rx,ry,rz,grip], ...]
    - action_mode=absolute: action_seq[i,:6] 当作绝对 pose6
    - action_mode=delta:    action_seq[i,:6] 当作 delta，累加到 base_target_pose6 上
    """
    k = action_seq.shape[0]
    out = []

    if action_mode == "absolute":
        for i in range(k):
            p7 = np.zeros(7, dtype=np.float64)
            p7[:6] = action_seq[i, :6].astype(np.float64)
            g = float(action_seq[i, 6])
            g = 1.0 if g > 0.5 else 0.0
            p7[6] = g
            out.append(p7)
        return out

    if action_mode == "delta":
        pose6 = base_target_pose6.astype(np.float64).copy()
        R_abs = R.from_rotvec(pose6[3:6])

        for i in range(k):
            d = action_seq[i].astype(np.float64)

            # Δpos
            pose6[:3] += d[:3]

            # Δrot (SO(3), left-multiply)
            dR = R.from_rotvec(d[3:6])
            R_abs = dR * R_abs
            pose6[3:6] = R_abs.as_rotvec()

            # grip (binarize)
            g = float(d[6])
            g = 1.0 if g > 0.5 else 0.0

            p7 = np.zeros(7, dtype=np.float64)
            p7[:6] = pose6
            p7[6] = g
            out.append(p7)

        return out

    raise ValueError(f"Unknown action_mode: {action_mode}")


@click.command()
@click.option("--ckpt", required=True, type=str, help="Path to workspace checkpoint containing workspace_state/policy.")
@click.option("--robot_ip", required=True, type=str)
@click.option("--usb_cam_id", default=2, type=int)
@click.option("--frequency", default=5.0, type=float)
@click.option("--steps_per_infer", default=1, type=int, help="每次推理执行前 k 步（闭环更稳：1~4）")
@click.option("--action_mode", type=click.Choice(["delta", "absolute"]), default="delta",
              help="delta=累加Δpose后发绝对目标；absolute=直接把action当绝对pose发送")
@click.option("--img_h", default=84, type=int)
@click.option("--img_w", default=84, type=int)
@click.option("--device", default="cuda:0", type=str)
def main(ckpt, robot_ip, usb_cam_id, frequency, steps_per_infer, action_mode, img_h, img_w, device):
    dt = 1.0 / float(frequency)
    policy = load_policy_from_ckpt(ckpt, device=device)

    with RealEnv(
        output_dir=".",                  # 推理不写数据，给占位
        robot_ip=robot_ip,
        obs_image_resolution=(1280, 720),
        frequency=frequency,
        n_obs_steps=1,
        usb_cam_id=usb_cam_id,
    ) as env:

        cv2.setNumThreads(1)
        time.sleep(0.5)
        print(f"[INFO] action_mode={action_mode}, dt={dt:.3f}s, steps_per_infer={steps_per_infer}")
        print("Press 'q' to quit.")

        t_start = time.monotonic()
        it = 0
        k = steps_per_infer
        while True:
            chunk_dt = 20 * dt
            t_cycle_end = t_start + (it + 1) * chunk_dt
            # 1) get obs (如果偶发空序列，简单跳过)
            obs = env.get_obs()
            if "raw_camera_0" not in obs or obs["raw_camera_0"].shape[0] == 0:
                precise_wait(t_cycle_end)
                it += 1
                continue

            # 8) visualize (cv2 expects BGR)
            vis = obs["raw_camera_0"][-1, :, :, ::-1].copy()
            cv2.putText(
                vis,
                f"mode={action_mode} k={k}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.imshow("infer", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # 2) build obs_dict
            obs_dict = build_obs_dict(
                obs,
                device=device,
                image_key="camera_0",
                rot_key="robot_eef_rot",
                grip_key="gripper_open_state",
                img_hw=(img_h, img_w),
            )

            # 自检
            # print("[PRE] obs_dict keys:", obs_dict.keys())
            # for k, v in obs_dict.items():
            #     if hasattr(v, "shape"):
            #         print(f"[PRE] {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            # import pdb; pdb.set_trace()
            # 3) policy inference -> (H,7)
            with torch.no_grad():
                out = policy.predict_action(obs_dict)
                action_all = out["action"][0].detach().cpu().numpy()  # (H,7)
            assert action_all.shape[-1] == 7, f"expected action_dim=7, got {action_all.shape}"

            # k = int(max(1, min(steps_per_infer, action_all.shape[0])))
            action_seq = action_all[:k].astype(np.float64)  # (k,7)

            # 4) base pose from robot current target
            state = env.get_robot_state()
            base_pose6 = np.array(state["TargetTCPPose"], dtype=np.float64).reshape(6)

            # 5) convert actions -> pose7 list to send
            pose_list = action_seq_to_target_poses(
                action_seq=action_seq,
                action_mode=action_mode,
                base_target_pose6=base_pose6,
            )

            # 6) timestamps with lead (避免落到过去)
            now_wall = time.time()
            lead = max(0.05, 2*dt)
            t0 = now_wall + lead
            ts = (t0 + np.arange(k) * dt).tolist()

            now2 = time.time()
            min_gap = max(0.02, 0.5 * dt)   # 最小安全间隔（20ms 或 半个控制周期）
            # 增强鲁棒性
            if ts[0] <= now2 + min_gap:
                # 把整串 timestamps 平移到“安全的未来”
                shift = (now2 + min_gap) - ts[0]
                ts = [t + shift for t in ts]
                print(f"[TS FIX] shifted by {shift*1000:.1f} ms")

            # 7) send
            env.exec_actions(
                obs=obs,  # 你的 RealEnv.exec_actions 需要 obs 参数（来自 demo_real_robot.py）
                actions=[p.copy() for p in pose_list],
                timestamps=ts,
            )


            precise_wait(t_cycle_end)
            it += 1



if __name__ == "__main__":
    main()
