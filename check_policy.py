import argparse
import numpy as np
import torch
import h5py
import cv2
from scipy.spatial.transform import Rotation as R
import hydra
from omegaconf import OmegaConf
import time
from torch.cuda.amp import autocast

def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)


def last_frame(x):
    x = np.asarray(x)
    return x[-1] if x.ndim >= 2 else x

def gpu_sanity_check(policy, obs_dict):
    print("[CUDA] available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[CUDA] device:", torch.cuda.get_device_name(0))
        print("[CUDA] policy param device:", next(policy.parameters()).device)
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"[CUDA] obs[{k}] device:", v.device, "dtype:", v.dtype, "shape:", tuple(v.shape))

def find_step_fields(obj, max_depth=2, prefix="policy"):
    hits = []
    for name in dir(obj):
        if "step" in name.lower() or "timestep" in name.lower():
            try:
                val = getattr(obj, name)
                if isinstance(val, (int, float, str, list, tuple)):
                    hits.append((f"{prefix}.{name}", val))
            except Exception:
                pass
    # 常见：policy 里还有 scheduler / noise_scheduler
    for sub_name in ["scheduler", "noise_scheduler", "diffusion", "model"]:
        if hasattr(obj, sub_name):
            sub = getattr(obj, sub_name)
            for name in dir(sub):
                if "step" in name.lower() or "timestep" in name.lower():
                    try:
                        val = getattr(sub, name)
                        if isinstance(val, (int, float, str, list, tuple)):
                            hits.append((f"{prefix}.{sub_name}.{name}", val))
                    except Exception:
                        pass
    return hits

def find_sampler_fields(obj):
    keys = []
    for name in dir(obj):
        if any(s in name.lower() for s in ["ddim", "dpm", "solver", "scheduler"]):
            keys.append(name)
    return keys

def timed_predict(policy, obs_dict, iters=10, warmup=3):
    # 只测 predict_action（不包含 .cpu().numpy()），并做同步，得到“真实GPU耗时”
    assert torch.cuda.is_available(), "CUDA not available"
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = policy.predict_action(obs_dict)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = policy.predict_action(obs_dict)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[TIME] predict_action avg: {(t1 - t0) * 1000 / iters:.2f} ms (GPU-synchronized, excl. cpu copy)")

def load_policy_from_ckpt(ckpt_path, device="cuda:0"):
    payload = torch.load(ckpt_path, map_location="cpu")
    assert "cfg" in payload and "state_dicts" in payload, "ckpt missing cfg/state_dicts"

    cfg = payload["cfg"]
    state_dicts = payload["state_dicts"]

    policy_cfg = None
    for path in ["policy", "workspace.policy", "model", "workspace.model"]:
        cand = OmegaConf.select(cfg, path)
        if cand is not None and (isinstance(cand, dict) or "_target_" in cand):
            if "_target_" in cand:
                policy_cfg = cand
                break
    if policy_cfg is None:
        raise RuntimeError("Cannot find policy _target_ in ckpt['cfg'].")

    policy = hydra.utils.instantiate(policy_cfg)

    if "ema_model" in state_dicts:
        sd = state_dicts["ema_model"]
        print("[LOAD] using state_dicts['ema_model']")
    elif "model" in state_dicts:
        sd = state_dicts["model"]
        print("[LOAD] using state_dicts['model']")
    else:
        raise RuntimeError("ckpt['state_dicts'] missing ema_model/model")

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
    policy.to(device).eval()
    return policy


def build_obs_dict_from_arrays(img_hwc_uint8, rot3, grip, device, img_hw=(84, 84)):
    """
    img_hwc_uint8: (H,W,3) RGB uint8
    rot3: (3,) rotvec
    grip: scalar (0/1)
    output:
      obs.image: (1,1,3,H,W) float32 [0,1]
      obs.state: (1,1,4) float32
    """
    img = img_hwc_uint8
    if (img.shape[0], img.shape[1]) != img_hw:
        img = cv2.resize(img, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW

    rot3 = np.asarray(rot3, dtype=np.float32).reshape(3)
    grip = float(grip)
    state = np.concatenate([rot3, np.array([grip], dtype=np.float32)], axis=0).astype(np.float32)

    return {
        "image": to_torch(img[None, None, ...], device),
        "state": to_torch(state[None, None, ...], device),
    }


def find_episode_root(h5: h5py.File):
    """
    兼容两种常见结构：
      A) /obs/camera_0, /obs/robot_eef_rot, /action ...
      B) /data/demo_0/obs/..., /data/demo_0/actions ...
    返回 (obs_group, action_array or None)
    """
    if "obs" in h5:
        obs_g = h5["obs"]
        act = None
        if "action" in h5:
            # 可能是 (T,7) 或 group
            if isinstance(h5["action"], h5py.Dataset):
                act = h5["action"][...]
            elif isinstance(h5["action"], h5py.Group) and "target_pose" in h5["action"]:
                act = h5["action/target_pose"][...]
        return obs_g, act

    if "data" in h5:
        data_g = h5["data"]
        # pick first demo
        demo_keys = sorted(list(data_g.keys()))
        if len(demo_keys) == 0:
            raise RuntimeError("No demos found under /data")
        demo = data_g[demo_keys[0]]
        obs_g = demo["obs"]
        act = None
        if "actions" in demo:
            act = demo["actions"][...]
        elif "action" in demo:
            act = demo["action"][...]
        return obs_g, act

    raise RuntimeError("Unknown hdf5 layout: expected /obs or /data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--h5", required=True, type=str)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--image_key", default="camera_0", type=str, help="e.g. camera_0 or raw_camera_0")
    ap.add_argument("--rot_key", default="robot_eef_rot", type=str)
    ap.add_argument("--grip_key", default="gripper_open_state", type=str)
    ap.add_argument("--img_h", default=84, type=int)
    ap.add_argument("--img_w", default=84, type=int)
    ap.add_argument("--max_T", default=200, type=int)
    ap.add_argument("--action_from", default="auto", choices=["auto", "action", "target_pose", "actions"],
                    help="which dataset to compare against if available")
    args = ap.parse_args()

    policy = load_policy_from_ckpt(args.ckpt, device=args.device)
    steps = 10
    policy.num_inference_steps = steps
    policy.noise_scheduler.set_timesteps(steps, device="cuda")
    print("-------------------------------------------------------")
    # print(type(policy))
    # print("\n".join([f"{k} = {v}" for k,v in find_step_fields(policy)]))
    # print(find_sampler_fields(policy))
    print(type(policy.noise_scheduler))
    print(hasattr(policy.noise_scheduler, "config"), hasattr(policy.noise_scheduler, "set_timesteps"))
    print("-------------------------------------------------------")
    for sub in ["scheduler", "noise_scheduler"]:
        if hasattr(policy, sub):
            print(sub, find_sampler_fields(getattr(policy, sub)))
    with h5py.File(args.h5, "r") as h5:
        obs_g, act_gt = find_episode_root(h5)

        assert args.image_key in obs_g, f"missing image key in h5 obs: {args.image_key}"
        assert args.rot_key in obs_g, f"missing rot key in h5 obs: {args.rot_key}"
        assert args.grip_key in obs_g, f"missing grip key in h5 obs: {args.grip_key}"

        imgs = obs_g[args.image_key][...]
        rots = obs_g[args.rot_key][...]
        grips = obs_g[args.grip_key][...]

        T = min(len(imgs), args.max_T)
        print(f"[INFO] loaded T={len(imgs)} (use {T})")
        print(f"[INFO] img shape: {imgs.shape}, rot shape: {rots.shape}, grip shape: {grips.shape}")
        if act_gt is not None:
            print(f"[INFO] gt action shape: {act_gt.shape}")

        preds = []
        gts = []

        for t in range(5):
            img = imgs[t]
            rot3 = rots[t] if np.asarray(rots).ndim >= 2 else rots  # allow (T,3) or (3,)
            grip = grips[t] if np.asarray(grips).ndim >= 2 else grips  # allow (T,1) or (1,)
            grip = float(np.asarray(grip).reshape(-1)[0])

            obs_dict = build_obs_dict_from_arrays(
                img_hwc_uint8=img,
                rot3=rot3,
                grip=grip,
                device=args.device,
                img_hw=(args.img_h, args.img_w),
            )
            if t == 0:
                gpu_sanity_check(policy, obs_dict)
                timed_predict(policy, obs_dict, iters=10, warmup=3)

            t1 = time.time()
            with torch.no_grad():
                with autocast():
                    out = policy.predict_action(obs_dict)
                    a = out["action"][0, 0].detach().cpu().numpy()  # 取第一步 (7,)
            t2 = time.time()
            print(f"[WARMUP] t={t} policy inference time: {(t2 - t1)*1000:.2f} ms")
            preds.append(a)

            if act_gt is not None and t < len(act_gt):
                gts.append(act_gt[t])

        preds = np.asarray(preds, dtype=np.float32)
        print("[PRED] action stats:", "mean", preds.mean(axis=0), "std", preds.std(axis=0))

        if len(gts) > 0:
            gts = np.asarray(gts, dtype=np.float32)
            # 维度对齐：有些保存的是 (T,7)；有些是 (T,?)，只比较前7
            D = min(preds.shape[1], gts.shape[1])
            diff = preds[:, :D] - gts[:, :D]
            mae = np.mean(np.abs(diff), axis=0)
            mse = np.mean(diff ** 2, axis=0)
            print("[COMPARE] D=", D)
            print("[COMPARE] MAE per-dim:", mae)
            print("[COMPARE] MSE per-dim:", mse)
            print("[COMPARE] MAE mean:", float(np.mean(mae)), "MSE mean:", float(np.mean(mse)))
        else:
            print("[WARN] no ground-truth actions found in this h5 layout; printed prediction stats only.")


if __name__ == "__main__":
    main()
