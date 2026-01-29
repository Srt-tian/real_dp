import argparse
import time
import numpy as np
import torch
import h5py
import cv2
import hydra
from omegaconf import OmegaConf
from torch.cuda.amp import autocast

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)

def normalize_img(img_hwc_uint8: np.ndarray, mode: str) -> np.ndarray:
    """
    img_hwc_uint8: (H,W,3) uint8 RGB
    return: CHW float32
    """
    img = img_hwc_uint8.astype(np.float32) / 255.0  # -> [0,1], HWC

    if mode == "0_1":
        pass
    elif mode == "imagenet":
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == "-1_1":
        img = img * 2.0 - 1.0
    else:
        raise ValueError(f"unknown img_norm={mode}")

    return img.transpose(2, 0, 1)  # CHW

def gpu_sanity_check(policy, obs_dict):
    print("[CUDA] available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[CUDA] device:", torch.cuda.get_device_name(0))
    try:
        print("[CUDA] policy param device:", next(policy.parameters()).device)
    except StopIteration:
        print("[CUDA] policy has no parameters??")

    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"[CUDA] obs[{k}] device:", v.device, "dtype:", v.dtype, "shape:", tuple(v.shape))

def timed_predict(policy, obs_dict, iters=10, warmup=3, use_amp=True):
    """只测 predict_action（不包含 .cpu().numpy()），并做 synchronize 得到真实 GPU 耗时。"""
    if not torch.cuda.is_available():
        print("[TIME] CUDA not available, skip GPU timing.")
        return
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            if use_amp:
                with autocast():
                    _ = policy.predict_action(obs_dict)
            else:
                _ = policy.predict_action(obs_dict)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            if use_amp:
                with autocast():
                    _ = policy.predict_action(obs_dict)
            else:
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

    def strip_prefix(sd_in, prefix):
        return {k[len(prefix):]: v for k, v in sd_in.items() if k.startswith(prefix)}

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

def build_obs_dict_from_arrays(
    img_hwc_uint8,
    rot3,
    grip,
    device,
    img_hw=(84, 84),
    img_norm="0_1",
):
    """
    img_hwc_uint8: (H,W,3) RGB uint8
    rot3: (3,) rotvec
    grip: scalar (0/1)
    output:
      obs["image"]: (1,1,3,H,W) float32
      obs["state"]: (1,1,4) float32
    """
    if (img_hwc_uint8.shape[0], img_hwc_uint8.shape[1]) != img_hw:
        img_hwc_uint8 = cv2.resize(img_hwc_uint8, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)

    img_chw = normalize_img(img_hwc_uint8, img_norm).astype(np.float32)

    rot3 = np.asarray(rot3, dtype=np.float32).reshape(3)
    grip = float(np.asarray(grip).reshape(-1)[0])
    state = np.concatenate([rot3, np.array([grip], dtype=np.float32)], axis=0).astype(np.float32)

    return {
        "image": to_torch(img_chw[None, None, ...], device).float(),
        "state": to_torch(state[None, None, ...], device).float(),
    }

def find_episode_root(h5: h5py.File):
    """
    兼容两种常见结构：
      A) /obs/..., /action ...
      B) /data/demo_0/obs/..., /data/demo_0/actions ...
    返回 (obs_group, action_array or None)
    """
    if "obs" in h5:
        obs_g = h5["obs"]
        act = None
        if "action" in h5:
            if isinstance(h5["action"], h5py.Dataset):
                act = h5["action"][...]
            elif isinstance(h5["action"], h5py.Group) and "target_pose" in h5["action"]:
                act = h5["action/target_pose"][...]
        return obs_g, act

    if "data" in h5:
        data_g = h5["data"]
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
    ap.add_argument("--eval_T", default=10, type=int, help="how many timesteps to compare per mode")
    ap.add_argument("--use_amp", action="store_true", help="use autocast for predict_action")
    args = ap.parse_args()

    policy = load_policy_from_ckpt(args.ckpt, device=args.device)

    # set diffusion inference steps if available
    steps = 10
    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = steps
    if hasattr(policy, "noise_scheduler") and hasattr(policy.noise_scheduler, "set_timesteps"):
        # device can be "cuda:0" / "cpu"
        policy.noise_scheduler.set_timesteps(steps, device=args.device)

    with h5py.File(args.h5, "r") as h5:
        obs_g, act_gt = find_episode_root(h5)

        assert args.image_key in obs_g, f"missing image key in h5 obs: {args.image_key}"
        assert args.rot_key in obs_g, f"missing rot key in h5 obs: {args.rot_key}"
        assert args.grip_key in obs_g, f"missing grip key in h5 obs: {args.grip_key}"

        imgs = obs_g[args.image_key][...]
        rots = obs_g[args.rot_key][...]
        grips = obs_g[args.grip_key][...]

        T = min(len(imgs), args.max_T)
        Teval = min(T, args.eval_T)

        print(f"[INFO] loaded T={len(imgs)} (use {T}) eval_T={Teval}")
        print(f"[INFO] img shape: {imgs.shape}, rot shape: {rots.shape}, grip shape: {grips.shape}")
        if act_gt is not None:
            print(f"[INFO] gt action shape: {act_gt.shape}")
        else:
            print("[WARN] no ground-truth action found in this h5 layout; will only print prediction stats.")

        # one-time sanity + timing with default mode 0_1 at t=0
        obs0 = build_obs_dict_from_arrays(
            img_hwc_uint8=imgs[0],
            rot3=rots[0],
            grip=grips[0],
            device=args.device,
            img_hw=(args.img_h, args.img_w),
            img_norm="0_1",
        )
        gpu_sanity_check(policy, obs0)
        timed_predict(policy, obs0, iters=10, warmup=3, use_amp=args.use_amp)

        results = {}
        for IMG_NORM in ["0_1", "imagenet", "-1_1"]:
            print("\n" + "=" * 80)
            print(f"[RUN] img_norm = {IMG_NORM}")
            print("=" * 80)

            maes = []
            preds = []
            gts = []

            for t in range(Teval):
                obs_dict = build_obs_dict_from_arrays(
                    img_hwc_uint8=imgs[t],
                    rot3=rots[t],
                    grip=grips[t],
                    device=args.device,
                    img_hw=(args.img_h, args.img_w),
                    img_norm=IMG_NORM,
                )

                with torch.no_grad():
                    if args.use_amp:
                        with autocast():
                            out = policy.predict_action(obs_dict)
                    else:
                        out = policy.predict_action(obs_dict)

                # out["action"] usually shape: (B, T_action, D)
                if not isinstance(out, dict) or "action" not in out:
                    raise RuntimeError(f"policy.predict_action output unexpected: {type(out)} keys={getattr(out, 'keys', lambda: [])()}")

                a = out["action"][0, 0].detach().float().cpu().numpy()  # first step (D,)
                preds.append(a)

                if act_gt is not None and t < len(act_gt):
                    gt = np.asarray(act_gt[t]).reshape(-1)
                    D = min(len(a), len(gt))
                    gts.append(gt[:D])
                    maes.append(float(np.mean(np.abs(a[:D] - gt[:D]))))

            preds = np.asarray(preds, dtype=np.float32)
            print("[PRED] action stats:", "mean", preds.mean(axis=0), "std", preds.std(axis=0))

            if act_gt is not None and len(maes) > 0:
                mae_mean = float(np.mean(maes))
                print(f"[RESULT] img_norm={IMG_NORM} | MAE mean = {mae_mean:.8f} (over {len(maes)} steps)")
                results[IMG_NORM] = mae_mean
            else:
                print(f"[RESULT] img_norm={IMG_NORM} | (no GT)")

        if results:
            best = min(results.items(), key=lambda kv: kv[1])
            print("\n" + "-" * 80)
            print("[SUMMARY] MAE mean per mode:")
            for k, v in results.items():
                print(f"  - {k:8s}: {v:.8f}")
            print(f"[BEST] {best[0]}  (MAE mean={best[1]:.8f})")
            print("-" * 80)

if __name__ == "__main__":
    main()
