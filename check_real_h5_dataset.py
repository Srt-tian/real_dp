#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import importlib.util
from typing import Any, Dict, Tuple, List

import yaml
import numpy as np
import h5py


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_dataset_class(py_path: str, class_name: str = "RealH5ImageDataset"):
    spec = importlib.util.spec_from_file_location("real_h5_image_dataset_mod", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, class_name):
        raise RuntimeError(f"{py_path} does not define class {class_name}")
    return getattr(mod, class_name)


def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def check_one_h5(
    h5_path: str,
    expect_hw: Tuple[int, int],
    expect_state_dim: int,
    expect_action_dim: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": h5_path, "ok": True, "errors": [], "warnings": []}

    def err(msg: str):
        report["ok"] = False
        report["errors"].append(msg)

    def warn(msg: str):
        report["warnings"].append(msg)

    with h5py.File(h5_path, "r") as f:
        # required keys by your dataset code
        # obs/camera_0, obs/robot_eef_rot, obs/gripper_open_state, action/target_pose
        if "obs" not in f:
            err("missing group: /obs")
            return report
        if "action" not in f:
            err("missing group: /action")
            return report

        for k in ["camera_0", "robot_eef_rot", "gripper_open_state"]:
            if k not in f["obs"]:
                err(f"missing dataset: /obs/{k}")
        if "target_pose" not in f["action"]:
            err("missing dataset: /action/target_pose")

        if not report["ok"]:
            return report

        img = f["obs"]["camera_0"]
        rot = f["obs"]["robot_eef_rot"]
        grip = f["obs"]["gripper_open_state"]
        act = f["action"]["target_pose"]

        # shapes
        T = img.shape[0]
        if rot.shape[0] != T:
            err(f"T mismatch: obs/robot_eef_rot has {rot.shape[0]} but image has {T}")
        if grip.shape[0] != T:
            err(f"T mismatch: obs/gripper_open_state has {grip.shape[0]} but image has {T}")
        if act.shape[0] != T:
            err(f"T mismatch: action/target_pose has {act.shape[0]} but image has {T}")

        # image shape
        if len(img.shape) != 4:
            err(f"image rank should be 4 (T,H,W,C), got {img.shape}")
        else:
            H, W, C = img.shape[1], img.shape[2], img.shape[3]
            if (H, W) != expect_hw:
                err(f"image HW expected {expect_hw}, got {(H, W)}")
            if C != 3:
                err(f"image channel expected 3, got {C}")

        # dtypes
        if img.dtype != np.uint8:
            warn(f"image dtype expected uint8, got {img.dtype} (not fatal, but confirm)")
        if not np.issubdtype(rot.dtype, np.number):
            err(f"robot_eef_rot dtype must be numeric, got {rot.dtype}")
        if not np.issubdtype(grip.dtype, np.number):
            err(f"gripper_open_state dtype must be numeric, got {grip.dtype}")
        if not np.issubdtype(act.dtype, np.number):
            err(f"target_pose dtype must be numeric, got {act.dtype}")

        # dims
        if rot.shape[-1] != 3:
            err(f"robot_eef_rot last dim expected 3, got {rot.shape}")
        # grip can be (T,) or (T,1) in your dataset code; we accept both
        if len(grip.shape) == 1:
            grip_dim = 1
        elif len(grip.shape) == 2 and grip.shape[1] == 1:
            grip_dim = 1
        else:
            err(f"gripper_open_state expected shape (T,) or (T,1), got {grip.shape}")
            grip_dim = None  # type: ignore

        if act.shape[-1] != expect_action_dim:
            err(f"action dim expected {expect_action_dim}, got {act.shape}")

        # quick numeric sanity (sample a few points, avoid reading all huge arrays)
        if report["ok"]:
            # sample indices
            sample_idx = [0, min(T - 1, 1), T // 2, T - 1] if T > 0 else []
            sample_idx = sorted(set([i for i in sample_idx if 0 <= i < T]))

            # image range check
            # only check a small slice for speed
            img0 = img[sample_idx[0]] if sample_idx else img[0]
            if img0.min() < 0 or img0.max() > 255:
                warn(f"image value range looks odd: min={img0.min()} max={img0.max()}")

            # grip values check
            g = grip[sample_idx]
            g = np.asarray(g).reshape(-1)
            uniq = np.unique(g)
            if len(uniq) > 10:
                warn(f"gripper_open_state has many unique values (expected binary-ish): n_unique={len(uniq)}")
            else:
                # common expectation: 0/1
                if not np.all(np.isin(uniq, [0, 1])):
                    warn(f"gripper_open_state unique values not subset of {{0,1}}: {uniq}")

            # NaN/inf check for rot/action
            r = np.asarray(rot[sample_idx], dtype=np.float32)
            a = np.asarray(act[sample_idx], dtype=np.float32)
            if not np.isfinite(r).all():
                err("robot_eef_rot contains NaN/Inf in sampled rows")
            if not np.isfinite(a).all():
                err("action/target_pose contains NaN/Inf in sampled rows")

            # state dim implied
            if grip_dim == 1:
                state_dim = 3 + 1
                if state_dim != expect_state_dim:
                    err(f"state dim expected {expect_state_dim}, but rot(3)+grip(1)={state_dim}")

        if verbose:
            report["T"] = int(T)
            report["img_shape"] = tuple(img.shape)
            report["rot_shape"] = tuple(rot.shape)
            report["grip_shape"] = tuple(grip.shape)
            report["act_shape"] = tuple(act.shape)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to yaml config (my_config.yaml)")
    ap.add_argument("--dataset_py", required=True, help="path to real_h5_image_dataset.py")
    ap.add_argument("--glob_override", default=None, help="override h5_glob_path in config")
    ap.add_argument("--max_files", type=int, default=20, help="max h5 files to check (0=all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # pull expected shapes from config
    # your config stores them under task.shape_meta
    shape_meta = _get(cfg, "task.shape_meta")
    if shape_meta is None:
        raise RuntimeError("config missing: task.shape_meta")

    exp_img = _get(shape_meta, "obs.image.shape")
    exp_state = _get(shape_meta, "obs.state.shape")
    exp_action = _get(shape_meta, "action.shape")
    if exp_img != [3, 84, 84]:
        print(f"[WARN] config obs.image.shape is {exp_img}, script assumes [3,84,84] (update script if needed)")
    if exp_state is None or len(exp_state) != 1:
        raise RuntimeError(f"config obs.state.shape invalid: {exp_state}")
    if exp_action is None or len(exp_action) != 1:
        raise RuntimeError(f"config action.shape invalid: {exp_action}")

    expect_hw = (exp_img[1], exp_img[2])
    expect_state_dim = int(exp_state[0])
    expect_action_dim = int(exp_action[0])

    horizon = _get(cfg, "horizon", 16)
    n_obs_steps = _get(cfg, "n_obs_steps", 1)
    pad_before = _get(cfg, "task.dataset.pad_before", 0)
    pad_after = _get(cfg, "task.dataset.pad_after", 0)

    print("[CONFIG] horizon:", horizon)
    print("[CONFIG] n_obs_steps:", n_obs_steps)
    print("[CONFIG] pad_before:", pad_before, "pad_after:", pad_after)
    print("[CONFIG] expected image:", exp_img, "state:", exp_state, "action:", exp_action)

    h5_glob_path = args.glob_override or _get(cfg, "task.dataset.h5_glob_path")
    if not h5_glob_path:
        raise RuntimeError("config missing: task.dataset.h5_glob_path")
    paths = sorted(glob.glob(os.path.expanduser(h5_glob_path), recursive=True))
    if len(paths) == 0:
        raise RuntimeError(f"No hdf5 files matched glob: {h5_glob_path}")

    if args.max_files and args.max_files > 0:
        paths_to_check = paths[: args.max_files]
    else:
        paths_to_check = paths

    print(f"[DATA] glob matched {len(paths)} files, checking {len(paths_to_check)}")

    # 1) file-level check
    all_ok = True
    for p in paths_to_check:
        rep = check_one_h5(
            p, expect_hw=expect_hw, expect_state_dim=expect_state_dim, expect_action_dim=expect_action_dim,
            verbose=args.verbose
        )
        if rep["ok"]:
            print(f"[OK] {p}")
        else:
            all_ok = False
            print(f"[FAIL] {p}")
            for e in rep["errors"]:
                print("   -", e)
        for w in rep["warnings"]:
            print("   [WARN]", w)

        if args.verbose:
            for k in ["T", "img_shape", "rot_shape", "grip_shape", "act_shape"]:
                if k in rep:
                    print(f"   {k}: {rep[k]}")

    # 2) dataset instantiation + sample shape check
    # This checks whether your dataset code + config parameters actually work together.
    DatasetCls = import_dataset_class(args.dataset_py, "RealH5ImageDataset")
    ds = DatasetCls(
        h5_glob_path=h5_glob_path,
        horizon=int(horizon),
        pad_before=int(pad_before),
        pad_after=int(pad_after),
        seed=42,
        val_ratio=_get(cfg, "task.dataset.val_ratio", 0.0),
        max_train_episodes=_get(cfg, "task.dataset.max_train_episodes", None),
    )

    print(f"[DATASET] len(dataset) = {len(ds)}")
    x0 = ds[0]
    img = x0["obs"]["image"]
    state = x0["obs"]["state"]
    act = x0["action"]

    print("[SAMPLE] obs.image:", tuple(img.shape), img.dtype, "min/max", float(img.min()), float(img.max()))
    print("[SAMPLE] obs.state:", tuple(state.shape), state.dtype)
    print("[SAMPLE] action   :", tuple(act.shape), act.dtype)

    # strict checks vs config
    assert tuple(img.shape[-3:]) == (3, exp_img[1], exp_img[2]), "image shape mismatch vs config"
    assert tuple(state.shape[-1:]) == (expect_state_dim,), "state dim mismatch vs config"
    assert tuple(act.shape[-1:]) == (expect_action_dim,), "action dim mismatch vs config"

    # also ensure horizon length matches dataset horizon
    assert img.shape[0] == horizon, "sample horizon mismatch"
    assert state.shape[0] == horizon, "sample horizon mismatch"
    assert act.shape[0] == horizon, "sample horizon mismatch"

    print("[PASS] Dataset + config are consistent.")

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
