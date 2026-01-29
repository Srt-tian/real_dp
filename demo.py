#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
from typing import List, Tuple, Optional

def _find_actions_in_file(h5: h5py.File, action_key: str = "auto") -> Optional[np.ndarray]:
    """
    Try to find actions array (T, D) in common layouts.
    action_key:
      - "auto": search common keys
      - otherwise: use that exact path
    """
    if action_key != "auto":
        if action_key in h5:
            return h5[action_key][...]
        # allow path like "data/demo_0/actions"
        if action_key.startswith("/"):
            action_key = action_key[1:]
        if action_key in h5:
            return h5[action_key][...]
        raise KeyError(f"action_key={action_key!r} not found in file")

    # Common patterns:
    # 1) root: /action (dataset) or /action/target_pose (dataset)
    if "action" in h5:
        obj = h5["action"]
        if isinstance(obj, h5py.Dataset):
            return obj[...]
        if isinstance(obj, h5py.Group) and "target_pose" in obj:
            return obj["target_pose"][...]

    # 2) root: /actions
    if "actions" in h5 and isinstance(h5["actions"], h5py.Dataset):
        return h5["actions"][...]

    # 3) /data/<demo>/actions
    if "data" in h5 and isinstance(h5["data"], h5py.Group):
        data_g = h5["data"]
        demo_keys = sorted(list(data_g.keys()))
        if len(demo_keys) > 0:
            demo = data_g[demo_keys[0]]
            for k in ["actions", "action"]:
                if k in demo and isinstance(demo[k], h5py.Dataset):
                    return demo[k][...]
            # sometimes nested: demo/action/target_pose
            if "action" in demo and isinstance(demo["action"], h5py.Group) and "target_pose" in demo["action"]:
                return demo["action/target_pose"][...]

    return None

def _iter_action_arrays(path_list: List[str], action_key: str) -> List[np.ndarray]:
    arrays = []
    for p in path_list:
        with h5py.File(p, "r") as h5:
            act = _find_actions_in_file(h5, action_key=action_key)
            if act is None:
                print(f"[WARN] no actions found in {p}")
                continue
            act = np.asarray(act)
            # flatten if (T,1,D) etc
            if act.ndim == 3 and act.shape[0] == 1:
                act = act[0]
            arrays.append(act)
    return arrays

def _robust_clip_suggestions(a: np.ndarray, qs=(0.99, 0.995, 0.999)) -> dict:
    """
    a: (N,7) actions
    returns stats for dpos/drot dims and norms
    """
    assert a.ndim == 2 and a.shape[1] >= 6, f"expect (N,D>=6), got {a.shape}"
    dpos = a[:, 0:3]
    drot = a[:, 3:6]

    abs_dpos = np.abs(dpos)
    abs_drot = np.abs(drot)

    pos_norm = np.linalg.norm(dpos, axis=1)
    rot_norm = np.linalg.norm(drot, axis=1)

    out = {}

    # per-dim percentiles of absolute values
    out["abs_pos_per_dim_percentiles"] = {q: np.quantile(abs_dpos, q, axis=0) for q in qs}
    out["abs_rot_per_dim_percentiles"] = {q: np.quantile(abs_drot, q, axis=0) for q in qs}

    # norm percentiles
    out["pos_norm_percentiles"] = {q: float(np.quantile(pos_norm, q)) for q in qs}
    out["rot_norm_percentiles"] = {q: float(np.quantile(rot_norm, q)) for q in qs}

    # basic stats
    out["pos_norm_mean_std"] = (float(pos_norm.mean()), float(pos_norm.std()))
    out["rot_norm_mean_std"] = (float(rot_norm.mean()), float(rot_norm.std()))
    out["dpos_mean_std"] = (dpos.mean(axis=0), dpos.std(axis=0))
    out["drot_mean_std"] = (drot.mean(axis=0), drot.std(axis=0))
    return out

def _pretty(v, unit_scale=1.0):
    v = np.asarray(v, dtype=np.float64) * unit_scale
    return "[" + ", ".join(f"{x:.6g}" for x in v.tolist()) + "]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True, help="one or more .hdf5 paths")
    ap.add_argument("--action_key", default="auto", help='action dataset key/path; default "auto"')
    ap.add_argument("--assume_units", default="m_rad", choices=["m_rad", "mm_deg"],
                    help="only affects printing convenience; does NOT change computed numbers")
    ap.add_argument("--qs", default="0.99,0.995,0.999", help="quantiles to report, comma separated")
    args = ap.parse_args()

    qs = tuple(float(x.strip()) for x in args.qs.split(",") if x.strip())
    arrays = _iter_action_arrays(args.h5, action_key=args.action_key)
    if len(arrays) == 0:
        raise SystemExit("No valid actions found in provided files.")

    # concatenate across files, but keep only first 7 dims if longer
    acts = []
    for a in arrays:
        a = np.asarray(a)
        if a.ndim != 2:
            print("[WARN] skipping non-2D action array with shape", a.shape)
            continue
        if a.shape[1] < 6:
            print("[WARN] skipping action dims < 6:", a.shape)
            continue
        # keep up to 7 dims if present
        D = min(a.shape[1], 7)
        acts.append(a[:, :D])
    a = np.concatenate(acts, axis=0)
    print(f"[INFO] loaded total actions: {a.shape} from {len(acts)} arrays / {len(args.h5)} files")

    stats = _robust_clip_suggestions(a, qs=qs)

    # units printing
    if args.assume_units == "m_rad":
        pos_scale = 1.0
        rot_scale = 1.0
        pos_unit = "m"
        rot_unit = "rad"
        pos_norm_unit = "m"
        rot_norm_unit = "rad"
    else:
        # convenient prints: m->mm, rad->deg
        pos_scale = 1000.0
        rot_scale = 180.0 / np.pi
        pos_unit = "mm"
        rot_unit = "deg"
        pos_norm_unit = "mm"
        rot_norm_unit = "deg"

    dpos_mean, dpos_std = stats["dpos_mean_std"]
    drot_mean, drot_std = stats["drot_mean_std"]

    print("\n==================== BASIC STATS ====================")
    print(f"[dpos] mean { _pretty(dpos_mean, pos_scale) } {pos_unit}  std { _pretty(dpos_std, pos_scale) } {pos_unit}")
    print(f"[drot] mean { _pretty(drot_mean, rot_scale) } {rot_unit}  std { _pretty(drot_std, rot_scale) } {rot_unit}")
    pn_mean, pn_std = stats["pos_norm_mean_std"]
    rn_mean, rn_std = stats["rot_norm_mean_std"]
    print(f"[||dpos||] mean {pn_mean*pos_scale:.6g} {pos_norm_unit}  std {pn_std*pos_scale:.6g} {pos_norm_unit}")
    print(f"[||drot||] mean {rn_mean*rot_scale:.6g} {rot_norm_unit}  std {rn_std*rot_scale:.6g} {rot_norm_unit}")

    print("\n==================== QUANTILE SUGGESTIONS ====================")
    for q in qs:
        pos_dim = stats["abs_pos_per_dim_percentiles"][q]
        rot_dim = stats["abs_rot_per_dim_percentiles"][q]
        pos_n = stats["pos_norm_percentiles"][q]
        rot_n = stats["rot_norm_percentiles"][q]

        print(f"\n--- q = {q:.4f} ---")
        print(f"per-dim clip |dpos| <= { _pretty(pos_dim, pos_scale) } {pos_unit}")
        print(f"per-dim clip |drot| <= { _pretty(rot_dim, rot_scale) } {rot_unit}")
        print(f"norm clip    ||dpos|| <= {pos_n*pos_scale:.6g} {pos_norm_unit}")
        print(f"norm clip    ||drot|| <= {rot_n*rot_scale:.6g} {rot_norm_unit}")

    # Provide a practical default recommendation:
    q_default = 0.995 if 0.995 in qs else qs[0]
    pos_clip = stats["pos_norm_percentiles"][q_default]
    rot_clip = stats["rot_norm_percentiles"][q_default]
    print("\n==================== RECOMMENDED DEFAULT (PRACTICAL) ====================")
    print(f"Recommend using norm-based clipping at q={q_default:.4f}:")
    if args.assume_units == "m_rad":
        print(f"  max_dpos_norm = {pos_clip:.6g}  # meters per step")
        print(f"  max_drot_norm = {rot_clip:.6g}  # rad per step")
    else:
        print(f"  max_dpos_norm = {pos_clip*1000.0:.6g}  # mm per step (printing)")
        print(f"  max_drot_norm = {rot_clip*(180.0/np.pi):.6g}  # deg per step (printing)")
    print("Tip: if you execute k steps per inference, keep per-step clipping (do NOT multiply by k).")

if __name__ == "__main__":
    main()
