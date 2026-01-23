import os, glob, shutil
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

IN_GLOB = "data/demo_real/**/episode.hdf5"
OUT_ROOT = "data/demo_real_delta"
ACTION_KEY = "action/target_pose"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def rotvec_delta_series(rotvec_abs: np.ndarray) -> np.ndarray:
    """
    rotvec_abs: (T, 3) absolute-ish rotvec sequence (each row is a rotvec)
    returns delta_rotvec: (T, 3) where
      delta[t] = (R_t * R_{t-1}^{-1}).as_rotvec(), delta[0]=0
    """
    T = rotvec_abs.shape[0]
    delta = np.zeros((T, 3), dtype=np.float32)
    if T <= 1:
        return delta

    # Convert each step to Rotation
    R_abs = R.from_rotvec(rotvec_abs.astype(np.float64))
    # Relative: R_t * inv(R_{t-1})
    R_rel = R_abs[1:] * R_abs[:-1].inv()
    delta[1:] = R_rel.as_rotvec().astype(np.float32)
    delta[0] = 0.0
    return delta

def convert_one(in_path, out_path):
    ensure_dir(os.path.dirname(out_path))
    shutil.copy2(in_path, out_path)

    with h5py.File(out_path, "r+") as f:
        a = f[ACTION_KEY][:].astype(np.float32)  # (T,7)
        assert a.ndim == 2 and a.shape[1] == 7, f"expected (T,7), got {a.shape}"
        T = a.shape[0]

        delta = np.zeros_like(a, dtype=np.float32)

        # Δpos
        if T > 1:
            delta[1:, :3] = a[1:, :3] - a[:-1, :3]
        delta[0, :3] = 0.0

        # Δrot (normal computation)
        delta[:, 3:6] = rotvec_delta_series(a[:, 3:6])

        # gripper: keep as command/state at time t
        delta[:, 6] = a[:, 6]

        # overwrite dataset
        del f[ACTION_KEY]
        f.create_dataset(ACTION_KEY, data=delta, dtype=np.float32)

    return T

def main():
    files = sorted(glob.glob(IN_GLOB, recursive=True))
    assert len(files) > 0, f"no files matched: {IN_GLOB}"
    print("found", len(files), "files")

    for i, in_path in enumerate(files, 1):
        rel = os.path.relpath(in_path, "data/demo_real")
        out_path = os.path.join(OUT_ROOT, rel)
        T = convert_one(in_path, out_path)
        if i <= 3:
            print("[OK]", in_path, "->", out_path, "T=", T)

    print("done. wrote to", OUT_ROOT)

if __name__ == "__main__":
    main()
