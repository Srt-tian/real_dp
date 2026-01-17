import h5py
import numpy as np
import cv2
import sys
from pathlib import Path

def main(h5_path):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"[ERROR] File not found: {h5_path}")
        return

    print(f"\n=== Checking: {h5_path} ===\n")

    with h5py.File(h5_path, "r") as f:
        # -------- 1. Print structure --------
        print("[INFO] HDF5 keys:")
        def print_tree(name, obj):
            print("  ", name)
        f.visititems(print_tree)

        # -------- 2. Load core datasets --------
        cam = f["obs/camera_0"][:]
        act = f["action/target_pose"][:]
        stage = f["stage"][:]
        ts = f["timestamp"][:]

        T = cam.shape[0]
        print("\n[INFO] Length check:")
        print("  T(camera):", cam.shape)
        print("  T(action):", act.shape)
        print("  T(stage):", stage.shape)
        print("  T(timestamp):", ts.shape)

        if not (len(act) == len(stage) == len(ts) == T):
            print("[ERROR] Length mismatch between camera/action/stage/timestamp!")
            return
        else:
            print("[PASS] All modalities aligned on time axis.")

        # -------- 3. Image sanity --------
        print("\n[INFO] Image check:")
        print("  dtype:", cam.dtype)
        print("  min/max:", cam.min(), cam.max())

        if cam.dtype != np.uint8:
            print("[WARN] camera_0 is not uint8 (expected uint8).")

        if cam.shape[1:3] != (84, 84):
            print("[WARN] Image resolution is not 84x84.")

        # -------- 4. Action sanity --------
        print("\n[INFO] Action check:")
        print("  action shape:", act.shape)
        print("  first action:", act[0])
        print("  last action:", act[-1])

        if act.shape[1] != 6:
            print("[WARN] action dim is not 6 (xyz + rotvec).")

        # -------- 5. Timestamp sanity --------
        print("\n[INFO] Timestamp check:")
        dt = np.diff(ts)
        print("  dt mean / min / max:", dt.mean(), dt.min(), dt.max())

        if np.any(dt <= 0):
            print("[WARN] Non-increasing timestamps detected!")

        # -------- 6. Visual check --------
        idx = np.random.randint(0, T)
        img = cam[idx]

        print(f"\n[INFO] Visualizing frame {idx} (should be CLEAN, no text)")
        cv2.imshow("dataset_frame (BGR)", img[..., ::-1])
        cv2.waitKey(50)
        cv2.destroyAllWindows()

        print("\n=== CHECK COMPLETE ===")
        print("If no ERROR above, dataset is basically OK for DP / imitation learning.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_episode_hdf5.py path/to/episode.hdf5")
        sys.exit(1)

    main(sys.argv[1])
