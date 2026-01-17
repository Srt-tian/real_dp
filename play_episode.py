import h5py
import numpy as np
import cv2
import sys
import time
from pathlib import Path

def play_hdf5(h5_path: str, fps: float = 10.0, scale: int = 6, bgr_display: bool = True):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"[ERROR] File not found: {h5_path}")
        return

    with h5py.File(h5_path, "r") as f:
        cam = f["obs/camera_0"][:]  # (T,H,W,3), uint8, typically RGB
        stage = f["stage"][:] if "stage" in f else None
        ts = f["timestamp"][:] if "timestamp" in f else None

    if cam.ndim != 4 or cam.shape[-1] != 3:
        print(f"[ERROR] Unexpected camera shape: {cam.shape}")
        return

    T, H, W, C = cam.shape
    print(f"[INFO] Loaded camera_0: shape={cam.shape}, dtype={cam.dtype}")
    print("[INFO] Controls:")
    print("  Space: pause/resume")
    print("  Left/Right: step frame when paused")
    print("  +/-: speed down/up")
    print("  q or ESC: quit")
    print("  Ctrl+C: quit\n")

    win = "episode_player"
    paused = False
    idx = 0

    delay_ms = int(max(1, 1000.0 / max(0.1, fps)))

    def make_vis(img_rgb: np.ndarray, idx_: int) -> np.ndarray:
        img = img_rgb
        if bgr_display:
            img = img[..., ::-1]  # RGB -> BGR for cv2.imshow
        if scale != 1:
            img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

        # overlay (only on display, not saved)
        lines = [f"frame {idx_+1}/{T}", f"fps={1000.0/delay_ms:.1f}  scale={scale}"]
        if stage is not None:
            lines.append(f"stage={int(stage[idx_])}")
        if ts is not None and idx_ > 0:
            dt = float(ts[idx_] - ts[idx_-1])
            lines.append(f"dt={dt:.3f}s")

        y = 25
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y += 28
        return img

    try:
        last_show = time.time()
        while True:
            vis = make_vis(cam[idx], idx)
            cv2.imshow(win, vis)

            # 用短 waitKey，保证 Ctrl+C 能打断
            key = cv2.waitKey(1 if not paused else 50) & 0xFF

            if key in (ord('q'), 27):  # q / ESC
                break

            if key == ord(' '):  # space pause/resume
                paused = not paused

            if key in (ord('+'), ord('=')):  # speed up
                delay_ms = max(1, int(delay_ms * 0.8))
            if key in (ord('-'), ord('_')):  # speed down
                delay_ms = min(500, int(delay_ms * 1.25))

            # OpenCV 方向键：不同平台值会变，这里用 waitKeyEx 更稳
            # 但为了最小依赖，这里同时兼容常见值：
            # Left: 81, Right: 83 (某些环境)
            if paused and key == 81:  # left
                idx = max(0, idx - 1)
            if paused and key == 83:  # right
                idx = min(T - 1, idx + 1)

            if not paused:
                # 按 delay_ms 播放
                now = time.time()
                if (now - last_show) * 1000.0 >= delay_ms:
                    idx += 1
                    last_show = now
                    if idx >= T:
                        # 从头再放一遍？这里默认播放一遍就退出
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Exit player.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_episode_hdf5.py path/to/episode.hdf5 [fps] [scale]")
        sys.exit(1)

    path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) >= 3 else 10.0
    scale = int(sys.argv[3]) if len(sys.argv) >= 4 else 6
    play_hdf5(path, fps=fps, scale=scale)
