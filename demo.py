import time
import cv2
import numpy as np

# 假设你的相机类在 real_env.py 里
from diffusion_policy.real_world.real_env import _USBCamera


def main():
    cam = _USBCamera(
        cam_id=2,              # 改成你的 USB 相机 id
        resolution=(640, 480), # 用你实际的分辨率
        fps=30,
    )

    print("Starting camera debug. Press Ctrl+C to exit.")

    try:
        i = 0
        while True:
            t0 = time.time()
            img = cam.read_rgb()
            dt = (time.time() - t0) * 1000  # ms

            print(f"[{i:04d}] read_rgb time = {dt:.2f} ms, "
                  f"img mean={img.mean():.1f}")

            cv2.imshow("debug_latest_frame", img[..., ::-1])  # RGB->BGR for imshow
            cv2.waitKey(1)

            # === 模拟你的真实控制频率 ===
            time.sleep(4.0)   # 0.25 Hz（每 4 秒一次）
            # time.sleep(0.2) # 5 Hz 时可以改成这个

            i += 1

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
