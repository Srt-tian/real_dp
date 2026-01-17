import cv2
import time

def main():
    cam_id = 2              # 如果你有多个USB相机，改成 1 / 2 / ...
    width, height = 1280, 720
    fps = 30

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open USB camera id={cam_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print("[INFO] USB camera opened successfully")
    print(f"       Resolution request: {width} x {height}, FPS: {fps}")

    time.sleep(0.5)  # give camera some warm-up time

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to read frame")
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"{w}x{h}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("USB Camera Check", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released, exit.")


if __name__ == "__main__":
    main()
