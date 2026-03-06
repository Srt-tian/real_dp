from diffusion_policy.real_world.real_env import _USBCamera
import cv2
my_cam = _USBCamera(cam_id=2, fps=30)
while True:
    img = my_cam.read_rgb()
    cv2.imshow("USB Camera Frame", img )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
