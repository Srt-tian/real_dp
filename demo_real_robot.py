import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as trans

from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
HOME_POSE6 = np.array([-0.02109733 ,-0.41616943 , 0.13228695 , 0.07429853 , 2.28826478, -2.14155153], dtype=np.float64)
# Keyboard controls:
#   q : quit program
#   c : start recording a new episode
#   p : stop recording and save episode
#   x : discard current episode (do not save)
#   w/s/a/d/r/f : translate TCP (+x/-x/+y/-y/+z/-z)
#   i/k/j/l/u/o : rotate TCP (rx/ry/rz)
#   t : toggle gripper open/close
#   h : go to HOME pose (target pose override)
#   g : print current TCP pose
# ---------------- keyboard teleop ----------------
def get_keyboard_delta_from_events(press_events, key_counter, pos_step, rot_step):
    boost = 2.0 if key_counter[Key.shift] else 1.0

    dpos = np.zeros(3, dtype=np.float32)
    drot_xyz = np.zeros(3, dtype=np.float32)

    for ev in press_events:
        # translation
        if ev == KeyCode(char='a'): dpos[0] += 1
        elif ev == KeyCode(char='d'): dpos[0] -= 1
        elif ev == KeyCode(char='w'): dpos[1] -= 1
        elif ev == KeyCode(char='s'): dpos[1] += 1
        elif ev == KeyCode(char='r'): dpos[2] += 1
        elif ev == KeyCode(char='f'): dpos[2] -= 1

        # rotation
        elif ev == KeyCode(char='i'): drot_xyz[0] += 1
        elif ev == KeyCode(char='k'): drot_xyz[0] -= 1
        elif ev == KeyCode(char='l'): drot_xyz[1] += 1
        elif ev == KeyCode(char='j'): drot_xyz[1] -= 1
        elif ev == KeyCode(char='u'): drot_xyz[2] += 1
        elif ev == KeyCode(char='o'): drot_xyz[2] -= 1

    dpos *= pos_step * boost
    drot_xyz *= rot_step * boost
    return dpos, drot_xyz

# ---------------- CLI ----------------
@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5 IP address, e.g. 192.168.0.204")
@click.option('--frequency', '-f', default=5.0, type=float)
@click.option('--command_latency', '-cl', default=0.01, type=float)
def main(output, robot_ip, frequency, command_latency):
    dt = 1.0 / frequency
    open_state = 0  # start open
    goto_home = False
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
             RealEnv(
                 output_dir=output,
                 robot_ip=robot_ip,
                 obs_image_resolution=(1280, 720),
                 frequency=frequency,
                 n_obs_steps=1,
                 usb_cam_id=2
             ) as env:

            cv2.setNumThreads(1)
            time.sleep(1.0)
            print("Ready!")

            state = env.get_robot_state()
            target_pose = np.zeros(7, dtype=np.float64)
            target_pose[:6] = np.array(state["TargetTCPPose"], dtype=np.float64)
            target_pose[6] = 0.0
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            while not stop:
                # timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency

                # get obs
                obs = env.get_obs()

                press_events = key_counter.get_press_events()
                # handle open/close gripper toggle

                # 更新 target_pose 位姿...
                target_pose[6] = float(open_state)
                # handle key events
                for ev in press_events:
                    if ev == KeyCode(char='q'):
                        stop = True
                    elif ev == KeyCode(char='c'):
                        env.start_episode(
                            t_start + (iter_idx + 2) * dt
                            - time.monotonic() + time.time()
                        )
                        key_counter.clear()
                        is_recording = True
                        print("Recording!")
                    elif ev == KeyCode(char='p'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print("Stopped.")
                    elif ev == KeyCode(char='x') and is_recording:
                        env.discard_episode("discard_by_key_x")
                        key_counter.clear()
                        is_recording = False
                        print("Discarded current episode.")
                    elif ev == KeyCode(char='t'):
                        open_state = 1 - open_state
                        target_pose[6] = float(open_state)
                        print(f"[GRIP] toggled by t -> open_state={open_state}")
                    elif ev == KeyCode(char='g'):
                        robot_state = env.get_robot_state()
                        q = np.asarray(robot_state.get("ActualTCPPose", []), dtype=np.float64)
                        print("Current TCP pose:", q)
                    elif ev == KeyCode(char='h'):
                        goto_home = True

                # visualize (USB cam is always camera_0)
                vis_img = obs["raw_camera_0"][-1, :, :, ::-1].copy()
                text = f""
                if is_recording:
                    text += " | Recording"
                cv2.putText(
                    vis_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2
                )
                cv2.imshow("teleop", vis_img)
                cv2.waitKey(1)


                # teleop
                pos_step = 0.002   # 2mm per press
                rot_step = 0.035   # ~2deg per press (0.035 rad)
                if goto_home:
                    target_pose[:6] = HOME_POSE6.copy()
                    goto_home = False
                else:
                    dpos, drot_xyz = get_keyboard_delta_from_events(
                        press_events, key_counter, pos_step, rot_step
                    )

                    drot = trans.Rotation.from_euler("xyz", drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:6] = (
                        drot * trans.Rotation.from_rotvec(target_pose[3:6])
                    ).as_rotvec()
                # send command
                # env.exec_actions(
                #     actions=[target_pose.copy()],
                #     timestamps=[t_command_target - time.monotonic() + time.time()],
                #     stages=[stage],
                # )
                precise_wait(t_sample)

                now_wall = time.time()
                # lead = max(2.0 * dt, 0.05)   # >= 2 control cycles or 50ms
                lead = 0.05
                ts_cmd = now_wall + lead

                env.exec_actions(
                    obs = obs,
                    actions=[target_pose.copy()],
                    timestamps=[ts_cmd],
                )

                precise_wait(t_cycle_end)
                iter_idx += 1
                if iter_idx % 50 == 0:
                    print("lateness(ms)=", (time.monotonic() - t_cycle_end) * 1000)

if __name__ == "__main__":
    main()
