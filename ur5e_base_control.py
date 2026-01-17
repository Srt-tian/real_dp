import time
import numpy as np

# ur-rtde
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# -----------------------------
# Config
# -----------------------------
ROBOT_IP = "192.168.3.15"   # <- 改成你的UR5e IP
CONTROL_FREQUENCY = 125      # RTDE内部控制频率常用125
SAFE_SPEED = 0.25            # moveJ速度（rad/s）保守一点
SAFE_ACCEL = 0.5             # moveJ加速度（rad/s^2）

def print_state(rtde_r: RTDEReceive):
    tcp = rtde_r.getActualTCPPose()       # [x,y,z,rx,ry,rz] (m, rad)
    q   = rtde_r.getActualQ()             # 6 joints (rad)
    tcp_speed = rtde_r.getActualTCPSpeed()# (m/s, rad/s)
    print("[STATE] TCP pose  :", np.array(tcp))
    print("[STATE] TCP speed :", np.array(tcp_speed))
    print("[STATE] Joints q  :", np.array(q))

def main():
    print(f"Connecting to UR5e @ {ROBOT_IP} ...")

    # 建议：receive先连（读状态）
    rtde_r = RTDEReceive(ROBOT_IP)

    # control再连（发指令）
    rtde_c = RTDEControl(ROBOT_IP, CONTROL_FREQUENCY)

    try:
        print("Connected. Reading state...")
        print_state(rtde_r)

        # 1) 让你确认“能发指令”：stopJ一次（不会动，但能验证控制通道）
        print("\n[TEST] stopJ(2.0) ...")
        rtde_c.stopJ(2.0)
        time.sleep(0.2)
        print("[OK] stopJ executed.")

        # 2) 可选：进入 freedrive（手拖机器人），验证你能切模式（很常用）
        #   你可以先手动拖一下，然后按回车退出 freedrive
        print("\n[TEST] Freedrive mode ON. You can hand-guide the robot now.")
        rtde_c.teachMode()
        input("Press ENTER to exit freedrive (teach mode) ...")
        rtde_c.endTeachMode()
        print("[OK] Freedrive OFF.")

        # 3) 可选：一个非常小的 moveJ 测试（默认注释，避免误动）
        #    如果你要开，就把下面的注释去掉，并保证周围安全、速度很低。
        #
        # print("\n[TEST] Small moveJ (VERY SLOW) ...")
        # q_now = np.array(rtde_r.getActualQ(), dtype=np.float64)
        # q_target = q_now.copy()
        # q_target[0] += np.deg2rad(5.0)   # 仅让关节1转5度（很小）
        # ok = rtde_c.moveJ(q_target.tolist(), SAFE_SPEED, SAFE_ACCEL)
        # print("[OK] moveJ done:", ok)
        # print_state(rtde_r)

        print("\nAll basic checks passed. You're ready for next steps.")

    except KeyboardInterrupt:
        print("\n[CTRL+C] Stopping robot...")
        try:
            rtde_c.stopScript()
        except Exception:
            pass

    finally:
        # 清理连接
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
