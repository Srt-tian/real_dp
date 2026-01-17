from serial import Serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 1000000         # 先用 115200；不通再换 57600/1_000_000 等
SERVO_ID = 1           # 常见默认 ID=1（不确定就用下面的“扫描”小节）
TIMEOUT = 0.1

# ===== 协议封装 =====
def checksum(servo_id: int, length: int, instruction: int, params: bytes) -> int:
    s = (servo_id + length + instruction + sum(params)) & 0xFF
    return (~s) & 0xFF  # ~ (sum) 取反，再取低8位

def build_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    length = len(params) + 2
    chk = checksum(servo_id, length, instruction, params)
    return bytes([0xFF, 0xFF, servo_id & 0xFF, length & 0xFF, instruction & 0xFF]) + params + bytes([chk])

def read_response(ser: Serial) -> bytes:
    """读取一帧返回：FF FF ID LEN ERR PARAMS... CHK"""
    # 读到帧头
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < TIMEOUT:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if len(buf) >= 2 and buf[-2] == 0xFF and buf[-1] == 0xFF:
            break
    if len(buf) < 2:
        return None

    # 接着读 ID/LEN
    rest = ser.read(2)
    if len(rest) < 2:
        return None
    servo_id = rest[0]
    length = rest[1]

    # 返回帧剩余：ERR + (LEN-2 个参数) + CHK  => 共 LEN 字节
    payload = ser.read(length)
    if len(payload) < length:
        return None

    return bytes([0xFF, 0xFF, servo_id, length]) + payload

# ===== 常用指令 =====
def ping(ser: Serial, servo_id: int) -> bool:
    pkt = build_packet(servo_id, 0x01, b"")   # PING
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    resp = read_response(ser)
    return resp is not None and len(resp) >= 6 and resp[2] == servo_id

def read_u16(ser: Serial, servo_id: int, addr: int) -> int:
    # READ: params = [addr, read_len]
    pkt = build_packet(servo_id, 0x02, bytes([addr & 0xFF, 0x02]))
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    resp = read_response(ser)
    if not resp:
        return None
    # resp: FF FF ID LEN ERR D0 D1 CHK
    if resp[4] != 0x00:  # ERR != 0
        return None
    d0 = resp[5]
    d1 = resp[6]
    return d0 | (d1 << 8)  # 低字节在前（示例也是 low->high）

def write_goal_pos(ser: Serial, servo_id: int, pos: int, t_ms: int = 0, speed: int = 0):
    """
    WRITE 到 0x2A 起的 6 字节：pos(2) + time(2) + speed(2)
    示例里 pos=2048 (0x0800) 发送为 00 08（低字节在前）:contentReference[oaicite:7]{index=7}
    """
    pos = int(pos) & 0xFFFF
    t_ms = int(t_ms) & 0xFFFF
    speed = int(speed) & 0xFFFF

    params = bytes([
        0x2A,                # 起始地址
        pos & 0xFF, (pos >> 8) & 0xFF,
        t_ms & 0xFF, (t_ms >> 8) & 0xFF,
        speed & 0xFF, (speed >> 8) & 0xFF,
    ])
    pkt = build_packet(servo_id, 0x03, params)  # WRITE
    ser.reset_input_buffer()
    ser.write(pkt)
    ser.flush()
    _ = read_response(ser)  # 非广播 ID 一般会回状态包（可忽略）

def main():
    with Serial(PORT, BAUD, timeout=TIMEOUT) as ser:
        print("Open:", PORT, "baud=", BAUD)

        ok = ping(ser, SERVO_ID)
        print("PING", SERVO_ID, "=>", ok)
        if not ok:
            print("PING 不通：请换 BAUD/ID，或检查 TTL->总线模块方向控制/供电/信号线")
            return

        # 读当前位置（示例地址 0x38）:contentReference[oaicite:8]{index=8}
        cur = read_u16(ser, SERVO_ID, 0x38)
        print("Current pos(raw) =", cur)

        # 转到 90 度（STS 类常用 0~4095 对应 0~360°，90°≈1024）
        goal = 4095
        print("Move to", goal)
        write_goal_pos(ser, SERVO_ID, pos=goal, t_ms=0, speed=0)

        time.sleep(0.5)
        cur2 = read_u16(ser, SERVO_ID, 0x38)
        print("Current pos(raw) =", cur2)

if __name__ == "__main__":
    main()
