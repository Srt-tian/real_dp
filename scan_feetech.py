# min_write_feetech.py
from serial import Serial

PORT = "/dev/ttyUSB0"  
BAUD = 115200
SERVO_ID = 1

def checksum(servo_id: int, length: int, inst: int, params: bytes) -> int:
    s = (servo_id + length + inst + sum(params)) & 0xFF
    return (~s) & 0xFF

def build_write_goal_pos_packet(servo_id: int, pos: int, t_ms: int = 0, speed: int = 0) -> bytes:
    # WRITE starting at 0x2A: pos(2) + time(2) + speed(2)
    pos = int(pos) & 0xFFFF
    t_ms = int(t_ms) & 0xFFFF
    speed = int(speed) & 0xFFFF

    params = bytes([
        0x2A,
        pos & 0xFF, (pos >> 8) & 0xFF,
        t_ms & 0xFF, (t_ms >> 8) & 0xFF,
        speed & 0xFF, (speed >> 8) & 0xFF,
    ])
    length = len(params) + 2
    chk = checksum(servo_id & 0xFF, length & 0xFF, 0x03, params)
    return bytes([0xFF, 0xFF, servo_id & 0xFF, length & 0xFF, 0x03]) + params + bytes([chk])

def arrive_goal(goal):
    pkt = build_write_goal_pos_packet(SERVO_ID, goal, t_ms=0, speed=0)
    print("TX HEX:", pkt.hex(" ").upper())

    with Serial(PORT, BAUD, timeout=0, write_timeout=0.5) as ser:
        # 不清输入缓存、不读回包，只写
        ser.write(pkt)
        ser.flush()

def control_open_state(state: int):
    if state == 0:
        arrive_goal(4000)
    elif state == 1:
        arrive_goal(3200)
    else:
        raise ValueError("state should be 0 (close) or 1 (open)")
    
def main():
    goal = 1000  

    pkt = build_write_goal_pos_packet(SERVO_ID, goal, t_ms=0, speed=0)
    print("TX HEX:", pkt.hex(" ").upper())

    with Serial(PORT, BAUD, timeout=0, write_timeout=0.5) as ser:
        # 不清输入缓存、不读回包，只写
        ser.write(pkt)
        ser.flush()

if __name__ == "__main__":
    main()
