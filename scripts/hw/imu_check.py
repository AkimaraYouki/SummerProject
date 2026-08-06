#!/usr/bin/env python3
"""BNO055 를 깨우고, 칩이 어떤 자세로 붙어 있는지 실측으로 알아낸다 (Jetson).

    ssh    parksuho@192.168.137.7 'python3 ~/imu_check.py --probe'
    ssh -t parksuho@192.168.137.7 'python3 ~/imu_check.py --calibrate'
    ssh    parksuho@192.168.137.7 'python3 ~/imu_check.py --verify'

────────────────────────────────────────────────────────────────────────
몸통 프레임 = **+x 앞 / +y 좌 / +z 위** (표준 오른손 프레임)
────────────────────────────────────────────────────────────────────────
독립적으로 두 번 확인했다:
  1) 전진 명령에서 디딤발이 몸통 기준 -29.8 mm 흐른다 (발이 고정이니 몸통이 +x
     로 전진). 후진 명령에서는 +42.9 mm 로 뒤집힌다.
  2) 발목 프레임에서 발이 +x 로 63.8 mm, -x 로 40.1 mm 뻗는다 -- 발가락이 앞,
     뒤꿈치가 뒤인 정상 형상.
  좌우는 URDF 로 자명하다: left_foot y=+90.9 mm, right_foot y=-91.7 mm.

⚠️ `docs/handoff/project_hardware_bringup_2026-08-06.md` 는 "-x 가 앞" 이라고
적었는데 **틀렸다.** 그 근거였던 "머리를 세우면 CoM 이 -x 로 가고 앞으로 넘어졌다"
에서 CoM 이동량(-14.26 mm)은 재현되지만, 그 전도는 정적 전복이 아니라 정책이
무너진 동적 결과라 방향 판정의 근거가 못 된다.

────────────────────────────────────────────────────────────────────────
시뮬이 기대하는 값 (joystick_env.py `_get_observations`)
────────────────────────────────────────────────────────────────────────
  gyro  = imu.data.ang_vel_b   [rad/s]  몸통 프레임
  accel = imu.data.lin_acc_b   [m/s^2]  몸통 프레임, **중력 포함**
          (Isaac Lab 이 gravity_bias=(0,0,+9.81) 을 더한다 -> 실제 가속도계가
           재는 비력과 같은 규약이다. 직립 정지 시 (0,0,+9.81))
  projected_gravity_b          직립 시 (0,0,-1). = -normalize(accel) (정지 시)

BNO055 의 축맵 레지스터(0x41/0x42)는 전원을 끊으면 날아간다. 그래서 칩에 굽지
않고 **소프트웨어 리맵**으로 두고, 결과를 레포의 imu_calibration.json 에 남긴다.
"""

import argparse
import json
import math
import os
import sys
import time

from smbus2 import SMBus, i2c_msg

BUS = 7
ADDR = 0x28

REG_CHIP_ID = 0x00
REG_ACC_ID = 0x01
REG_MAG_ID = 0x02
REG_GYR_ID = 0x03
REG_ACC_DATA = 0x08
REG_GYR_DATA = 0x14
REG_GRAVITY = 0x2E
REG_UNIT_SEL = 0x3B
REG_OPR_MODE = 0x3D
REG_PWR_MODE = 0x3E
REG_SYS_TRIGGER = 0x3F
REG_CALIB_STAT = 0x35
REG_AXIS_MAP_CONFIG = 0x41
REG_AXIS_MAP_SIGN = 0x42

MODE_CONFIG = 0x00
MODE_AMG = 0x07
MODE_IMU = 0x08
MODE_NDOF = 0x0C
MODE_NAME = {0x00: "CONFIG(출력없음)", 0x07: "AMG(원시)", 0x08: "IMU(자기제외융합)",
             0x0B: "NDOF_FMC_OFF", 0x0C: "NDOF(융합)"}

ACC_LSB = 100.0   # 1 m/s^2 = 100 LSB (UNIT_SEL 의 acc 를 m/s^2 로 둘 때)
GYR_LSB = 16.0    # 1 dps = 16 LSB (기본 dps 단위)
GRV_LSB = 100.0

CAL_JSON = os.path.expanduser("~/imu_calibration.json")

# 판정용 자세. (이름, 안내문, 몸통 프레임에서 가속도계가 읽어야 할 방향)
# 정지 가속도계는 비력을 잰다: a_body = R^T * (0,0,+9.81).
POSES = [
    ("upright", "로봇을 **똑바로 세워서** 평소 서 있는 자세로 두세요", (0.0, 0.0, +1.0)),
    ("nose_down", "로봇을 **코가 바닥을 보도록** 앞으로 90° 숙이세요", (-1.0, 0.0, 0.0)),
    ("roll_right", "로봇을 **오른쪽으로 눕혀서** 왼쪽 옆구리가 위를 보게 하세요", (0.0, +1.0, 0.0)),
]
AXIS_NAME = ["x(앞)", "y(좌)", "z(위)"]


def rd(bus, reg, n=1):
    w = i2c_msg.write(ADDR, [reg])
    r = i2c_msg.read(ADDR, n)
    bus.i2c_rdwr(w, r)
    return list(r)


def wr(bus, reg, val):
    bus.write_byte_data(ADDR, reg, val)


def s16(lo, hi):
    v = lo | (hi << 8)
    return v - 65536 if v & 0x8000 else v


def vec(bus, reg, lsb):
    d = rd(bus, reg, 6)
    return tuple(s16(d[i], d[i + 1]) / lsb for i in (0, 2, 4))


def set_mode(bus, mode):
    cur = rd(bus, REG_OPR_MODE)[0] & 0x0F
    if cur == mode:
        return
    wr(bus, REG_OPR_MODE, MODE_CONFIG)
    time.sleep(0.03)          # 운영->CONFIG 19 ms
    if mode != MODE_CONFIG:
        wr(bus, REG_OPR_MODE, mode)
        time.sleep(0.02)      # CONFIG->운영 7 ms


def probe(bus):
    ids = rd(bus, REG_CHIP_ID, 4)
    ok = ids == [0xA0, 0xFB, 0x32, 0x0F]
    print(f"CHIP {ids[0]:#04x} ACC {ids[1]:#04x} MAG {ids[2]:#04x} GYR {ids[3]:#04x}"
          f"  {'-> BNO055 확인' if ok else '-> ⚠️ 예상과 다름'}")
    m = rd(bus, REG_OPR_MODE)[0] & 0x0F
    print(f"OPR_MODE {m:#04x} = {MODE_NAME.get(m, '?')}")
    print(f"UNIT_SEL {rd(bus, REG_UNIT_SEL)[0]:#04x}  "
          f"AXIS_MAP_CONFIG {rd(bus, REG_AXIS_MAP_CONFIG)[0]:#04x}  "
          f"AXIS_MAP_SIGN {rd(bus, REG_AXIS_MAP_SIGN)[0]:#04x}")
    c = rd(bus, REG_CALIB_STAT)[0]
    print(f"CALIB_STAT {c:#04x}  sys{(c>>6)&3} gyr{(c>>4)&3} acc{(c>>2)&3} mag{c&3}  (각 0~3)")
    return ok


def wake(bus, mode):
    wr(bus, REG_PWR_MODE, 0x00)      # NORMAL
    time.sleep(0.02)
    set_mode(bus, MODE_CONFIG)
    wr(bus, REG_UNIT_SEL, 0x00)      # acc m/s^2, gyr dps, euler deg
    time.sleep(0.01)
    set_mode(bus, mode)
    time.sleep(0.05)


def read_all(bus):
    a = vec(bus, REG_ACC_DATA, ACC_LSB)
    g = vec(bus, REG_GYR_DATA, GYR_LSB)
    g = tuple(v * math.pi / 180.0 for v in g)
    return a, g


def avg_acc(bus, n=40):
    s = [0.0, 0.0, 0.0]
    for _ in range(n):
        a, _ = read_all(bus)
        s = [s[i] + a[i] for i in range(3)]
        time.sleep(0.02)
    return tuple(v / n for v in s)


def norm(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / m for c in v)


def snap_permutation(rows):
    """3x3 행렬을 가장 가까운 부호 있는 치환행렬로. (행렬, 최대잔차) 를 돌려준다."""
    used, out, resid = set(), [], 0.0
    order = sorted(range(3), key=lambda i: -max(abs(c) for c in rows[i]))
    snapped = [None] * 3
    for i in order:
        best = max((j for j in range(3) if j not in used), key=lambda j: abs(rows[i][j]))
        used.add(best)
        s = 1.0 if rows[i][best] >= 0 else -1.0
        r = [0.0, 0.0, 0.0]
        r[best] = s
        snapped[i] = r
        resid = max(resid, max(abs(rows[i][j] - r[j]) for j in range(3)))
    return snapped, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=BUS)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mode", default="ndof", choices=["ndof", "amg", "imu", "config"])
    ap.add_argument("--out", default=CAL_JSON)
    args = ap.parse_args()

    mode = {"ndof": MODE_NDOF, "amg": MODE_AMG, "imu": MODE_IMU, "config": MODE_CONFIG}[args.mode]

    with SMBus(args.bus) as bus:
        if not probe(bus):
            print("⚠️ 칩 ID 가 BNO055 와 다르다. 주소/버스를 확인할 것.")
        if args.probe and not (args.calibrate or args.verify):
            wake(bus, mode)
            print(f"\n{args.mode.upper()} 로 깨웠다. 5회 읽기:")
            for _ in range(5):
                a, g = read_all(bus)
                print(f"  accel ({a[0]:+6.2f},{a[1]:+6.2f},{a[2]:+6.2f}) m/s²  "
                      f"|a|={math.sqrt(sum(c*c for c in a)):5.2f}   "
                      f"gyro ({g[0]:+6.3f},{g[1]:+6.3f},{g[2]:+6.3f}) rad/s")
                time.sleep(0.2)
            return

        wake(bus, mode)

        if args.calibrate:
            if not sys.stdin.isatty():
                raise SystemExit("TTY 가 필요하다 (사람이 로봇을 잡고 자세를 바꿔야 한다).\n"
                                 "  ssh -t parksuho@192.168.137.7 'python3 ~/imu_check.py --calibrate'")
            print("\n칩이 몸통에 어떤 자세로 붙어 있는지 세 자세로 알아낸다.")
            print("각 자세에서 **가만히** 들고 Enter. 흔들리면 값이 튄다.\n")
            meas = {}
            for name, guide, _ in POSES:
                input(f"[{name}] {guide} — 준비되면 Enter: ")
                a = avg_acc(bus)
                n = math.sqrt(sum(c * c for c in a))
                meas[name] = a
                print(f"    읽음 ({a[0]:+6.2f},{a[1]:+6.2f},{a[2]:+6.2f})  |a|={n:5.2f}"
                      f"{'  ⚠️ 9.81 에서 많이 벗어남 — 움직였을 수 있다' if abs(n-9.81)>1.5 else ''}")

            # 각 자세의 읽음값 = 그 자세에서 '위' 를 가리키는 방향을 IMU 프레임으로 본 것.
            # upright   -> 몸통 +z 를 IMU 프레임에서 본 것
            # nose_down -> 몸통 -x
            # roll_right-> 몸통 +y
            z_i = norm(meas["upright"])
            x_i = tuple(-c for c in norm(meas["nose_down"]))
            y_i = norm(meas["roll_right"])
            rows = [list(x_i), list(y_i), list(z_i)]   # a_trunk = M @ a_imu

            print("\n측정된 변환 (a_몸통 = M · a_IMU):")
            for i, r in enumerate(rows):
                print(f"  {AXIS_NAME[i]:8s} <- ({r[0]:+6.3f}, {r[1]:+6.3f}, {r[2]:+6.3f})")

            snapped, resid = snap_permutation(rows)
            print(f"\n가장 가까운 직각 축맵 (잔차 {resid:.3f}):")
            for i, r in enumerate(snapped):
                j = max(range(3), key=lambda k: abs(r[k]))
                print(f"  몸통 {AXIS_NAME[i]:8s} = {'-' if r[j] < 0 else '+'}IMU_{'xyz'[j]}")
            if resid > 0.25:
                print("  ⚠️ 잔차가 크다 — 칩이 축에 정렬돼 붙어 있지 않거나 자세가 부정확했다.")
                print("     자세를 더 정확히 잡고 다시 하거나, 스냅 대신 측정행렬을 그대로 쓸 것.")

            doc = {"matrix": snapped, "measured": rows, "residual": resid,
                   "frame": "+x forward, +y left, +z up",
                   "mode": args.mode, "bus": args.bus, "addr": ADDR,
                   "poses": {k: list(v) for k, v in meas.items()}}
            with open(args.out, "w") as f:
                json.dump(doc, f, indent=2)
            print(f"\n[저장] {args.out}")
            print("이 파일을 레포로 가져가서 source/open_duck_mini_isaaclab/imu_calibration.json 에 둘 것.")

        if args.verify:
            if not os.path.exists(args.out):
                raise SystemExit(f"{args.out} 이 없다. 먼저 --calibrate 를 돌릴 것.")
            with open(args.out) as f:
                M = json.load(f)["matrix"]
            print("\n리맵 적용 후 (로봇을 똑바로 세우고 보세요). 기대: accel≈(0,0,+9.81), "
                  "projected_gravity≈(0,0,-1)")
            for _ in range(10):
                a, g = read_all(bus)
                at = tuple(sum(M[i][j] * a[j] for j in range(3)) for i in range(3))
                gt = tuple(sum(M[i][j] * g[j] for j in range(3)) for i in range(3))
                pg = tuple(-c for c in norm(at))
                print(f"  accel ({at[0]:+6.2f},{at[1]:+6.2f},{at[2]:+6.2f})  "
                      f"gyro ({gt[0]:+6.3f},{gt[1]:+6.3f},{gt[2]:+6.3f})  "
                      f"proj_grav ({pg[0]:+5.2f},{pg[1]:+5.2f},{pg[2]:+5.2f})")
                time.sleep(0.3)


if __name__ == "__main__":
    main()
