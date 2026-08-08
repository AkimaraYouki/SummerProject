#!/usr/bin/env python3
"""검증된 부호로 실기를 READY 자세까지 천천히 이동시킨다.

    ssh -t parksuho@192.168.137.7 'python3 ~/goto_ready.py'          # 이동
    ssh    parksuho@192.168.137.7 'python3 ~/goto_ready.py --dry-run' # 계산만
    ssh -t parksuho@192.168.137.7 'python3 ~/goto_ready.py --home'    # 2048 복귀

부호/영점은 2026-08-06 에 실기에서 14축 전부 눈으로 확인한 값이다
(joint_calibration.json, hardware_map.py). 그 전 버전은 ID 순서를 추측했고
그래서 3축이 기구 스토퍼에 걸렸다 — 여기 박힌 표가 그 사고의 산물이다.

안전:
  * 전류 상한 (기본 400 unit = 1.08 A, 스톨 2.3 A 의 47 %).
  * **로컬 스톨 감지.** 매 스텝에서 추종 오차를 보고, 어느 축이든 STALL_TICK 이상
    벌어진 채 STALL_HOLD 초를 넘기면 그 자리에서 전 축 목표를 현재 위치로 덮어써
    미는 힘을 없앤다. 원격에서 사람이 "멈춰" 를 보내는 것보다 항상 빠르다.
  * smoothstep 보간 — 시작과 끝의 가속도가 0.
  * Ctrl+C 즉시 현재 위치 홀드.
"""

import argparse
import math
import sys
import time

from dynamixel_sdk import (PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
                           COMM_SUCCESS)

BAUD = 1_000_000
TICK_RAD = 2.0 * math.pi / 4096.0
CENTER = 2048

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_TEMP = 146
MODE_CURRENT_POSITION = 5
CURRENT_UNIT_MA = 2.69

STALL_TICK = 120      # 10.5°. 이만큼 벌어지면 추종 실패로 본다
STALL_HOLD = 1.0      # 초. 순간적인 지연은 무시하고 지속될 때만 중단

# (관절, ID, 부호, READY rad, URDF 한계 rad) — 전부 실기 확인 완료.
JOINTS = [
    ("left_hip_yaw",     3, -1,  0.0003, 0.5236),
    ("left_hip_roll",    8, -1,  0.0213, 0.4363),
    ("left_hip_pitch",   9, -1,  0.9910, None),
    ("left_knee",       10, +1, -1.7852, 2.0944),
    ("left_ankle",      11, +1,  0.8647, 1.5708),
    ("neck_pitch",       2, +1,  0.0000, None),
    ("head_pitch",      12, -1,  0.0000, 0.7854),
    ("head_yaw",        13, -1,  0.0000, 2.7925),
    ("head_roll",       14, -1,  0.0000, 0.5236),
    ("right_hip_yaw",    1, -1, -0.0005, 0.5236),
    ("right_hip_roll",   4, +1, -0.0092, 0.4363),
    ("right_hip_pitch",  5, +1,  1.0114, None),
    ("right_knee",       6, -1,  1.8163, 2.0944),
    ("right_ankle",      7, -1, -0.8754, 1.5708),
]
# hip_pitch / neck_pitch 는 좌우 비대칭 한계라 따로 준다 (lower, upper).
ASYM = {"left_hip_pitch": (-0.5236, 1.2217), "right_hip_pitch": (-0.5236, 1.2217),
        "neck_pitch": (-0.3491, 1.1345)}


def limits(name, sign, sym):
    lo_r, hi_r = ASYM[name] if name in ASYM else (-sym, sym)
    a = CENTER + sign * lo_r / TICK_RAD
    b = CENTER + sign * hi_r / TICK_RAD
    return int(round(min(a, b))), int(round(max(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--current", type=int, default=400,
                    help="Goal Current 상한 (1 unit = 2.69 mA). 기본 400 = 1.08 A")
    ap.add_argument("--rate", type=float, default=50.0, help="목표 갱신 주파수 Hz")
    ap.add_argument("--pvel", type=int, default=0,
                    help="서보 Profile Velocity. 0 = 속도 제한 없음 (보간이 속도를 정한다)")
    ap.add_argument("--check-every", type=int, default=1,
                    help="몇 루프마다 위치를 읽어 스톨을 볼지. 0 = 안 읽음(개루프). "
                         "57600 에서 14축 SyncRead 는 64 ms 라 읽기를 섞으면 루프가 "
                         "그만큼 느려진다 — 쓰기만 하면 13 ms(75 Hz)")
    ap.add_argument("--stall-tick", type=int, default=STALL_TICK,
                    help="추종 오차 몇 tick 부터 스톨로 볼지. 빠르게 갈수록 지연이 커지니 올린다")
    ap.add_argument("--home", action="store_true", help="READY 대신 2048 로 복귀")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    port = PortHandler(args.port)
    packet = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(BAUD):
        raise SystemExit(f"포트 열기 실패: {args.port}")

    start, goal, names = {}, {}, {}
    print(f"{'ID':>3} {'joint':16s} {'현재':>6s} {'목표':>6s} {'이동':>7s} {'도':>7s}")
    for name, i, sign, rad, sym in JOINTS:
        pos, r, e = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
        if r != COMM_SUCCESS or e:
            print(f"{i:3d} {name:16s} 응답 없음 — 중단한다 (전원/배선 확인)")
            port.closePort()
            raise SystemExit(1)
        lo, hi = limits(name, sign, sym)
        tgt = CENTER if args.home else int(round(CENTER + sign * rad / TICK_RAD))
        tgt = max(lo, min(hi, tgt))
        start[i], goal[i], names[i] = pos, tgt, name
        print(f"{i:3d} {name:16s} {pos:6d} {tgt:6d} {tgt - pos:+7d} "
              f"{math.degrees((tgt - pos) * TICK_RAD):+7.1f}")

    biggest = max(abs(goal[i] - start[i]) for i in start)
    print(f"\n최대 이동 {biggest} tick ({math.degrees(biggest * TICK_RAD):.1f}°), "
          f"{args.duration:.0f}초, 평균 {math.degrees(biggest * TICK_RAD) / args.duration:.1f} deg/s")
    print(f"전류 상한 {args.current * CURRENT_UNIT_MA / 1000:.2f} A, "
          f"스톨 감지 {args.stall_tick} tick 이 {STALL_HOLD}초 지속되면 자동 중단")
    print(f"목표 갱신 {args.rate:.0f} Hz, Profile Velocity "
          f"{'0 (속도 제한 없음)' if args.pvel == 0 else args.pvel}")
    if args.check_every == 0:
        print("** 개루프: 이동 중 스톨 감지 없음. 전류 상한만이 보호 장치다. **")
    if args.dry_run:
        port.closePort()
        print("--dry-run: 움직이지 않고 종료")
        return
    if sys.stdin.isatty():
        input("\n로봇을 받친 뒤 Enter (중단은 Ctrl+C): ")

    for i in start:
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
        packet.write1ByteTxRx(port, i, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
        packet.write4ByteTxRx(port, i, ADDR_PROFILE_ACCEL, 0 if args.pvel == 0 else 10)
        packet.write4ByteTxRx(port, i, ADDR_PROFILE_VELOCITY, args.pvel)
        packet.write4ByteTxRx(port, i, ADDR_GOAL_POSITION, start[i] & 0xFFFFFFFF)
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)
        packet.write2ByteTxRx(port, i, ADDR_GOAL_CURRENT, args.current)

    def hold_here(msg):
        print(f"\n{msg} — 전 축 목표를 현재 위치로 덮어써 미는 힘을 없앤다.")
        for i in start:
            p, _, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
            packet.write4ByteTxRx(port, i, ADDR_GOAL_POSITION, p & 0xFFFFFFFF)

    sync = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 4)
    # 14축을 개별로 읽으면 57600 baud 에서 왕복만 30 ms 넘게 먹어 루프가 20 Hz 를
    # 못 넘긴다. 그게 곧 목표 갱신 간격이라 빠르게 움직이면 계단처럼 끊긴다.
    # SyncRead 는 한 트랜잭션이라 50 Hz 이상이 나온다.
    reader = GroupSyncRead(port, packet, ADDR_PRESENT_POSITION, 4)
    ids = list(start)
    for i in ids:
        reader.addParam(i)
    since = {i: None for i in ids}
    dt = 1.0 / args.rate
    loops = 0
    t0 = time.time()
    try:
        while True:
            tick_t0 = time.time()
            s = min(1.0, (tick_t0 - t0) / args.duration)
            a = s * s * (3 - 2 * s)          # smoothstep
            sync.clearParam()
            want = {}
            for i in ids:
                v = int(round(start[i] + a * (goal[i] - start[i])))
                want[i] = v
                sync.addParam(i, v.to_bytes(4, "little", signed=False))
            sync.txPacket()

            loops += 1
            do_read = args.check_every > 0 and loops % args.check_every == 0
            if do_read and reader.txRxPacket() == COMM_SUCCESS:
                now = time.time()
                for i in ids:
                    if not reader.isAvailable(i, ADDR_PRESENT_POSITION, 4):
                        continue
                    p = reader.getData(i, ADDR_PRESENT_POSITION, 4)
                    if p > 2147483647:
                        p -= 4294967296
                    if abs(p - want[i]) > args.stall_tick:
                        if since[i] is None:
                            since[i] = now
                        elif now - since[i] > STALL_HOLD:
                            hold_here(f"!! {names[i]} (ID {i}) 추종 오차 {p - want[i]:+d} tick 이 "
                                      f"{STALL_HOLD}초 지속")
                            return
                    else:
                        since[i] = None

            if s >= 1.0:
                break
            sleep = dt - (time.time() - tick_t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        hold_here("Ctrl+C")
        return
    finally:
        port.closePort() if False else None

    elapsed = time.time() - t0
    print(f"\n이동 {elapsed:.2f}초, 루프 {loops}회, 실측 {loops / elapsed:.0f} Hz")
    time.sleep(1.5)
    print(f"\n{'ID':>3} {'joint':16s} {'목표':>6s} {'실제':>6s} {'오차':>6s} {'도':>7s} "
          f"{'전류mA':>7s} {'온도':>5s}")
    bad = []
    for i in ids:
        p, _, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
        c, _, _ = packet.read2ByteTxRx(port, i, ADDR_PRESENT_CURRENT)
        t, _, _ = packet.read1ByteTxRx(port, i, ADDR_PRESENT_TEMP)
        c = c - 65536 if c and c > 32767 else (c or 0)
        err = p - goal[i]
        if abs(err) > 30:
            bad.append(names[i])
        print(f"{i:3d} {names[i]:16s} {goal[i]:6d} {p:6d} {err:+6d} "
              f"{math.degrees(err * TICK_RAD):+7.1f} {abs(c) * CURRENT_UNIT_MA:7.0f} {t:5d}")
    print("\n" + ("전 축 도달" if not bad else f"오차 큰 축: {', '.join(bad)}"))
    print("토크 유지 중. 복귀는 --home, 힘 빼려면 home_position.py --release")
    port.closePort()


if __name__ == "__main__":
    main()
