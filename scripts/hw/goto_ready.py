#!/usr/bin/env python3
"""실기를 정책의 READY 자세까지 천천히 이동시킨다.

    ssh -t parksuho@192.168.137.7 'python3 ~/goto_ready.py'                # 최신 정책의 READY 로
    ssh -t parksuho@192.168.137.7 'python3 ~/goto_ready.py --policy ~/policy_v46'
    ssh    parksuho@192.168.137.7 'python3 ~/goto_ready.py --dry-run'      # 계산만
    ssh -t parksuho@192.168.137.7 'python3 ~/goto_ready.py --home'         # 2048 복귀

## READY 는 어디서 오는가

**정책 폴더의 `policy.meta.json` 이다.** 인자를 안 주면 `~/policy*/` 중 메타가
가장 최근인 것을 자동으로 고른다. 아래 `FALLBACK_READY` 는 메타가 하나도 없을
때만 쓰는 2026-08-08 자 값이고, 지금은 학습 설정이 여러 번 바뀌어 **한참
어긋나 있다** (v46 기준 무릎 25.6도, 발목 13.6도, 고관절 11.9도 차이).

READY 가 어긋나면 두 가지가 동시에 깨진다. 정책이 보는 관절각 관측이 통째로
밀리고, rl_walk 가 시작하면서 그 차이를 한 스텝에 메우려 들어 로봇이 뚝
떨어진다. 2026-08-12 에 "시작하자마자 앞으로 꼬꾸라진다" 던 것이 이거였다.

## 부호·ID·관절한계

`rustypot_hwi.py` 에서 가져온다. **여기서 다시 정의하지 않는다.**

2026-08-12 에 이 파일이 `TICK_RAD` 를 자기 것으로 들고 있다가 사고가 났다.
젯슨의 사본만 `4.0*pi/4096` 으로 되어 있었는데 (맞는 값은 `2.0*pi/4096`,
XM430 은 4096 tick 이 1 회전이다) 그래서 **모든 목표각이 절반으로 줄어**
있었다. v46 무릎 -76.7도 를 명령하면 실제로는 -38.3도 에서 멈췄다. 다른
파일 넷은 전부 맞았는데 이 파일만 틀렸다 — 표를 복제한 대가다.

## 안전

  * 전류 상한 (기본 400 unit = 1.08 A, 스톨 2.3 A 의 47 %).
  * **로컬 스톨 감지.** 매 스텝에서 추종 오차를 보고, 어느 축이든 STALL_TICK 이상
    벌어진 채 STALL_HOLD 초를 넘기면 그 자리에서 전 축 목표를 현재 위치로 덮어써
    미는 힘을 없앤다. 원격에서 사람이 "멈춰" 를 보내는 것보다 항상 빠르다.
  * smoothstep 보간 — 시작과 끝의 가속도가 0.
  * Ctrl+C 즉시 현재 위치 홀드.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

from dynamixel_sdk import (PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
                           COMM_SUCCESS)

sys.path.insert(0, os.path.expanduser("~"))
# 부호·ID·관절한계·tick 변환의 단일 출처. 위 독스트링의 사고 기록 참고.
from rustypot_hwi import (  # noqa: E402
    BAUD,
    BY_NAME,
    CENTER,
    CURRENT_UNIT_MA,
    NAMES,
    TICK_RAD,
    tick_of,
)

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

STALL_TICK = 120      # 10.5°. 이만큼 벌어지면 추종 실패로 본다
STALL_HOLD = 1.0      # 초. 순간적인 지연은 무시하고 지속될 때만 중단

# 정책 메타를 하나도 못 찾았을 때만 쓴다. 2026-08-08 자 — 지금 정책과 다르다.
FALLBACK_READY = {
    "left_hip_yaw": 0.0003, "left_hip_roll": 0.0213, "left_hip_pitch": 0.9910,
    "left_knee": -1.7852, "left_ankle": 0.8647,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.0005, "right_hip_roll": -0.0092, "right_hip_pitch": 1.0114,
    "right_knee": 1.8163, "right_ankle": -0.8754,
}


def newest_policy_meta():
    """`ready_joint_pos` 가 들어 있는 policy.meta.json 중 가장 최근 것."""
    best, best_mt = None, -1.0
    for p in glob.glob(os.path.expanduser("~/policy*/policy.meta.json")):
        try:
            if not (json.load(open(p)) or {}).get("ready_joint_pos"):
                continue
        except Exception:
            continue
        mt = os.path.getmtime(p)
        if mt > best_mt:
            best, best_mt = p, mt
    return best


def load_ready(policy_arg):
    """(READY dict, 출처 설명) 을 준다."""
    if policy_arg == "-":
        return dict(FALLBACK_READY), "FALLBACK_READY (2026-08-08 하드코딩)"
    mp = (os.path.join(os.path.expanduser(policy_arg), "policy.meta.json")
          if policy_arg else newest_policy_meta())
    if mp is None:
        print("!! policy.meta.json 을 하나도 못 찾았다 — 2026-08-08 하드코딩 값을 쓴다.")
        print("   지금 정책과 다를 가능성이 높다. odm onnx 로 메타를 다시 뽑을 것.")
        return dict(FALLBACK_READY), "FALLBACK_READY (메타 없음)"
    if not os.path.exists(mp):
        raise SystemExit(f"policy.meta.json 이 없다: {mp}")
    meta = json.load(open(mp))
    ready = meta.get("ready_joint_pos") or {}
    if not ready:
        raise SystemExit(f"{mp} 에 ready_joint_pos 가 없다 — odm onnx 로 다시 뽑을 것")
    missing = [n for n in NAMES if n not in ready]
    if missing:
        raise SystemExit(f"메타에 빠진 관절: {missing}")
    return ready, f"{mp}  (run {meta.get('run','?')} iter {meta.get('iter','?')})"


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
                         "읽기를 섞으면 SyncRead 왕복만큼 루프가 느려진다")
    ap.add_argument("--stall-tick", type=int, default=STALL_TICK,
                    help="추종 오차 몇 tick 부터 스톨로 볼지. 빠르게 갈수록 지연이 커지니 올린다")
    ap.add_argument("--policy", default=None, metavar="DIR",
                    help="정책 폴더(예: ~/policy_v46). 생략하면 ~/policy*/ 중 메타가 "
                         "가장 최근인 것을 자동으로 고른다. '-' 를 주면 하드코딩 값을 쓴다.")
    ap.add_argument("--home", action="store_true", help="READY 대신 2048 로 복귀")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ready, src = load_ready(args.policy)
    print(f"READY 출처: {src}")
    diff = [(n, FALLBACK_READY[n], ready[n]) for n in NAMES
            if abs(FALLBACK_READY[n] - ready[n]) > 1e-4]
    if diff and not src.startswith("FALLBACK"):
        print("  2026-08-08 하드코딩 표와 다른 축:")
        for n, o, v in diff:
            print(f"    {n:16} {math.degrees(o):+7.2f}° -> {math.degrees(v):+7.2f}°"
                  f"   ({math.degrees(v - o):+.2f}°)")

    port = PortHandler(args.port)
    packet = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(BAUD):
        raise SystemExit(f"포트 열기 실패: {args.port}")

    start, goal, names = {}, {}, {}
    print(f"\n{'ID':>3} {'joint':16s} {'현재':>6s} {'목표':>6s} {'이동':>7s} {'도':>7s} {'READY°':>8s}")
    for name in NAMES:
        i = BY_NAME[name][1]
        pos, r, e = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
        if r != COMM_SUCCESS or e:
            print(f"{i:3d} {name:16s} 응답 없음 — 중단한다 (전원/배선 확인)")
            port.closePort()
            raise SystemExit(1)
        # tick_of 가 URDF 한계 클램프까지 해 준다 (rustypot_hwi 와 같은 변환).
        tgt = CENTER if args.home else tick_of(name, ready[name])
        start[i], goal[i], names[i] = pos, tgt, name
        clamped = "" if args.home or tgt == int(round(
            CENTER + BY_NAME[name][2] * ready[name] / TICK_RAD)) else "  <한계로 잘림"
        print(f"{i:3d} {name:16s} {pos:6d} {tgt:6d} {tgt - pos:+7d} "
              f"{math.degrees((tgt - pos) * TICK_RAD):+7.1f} "
              f"{math.degrees(ready[name]):+8.2f}{clamped}")

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

    ids = [BY_NAME[n][1] for n in NAMES]
    for i in ids:
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
        packet.write1ByteTxRx(port, i, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
        packet.write4ByteTxRx(port, i, ADDR_PROFILE_ACCEL, 0 if args.pvel == 0 else 10)
        packet.write4ByteTxRx(port, i, ADDR_PROFILE_VELOCITY, args.pvel)
        packet.write4ByteTxRx(port, i, ADDR_GOAL_POSITION, start[i] & 0xFFFFFFFF)
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)
        packet.write2ByteTxRx(port, i, ADDR_GOAL_CURRENT, args.current)

    def hold_here(msg):
        print(f"\n{msg} — 전 축 목표를 현재 위치로 덮어써 미는 힘을 없앤다.")
        for i in ids:
            p, _, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
            packet.write4ByteTxRx(port, i, ADDR_GOAL_POSITION, p & 0xFFFFFFFF)

    sync = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 4)
    # 14축을 개별로 읽으면 왕복만 30 ms 넘게 먹어 루프가 20 Hz 를 못 넘긴다.
    # 그게 곧 목표 갱신 간격이라 빠르게 움직이면 계단처럼 끊긴다.
    # SyncRead 는 한 트랜잭션이라 50 Hz 이상이 나온다.
    reader = GroupSyncRead(port, packet, ADDR_PRESENT_POSITION, 4)
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
