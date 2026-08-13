#!/usr/bin/env python3
"""레퍼런스 보행을 실기에 재생한다 (Jetson 에서 실행).

    # 데스크탑에서 궤적을 뽑아 보낸다
    ~/Desktop/IsaacLab/isaaclab.sh -p scripts/hw/export_ref_gait.py --vx 0.15
    scp /tmp/ref_gait.json parksuho@192.168.137.7:~/
    scp scripts/hw/play_ref_gait.py parksuho@192.168.137.7:~/

    # Jetson 에서 (TTY 필요)
    ssh -t parksuho@192.168.137.7 'python3 ~/play_ref_gait.py --dry-run'
    ssh -t parksuho@192.168.137.7 'python3 ~/play_ref_gait.py --speed 0.25 --cycles 3'

────────────────────────────────────────────────────────────────────────
⚠️  로봇을 **매달아 놓고** 할 것. 바닥에 세우면 넘어진다.
────────────────────────────────────────────────────────────────────────
이건 개루프 재생이다. 균형 피드백이 없다 -- IMU 는 아직 CONFIG 모드로 꺼져
있고, 레퍼런스는 "이렇게 움직이면 걷는다" 는 궤적일 뿐 넘어짐을 복구하지 않는다.
정책이 하는 일이 바로 그 복구인데, 정책은 50 Hz 폐루프가 필요하고 지금 버스는
10.8 Hz 다 (bringup 문서 §4).

**왜 이건 지금 되는가**: 읽기를 안 하면 SyncWrite 만 남아 14축 13.4 ms → 70 Hz
가 나온다. 궤적이 미리 정해진 동작에는 충분하다. 대신 추종 오차를 못 보므로
**전류 제한이 유일한 보호**다 -- 관절이 걸려도 밀어붙이지 못하고 그냥 진다.

안전 설계:
  * 전류 상한 (기본 350 unit = 0.94 A, 스톨 2.3 A 의 41 %)
  * 궤적은 export 단계에서 URDF 한계 - 여유 로 클립됨. 여기서 tick 으로 한 번 더 클램프
  * 현재 자세 -> 사이클 시작 자세로 smoothstep 램프인 (급출발 없음)
  * Ctrl+C / SIGHUP / SIGTERM -> 즉시 현재 위치 홀드
  * TTY 가 아니면 거부 (사람이 보고 있어야 하는 동작이다)
"""

import argparse
import json
import math
import os
import signal
import sys
import time

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, COMM_SUCCESS

BAUD = 1_000_000
TICK_RAD = 2.0 * math.pi / 4096.0
CENTER = 2048

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
MODE_CURRENT_POSITION = 5
CURRENT_UNIT_MA = 2.69

# (관절, ID, 부호, URDF 하한, 상한) — 2026-08-06 실기 확인 완료.
# goto_ready.py 와 같은 표다. Jetson 에는 패키지가 없어 각 스크립트가 자립형이다.
JOINTS = [
    ("left_hip_yaw",     3, -1, -0.5236, 0.5236),
    ("left_hip_roll",    8, -1, -0.4363, 0.4363),
    ("left_hip_pitch",   9, -1, -0.5236, 1.2217),
    ("left_knee",       10, +1, -2.0944, 2.0944),
    ("left_ankle",      11, +1, -1.5708, 1.5708),
    ("neck_pitch",       2, +1, -0.3491, 1.1345),
    ("head_pitch",      12, -1, -0.7854, 0.7854),
    ("head_yaw",        13, -1, -2.7925, 2.7925),
    ("head_roll",       14, -1, -0.5236, 0.5236),
    ("right_hip_yaw",    1, -1, -0.5236, 0.5236),
    ("right_hip_roll",   4, +1, -0.4363, 0.4363),
    ("right_hip_pitch",  5, +1, -0.5236, 1.2217),
    ("right_knee",       6, -1, -2.0944, 2.0944),
    ("right_ankle",      7, -1, -1.5708, 1.5708),
]
BY_NAME = {j[0]: j for j in JOINTS}

_hold = {"stop": False}


def to_tick(name, rad):
    _, _, sign, lo, hi = BY_NAME[name]
    a = CENTER + sign * lo / TICK_RAD
    b = CENTER + sign * hi / TICK_RAD
    lo_t, hi_t = int(round(min(a, b))), int(round(max(a, b)))
    return max(lo_t, min(hi_t, int(round(CENTER + sign * rad / TICK_RAD))))


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def sync_write(packet, port, targets):
    """targets: {id: tick}. 4바이트 goal position 을 한 번에."""
    gw = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, 4)
    for i, t in targets.items():
        gw.addParam(i, bytes([t & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, (t >> 24) & 0xFF]))
    gw.txPacket()
    gw.clearParam()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.expanduser("~/ref_gait.json"))
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--speed", type=float, default=0.25,
                    help="사이클 속도 배율. 1.0 = 레퍼런스 원속도(0.54 s/사이클). "
                         "처음에는 0.25 로 보면서 올릴 것")
    ap.add_argument("--cycles", type=float, default=3.0, help="몇 사이클 재생할지")
    ap.add_argument("--rate", type=float, default=50.0, help="목표 갱신 Hz")
    ap.add_argument("--current", type=int, default=350,
                    help="Goal Current 상한 (1 unit = 2.69 mA). 기본 350 = 0.94 A")
    ap.add_argument("--rampin", type=float, default=3.0,
                    help="현재 자세에서 사이클 시작 자세까지 이동할 시간(초)")
    ap.add_argument("--dry-run", action="store_true", help="모터를 안 건드리고 계산만")
    args = ap.parse_args()

    with open(args.json) as f:
        doc = json.load(f)
    names, frames = doc["joint_names"], doc["frames"]
    period, dt = doc["period"], doc["dt"]
    cyc_time = period * dt / max(args.speed, 1e-6)

    # 실제로 움직이는 관절만 쓴다. 머리 4축은 레퍼런스에서 0 이라 건드릴 이유가
    # 없고, 쓰는 축이 적을수록 SyncWrite 가 짧아져 루프에 여유가 생긴다.
    moving = []
    for k, n in enumerate(names):
        span = max(r[k] for r in frames) - min(r[k] for r in frames)
        if span > math.radians(0.1):
            moving.append((k, n))

    print(f"[궤적] {args.json}")
    print(f"       명령 {doc['command']} · 주기 {period} × dt {dt} = {period*dt:.2f} s")
    print(f"       재생 속도 {args.speed}× -> {cyc_time:.2f} s/사이클 · {args.cycles} 사이클 "
          f"= {cyc_time*args.cycles:.1f} s")
    print(f"       움직이는 축 {len(moving)}개: {', '.join(n for _, n in moving)}")
    for k, n in moving:
        a = min(r[k] for r in frames)
        b = max(r[k] for r in frames)
        print(f"         {n:16s} {math.degrees(a):+7.1f}° .. {math.degrees(b):+7.1f}°"
              f"   tick {to_tick(n, a)} .. {to_tick(n, b)}")

    if args.dry_run:
        print("\n[dry-run] 모터를 건드리지 않았다.")
        return

    if not sys.stdin.isatty():
        raise SystemExit(
            "TTY 가 아니다. 로봇이 움직이는 동작이라 사람이 보고 있어야 한다.\n"
            "  ssh -t parksuho@192.168.137.7 'python3 ~/play_ref_gait.py ...'")

    print("\n⚠️  로봇이 매달려 있는지(발이 바닥에 안 닿는지) 확인할 것. "
          "바닥에 세우면 넘어진다.")
    if input("   매달려 있으면 'go' 입력: ").strip().lower() != "go":
        print("취소했다.")
        return

    port, packet = PortHandler(args.port), PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(BAUD):
        raise SystemExit(f"포트 열기 실패: {args.port}")

    ids = [BY_NAME[n][1] for _, n in moving]
    start = {}
    for (k, n) in moving:
        i = BY_NAME[n][1]
        pos, r, e = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
        if r != COMM_SUCCESS or e:
            port.closePort()
            raise SystemExit(f"{n}(ID {i}) 응답 없음 — 전원/배선 확인. 중단한다.")
        start[i] = pos

    # 전류제한 위치제어. Mode 3 은 Current Limit 이 EEPROM 이라 런타임 조절이 안 된다.
    for i in ids:
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
        packet.write1ByteTxRx(port, i, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
        packet.write2ByteTxRx(port, i, ADDR_GOAL_CURRENT, args.current)
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)
    print(f"전류 상한 {args.current * CURRENT_UNIT_MA / 1000:.2f} A · "
          f"토크 켬 {len(ids)}축")

    def stop(*_):
        _hold["stop"] = True
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)

    def frame_at(phase):
        """주기 안의 연속 위상 -> 관절각. 27 스텝을 선형보간해 감는다."""
        p = phase % period
        i0 = int(p) % period
        i1 = (i0 + 1) % period
        f = p - int(p)
        return {BY_NAME[n][1]: to_tick(n, frames[i0][k] * (1 - f) + frames[i1][k] * f)
                for k, n in moving}

    period_s = 1.0 / args.rate
    try:
        # 1) 램프인 — 현재 자세에서 위상 0 자세까지 smoothstep.
        goal0 = frame_at(0.0)
        big = max(abs(goal0[i] - start[i]) for i in ids)
        print(f"램프인 {args.rampin:.1f}s · 최대 이동 {big} tick "
              f"({math.degrees(big*TICK_RAD):.1f}°)")
        t0 = time.time()
        while not _hold["stop"]:
            u = (time.time() - t0) / max(args.rampin, 1e-6)
            if u >= 1.0:
                break
            s = smoothstep(u)
            sync_write(packet, port, {i: int(round(start[i] + (goal0[i] - start[i]) * s))
                                      for i in ids})
            time.sleep(period_s)

        # 2) 사이클 재생.
        print(f"재생 시작 — Ctrl+C 로 즉시 정지")
        t0, loops = time.time(), 0
        total = cyc_time * args.cycles
        while not _hold["stop"]:
            el = time.time() - t0
            if el >= total:
                break
            sync_write(packet, port, frame_at(el / cyc_time * period))
            loops += 1
            time.sleep(period_s)
        el = max(time.time() - t0, 1e-9)
        print(f"\n재생 종료 — {loops} 루프 / {el:.2f}s = {loops/el:.1f} Hz")
    finally:
        # 어떤 경로로 끝나든 현재 위치를 목표로 덮어써 미는 힘을 없앤다.
        held = {}
        for i in ids:
            pos, r, e = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
            if r == COMM_SUCCESS and not e:
                held[i] = pos
        if held:
            sync_write(packet, port, held)
        print(f"현재 위치 홀드 ({len(held)}축). 토크는 켜둔 채로 둔다 — "
              f"풀려면 python3 ~/home_position.py --release")
        port.closePort()


if __name__ == "__main__":
    main()
