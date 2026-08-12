#!/usr/bin/env python3
"""관절을 아주 천천히 전 범위 훑으며 **각도별 마찰**을 잰다 (Jetson).

    ssh -t parksuho@192.168.137.7 'python3 ~/friction_map.py left_knee,right_knee'
    ssh -t parksuho@192.168.137.7 'python3 ~/friction_map.py left_knee --lo -60 --hi -118'

────────────────────────────────────────────────────────────────────────
⚠️  로봇을 **매달아 놓고** 돌릴 것. 대상 축만 토크가 켜진다.
────────────────────────────────────────────────────────────────────────

## 왜

2026-08-12 에 실기 왼무릎이 보행 중 목표보다 항상 22~32도 더 접힌 쪽에
밀려 있었다. 그런데 두 시험은 통과했다:

  * 주파수 응답 (READY 중심 ±10도): 이득 0.99, 좌우차 1 %
  * 손으로 몸통 들고 보행: 정상 추종

즉 **특정 각도 범위에서만** 문제가 난다. 손으로 만져 보니 뻑뻑하다고 한다 —
헐거운 게 아니라 마찰이다. 그렇다면 그 마찰이 **어느 각도에서** 커지는지가
곧 답이다.

## 무엇을 하는가

목표각을 아주 느리게(기본 6 도/초) 왕복시킨다. 이 속도에서는 관성과 점성이
무시할 만하므로, **전류이 곧 그 각도에서 이겨야 하는 마찰**이다.

    올라갈 때 전류  =  마찰 + 중력
    내려갈 때 전류  =  마찰 - 중력
    -> (올라갈때 - 내려갈때)/2  =  중력    (각도별 부하)
    -> (올라갈때 + 내려갈때)/2  =  마찰    (각도별 뻑뻑함)

왕복해서 두 방향을 다 재므로 **중력과 마찰이 분리된다.** 좌우를 같이 주면
같은 각도에서 나란히 비교한다 (거울 부호는 자동 처리).

## 읽는 법

    마찰이 전 구간 평평   정상
    특정 각도에서 치솟음  거기서 기구가 걸린다 (기어·링크·간섭)
    좌우 마찰이 2배 이상  한쪽 하드웨어 이상
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import sys
import time

sys.path.insert(0, os.path.expanduser("~"))
from rustypot_hwi import (  # noqa: E402
    BY_NAME,
    CURRENT_UNIT_MA,
    HWI,
    MODE_CURRENT_POSITION,
    rad_of,
    tick_of,
)

LOG = os.path.expanduser("~/friction_map.csv")
DT = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("joint", help="관절 이름, 쉼표 구분 (예: left_knee,right_knee)")
    ap.add_argument("--lo", type=float, default=None,
                    help="훑을 하한 (도, 왼쪽 기준). 기본은 현재위치 -35도")
    ap.add_argument("--hi", type=float, default=None,
                    help="훑을 상한 (도, 왼쪽 기준). 기본은 현재위치 +10도")
    ap.add_argument("--speed", type=float, default=6.0, help="훑는 속도 (도/초)")
    ap.add_argument("--current", type=int, default=700)
    ap.add_argument("--bins", type=int, default=12, help="각도 구간 수")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    args = ap.parse_args()

    names = [n.strip() for n in args.joint.split(",")]
    for n in names:
        if n not in BY_NAME:
            sys.exit(f"모르는 관절: {n}")
    ids = [BY_NAME[n][1] for n in names]

    if not sys.stdin.isatty():
        sys.exit("TTY 가 아니다. ssh -t 로 붙어서 돌릴 것.")

    hwi = HWI(port=args.port, current_limit=args.current)
    hwi.io.sync_write_torque_enable(ids, [0] * len(ids))
    hwi.io.sync_write_operating_mode(ids, [MODE_CURRENT_POSITION] * len(ids))
    hwi.io.sync_write_current_limit(ids, [args.current] * len(ids))

    start = [rad_of(n, hwi.io.sync_read_present_position([i])[0]) for n, i in zip(names, ids)]
    # 좌우를 같은 '굽힘량' 으로 다루려고 각 축의 부호를 본다. 무릎은 좌 음수 /
    # 우 양수라 그대로 비교하면 방향이 반대가 된다.
    sign = [1.0 if s >= 0 else -1.0 for s in start]
    lo = args.lo if args.lo is not None else -35.0
    hi = args.hi if args.hi is not None else +10.0

    print(f"대상: {', '.join(f'{n}(ID{i})' for n, i in zip(names, ids))}")
    print("시작 위치: " + ", ".join(f"{n} {math.degrees(s):+.1f}°" for n, s in zip(names, start)))
    print(f"훑기: 시작에서 {lo:+.0f}° ~ {hi:+.0f}° (굽힘 방향 기준) · {args.speed}°/s · 왕복 1회")
    span = hi - lo
    print(f"소요 약 {2 * span / args.speed:.0f} 초")
    print("\n⚠️  로봇이 **매달려 있는지** 확인할 것.")
    if input("계속하려면 go 입력: ").strip() != "go":
        sys.exit("취소")

    # 목표가 URDF 한계에 잘리면 그 구간 데이터가 무의미하다 — 미리 알린다.
    for n, s, sg in zip(names, start, sign):
        for edge in (lo, hi):
            want = s + sg * math.radians(edge)
            if abs(rad_of(n, tick_of(n, want)) - want) > math.radians(0.5):
                print(f"  !! {n}: {edge:+.0f}° 끝이 URDF 한계에 잘린다 — 그 구간은 못 잰다")

    hwi.io.sync_write_torque_enable(ids, [1] * len(ids))
    f = open(LOG, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["t", "dir"] + [f"{p}_{n}" for n in names for p in ("bend_deg", "cur_A")])

    data = {n: {"up": [], "dn": []} for n in names}
    t0 = time.time()
    try:
        for direction, (a, b) in (("up", (lo, hi)), ("dn", (hi, lo))):
            steps = int(abs(b - a) / args.speed / DT)
            for k in range(steps):
                u = k / max(steps - 1, 1)
                bend = a + (b - a) * u
                goals = [tick_of(n, s + sg * math.radians(bend))
                         for n, s, sg in zip(names, start, sign)]
                hwi.io.sync_write_goal_position(ids, goals)
                raw = hwi.io.sync_read_raw_data(ids, 126, 10)
                if any(len(x) != 10 for x in raw):
                    time.sleep(DT)
                    continue
                row = [f"{time.time()-t0:.3f}", direction]
                for j, (n, s, sg) in enumerate(zip(names, start, sign)):
                    c_raw, _v, p_raw = struct.unpack("<hii", raw[j])
                    pos_bend = sg * (rad_of(n, p_raw) - s)
                    cur = abs(c_raw * CURRENT_UNIT_MA / 1000.0)
                    data[n][direction].append((math.degrees(pos_bend), cur))
                    row += [f"{math.degrees(pos_bend):.2f}", f"{cur:.4f}"]
                wr.writerow(row)
                time.sleep(DT)
            print(f"  {direction} 완료 ({time.time()-t0:.0f}s)")
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        f.close()
        hwi.io.sync_write_torque_enable(ids, [0] * len(ids))
        print("토크 끔.")

    print("\n" + "=" * 78)
    print("각도별 마찰 = (올라갈때 + 내려갈때)/2 · 중력 = (올라갈때 - 내려갈때)/2  [A]")
    print(f"  {'굽힘각':>10}" + "".join(f"{n[:13]:>26}" for n in names))
    print(f"  {'':>10}" + "".join(f"{'마찰 / 중력':>26}" for _ in names))
    edges = [lo + (hi - lo) * i / args.bins for i in range(args.bins + 1)]
    prev = {n: None for n in names}
    for i in range(args.bins):
        a, b = edges[i], edges[i + 1]
        line = f"  {a:+5.0f}~{b:+4.0f}"
        for n in names:
            up = [c for ang, c in data[n]["up"] if a <= ang < b]
            dn = [c for ang, c in data[n]["dn"] if a <= ang < b]
            if not up or not dn:
                line += f"{'-':>26}"
                continue
            mu, md = sum(up) / len(up), sum(dn) / len(dn)
            fr, gr = (mu + md) / 2, (mu - md) / 2
            jump = ""
            if prev[n] is not None and fr > prev[n] * 1.8 and fr > 0.15:
                jump = " <<"
            prev[n] = fr
            line += f"{fr:12.3f} /{gr:+8.3f}{jump:>5}"
        print(line)
    print("\n  '<<' = 직전 구간 대비 마찰이 1.8배 이상 뛴 곳 (기구가 걸리는 지점)")
    if len(names) == 2:
        A, B = names
        fa = [c for _a, c in data[A]["up"]] + [c for _a, c in data[A]["dn"]]
        fb = [c for _a, c in data[B]["up"]] + [c for _a, c in data[B]["dn"]]
        if fa and fb:
            ma, mb = sum(fa) / len(fa), sum(fb) / len(fb)
            print(f"\n  전체 평균 전류  {A} {ma:.3f}A   {B} {mb:.3f}A   비 {mb/max(ma,1e-9):.2f}x")
            print("  (2배 넘게 차이나면 한쪽 하드웨어 이상)")
    print(f"\n로그: {LOG}")
    print("=" * 78)


if __name__ == "__main__":
    main()
