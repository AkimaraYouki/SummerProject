#!/usr/bin/env python3
"""P 게인을 바꿔 가며 같은 명령으로 걷게 하고 **착지 충격과 추종**을 잰다 (Jetson).

    ssh -t parksuho@192.168.137.7 'python3 ~/pgain_sweep.py'
    ssh -t parksuho@192.168.137.7 'python3 ~/pgain_sweep.py --gains 1402,1150,900 --secs 20'

────────────────────────────────────────────────────────────────────────
⚠️  로봇이 **바닥에서 걷는다.** 옆에서 받칠 준비를 하고, 주변을 치울 것.
    각 구간 사이에 멈추므로 그때 자세를 바로잡을 수 있다.
────────────────────────────────────────────────────────────────────────

## 왜

2026-08-14, 사용자가 "걸음이 퉁퉁거리지 않고 부드러웠으면 좋겠다" 고 했다.
발에 3 mm 고무패드를 붙여 수직 가속도 RMS 가 6.46 -> 3.92 로 이미 39 % 줄었고,
다음 손잡이가 P 게인이다.

P 가 낮으면 다리가 물러서 착지 충격을 흡수한다. 대신 추종이 나빠진다. 지금은
**낮출 여지가 있다** — 실기 무릎 추종 이득이 0.76~0.83 인데 심은 0.69~0.71 로,
실기가 오히려 뻣뻣하다. 심 수준까지 깎으면 충격은 줄고 sim2real 갭은 좁아진다.

다만 그건 이론이고, 실제 최적점은 재야 안다. `rl_walk.py --pgain` 이 이미
있으므로 값만 바꿔 가며 같은 명령으로 돌리고 로그를 남긴다.

## 무엇을 재는가

    착지 피크 가속도   접지 0->1 순간의 |a| 최댓값. "팍 찍히는" 정도.
    수직 가속도 RMS    중력 방향 성분. "퉁퉁거림" 그 자체.
    관절 추종 이득     실제진폭/목표진폭. 1.0 이면 명령대로 움직인다.
                       너무 낮아지면 P 를 과하게 깎은 것이다.

## 안전

`rl_walk.py` 의 TTY + "go" 확인은 이 스크립트가 사람에게 한 번 받고 pty 로
전달한다 (`pad_ctl.py` 와 같은 방식). 각 구간이 끝나면 rl_walk 가 스스로
자세를 붙잡고 종료한다.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pty
import statistics
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
RL_WALK = os.path.join(HOME, "rl_walk.py")
LOG = os.path.join(HOME, "rl_walk_log.csv")
G = 9.81
LEG = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
       "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def run_one(onnx, pgain, secs, vx, extra):
    argv = [sys.executable, RL_WALK, "--onnx", onnx, "--pgain", str(pgain),
            "--vx", str(vx), "--seconds", str(secs)]
    if extra:
        argv += extra.split()
    master, slave = pty.openpty()
    p = subprocess.Popen(argv, stdin=slave, close_fds=True)
    os.close(slave)
    os.write(master, b"go\n")     # 사람 확인은 이 스크립트가 위에서 받았다
    p.wait()
    try:
        os.close(master)
    except OSError:
        pass
    return p.returncode


def analyse(path):
    rows = [r for r in csv.DictReader(open(path)) if abs(f(r, "cmd_vx")) > 0.02]
    if len(rows) < 60:
        return None
    o = {"n": len(rows)}
    mag, vert = [], []
    for r in rows:
        a = (f(r, "accel_x"), f(r, "accel_y"), f(r, "accel_z"))
        g = (f(r, "proj_grav_x"), f(r, "proj_grav_y"), f(r, "proj_grav_z"))
        n = math.sqrt(sum(x * x for x in g)) or 1.0
        u = [x / n for x in g]
        mag.append(math.sqrt(sum(x * x for x in a)))
        vert.append(sum(a[i] * u[i] for i in range(3)) + G)
    o["vert_rms"] = statistics.pstdev(vert)
    strikes = []
    for i in range(1, len(rows)):
        for s in ("l", "r"):
            if f(rows[i - 1], "contact_" + s) < 0.5 and f(rows[i], "contact_" + s) > 0.5:
                w = mag[i:i + 4]
                if w:
                    strikes.append(max(w))
    if strikes:
        strikes.sort()
        o["hit_med"] = statistics.median(strikes)
        o["hit_max"] = strikes[-1]
        o["hits"] = len(strikes)
    gains = []
    for nm in LEG:
        tg = [math.degrees(f(r, "goal_" + nm)) for r in rows]
        ps = [math.degrees(f(r, "pos_" + nm)) for r in rows]
        st, sp = statistics.pstdev(tg), statistics.pstdev(ps)
        if st > 1e-6:
            gains.append(sp / st)
    if gains:
        o["gain"] = statistics.median(gains)
    V = [min(f(r, "volt_" + n) for n in LEG) for r in rows]
    o["vmin"] = min(V)
    T = [f(r, "t") for r in rows]
    cl = [f(r, "contact_l") for r in rows]
    cr = [f(r, "contact_r") for r in rows]
    dur = T[-1] - T[0]
    o["contact_hz"] = (sum(1 for a, b in zip(cl, cl[1:]) if a != b)
                       + sum(1 for a, b in zip(cr, cr[1:]) if a != b)) / max(dur, 1e-3)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HOME, "policy_v52/policy.onnx"))
    ap.add_argument("--gains", default="1402,1150,900")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--rl-args", default="")
    ap.add_argument("--keep", default=os.path.join(HOME, "pgain_sweep"))
    args = ap.parse_args()

    if not sys.stdin.isatty():
        sys.exit("TTY 가 아니다. ssh -t 로 붙어서 돌릴 것.")
    if not os.path.exists(args.onnx):
        sys.exit(f"onnx 가 없다: {args.onnx}")
    gains = [int(x) for x in args.gains.split(",")]
    os.makedirs(args.keep, exist_ok=True)

    print("=" * 72)
    print(f"P 게인 스윕  {gains}   각 {args.secs:.0f}초 · vx {args.vx:+.2f}")
    print(f"정책: {args.onnx}")
    print("=" * 72)
    print("\n⚠️  로봇이 바닥에서 걷는다. 받칠 준비를 하고 주변을 치울 것.")
    print("   구간 사이에 멈추므로 그때 자세를 바로잡으면 된다.")
    if input("계속하려면 go 입력: ").strip().lower() != "go":
        return 1

    res = []
    for i, g in enumerate(gains, 1):
        print(f"\n[{i}/{len(gains)}] P={g} — {args.secs:.0f}초")
        rc = run_one(args.onnx, g, args.secs, args.vx, args.rl_args)
        dst = os.path.join(args.keep, f"p{g}.csv")
        if os.path.exists(LOG):
            os.replace(LOG, dst)
            a = analyse(dst)
            res.append((g, a))
            print(f"     -> {dst}  (rl_walk 종료 {rc})")
        else:
            print(f"     !! 로그가 없다 (종료 {rc})")
            res.append((g, None))
        if i < len(gains):
            input("   자세를 바로잡고 Enter: ")

    print("\n" + "=" * 72)
    print(f"  {'P':>6}{'착지피크':>10}{'최대':>8}{'수직RMS':>9}{'추종이득':>9}"
          f"{'접지/s':>8}{'최저V':>7}")
    print("  " + "-" * 62)
    for g, a in res:
        if not a:
            print(f"  {g:6d}   측정 실패")
            continue
        print(f"  {g:6d}{a.get('hit_med', float('nan')):9.2f}"
              f"{a.get('hit_max', float('nan')):8.2f}{a['vert_rms']:9.2f}"
              f"{a.get('gain', float('nan')):9.2f}{a['contact_hz']:8.2f}{a['vmin']:7.1f}")
    print("\n  착지피크·수직RMS 는 낮을수록 부드럽다 (m/s2, 중력 9.81).")
    print("  추종이득이 0.7 아래로 떨어지면 P 를 과하게 깎은 것이다 —")
    print("  심의 무릎 이득이 0.69~0.71 이므로 그보다 낮아지면 심보다 무르다.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
