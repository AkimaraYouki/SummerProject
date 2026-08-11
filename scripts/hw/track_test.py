#!/usr/bin/env python3
"""모터가 명령을 얼마나 따라가는지 **주파수별로** 잰다 (Jetson).

    ssh -t parksuho@192.168.137.7 'python3 ~/track_test.py left_knee,right_knee'
    ssh -t parksuho@192.168.137.7 'python3 ~/track_test.py left_knee --amp 15 --freqs 0.5,1,2,3,4'

────────────────────────────────────────────────────────────────────────
⚠️  로봇을 **매달아 놓고** 돌릴 것. 지정한 축만 토크가 켜진다.
────────────────────────────────────────────────────────────────────────

## 왜 계단 응답만으로는 부족한가

`joint_step_test.py` 는 한 번의 과도 응답을 본다 — "토크가 걸리나 / 기계적으로
막혔나" 를 가르는 데는 그게 맞다. 하지만 **"명령을 잘 따라가나"** 라는 질문의
답은 주파수 응답이다. 보행은 정지 상태에서 한 번 움직이는 게 아니라 1.85 Hz
로 계속 왕복하는 동작이고, 모터가 **어느 주파수부터 못 따라가기 시작하는지**
가 곧 그 보행이 실기에서 성립하는지를 결정한다.

2026-08-12 실기 전진 로그가 그 예다. 정책이 무릎에 지령한 속도 p95 가
5.90 rad/s 인데 실측은 1.25 (21 %) 였다. 발이 15 초 동안 한 번도 안 떨어졌고
몸통이 +16° 앞으로 기울었다. 이게 "모터가 고장" 인지 "원래 이 주파수에서는
이만큼밖에 못 따라감" 인지는 이 시험으로만 갈린다.

## 무엇을 재는가

각 주파수에서 사인파를 넣고

    goal(t) = center + amp * sin(2*pi*f*t)

세 가지를 뽑는다.

  * **이득(gain)** = 실제 진폭 / 지령 진폭.  1.0 이면 완벽 추종.
    0.71(-3 dB) 이 되는 주파수가 그 관절의 **대역폭**이다.
  * **위상 지연** = 실제가 지령보다 얼마나 늦는가 (도). 보행에서는 이게
    접지 타이밍을 밀어 넘어짐으로 이어진다.
  * **최대 속도 / 전류** = 그 주파수에서 실제로 낸 값. 지령 속도와 비교하면
    속도 포화인지 힘 부족인지 갈린다.

이득과 위상은 **상관 적분**으로 뽑는다 (단순 최대-최소보다 잡음에 강하다):

    A_sin = 2/T * ∫ x(t) sin(wt) dt      A_cos = 2/T * ∫ x(t) cos(wt) dt
    진폭 = hypot(A_sin, A_cos)           위상 = atan2(-A_cos, A_sin)

## 읽는 법

    이득 ~1.0, 위상 ~0°      -> 그 주파수까지는 잘 따라간다
    이득 뚝 떨어짐            -> 대역폭 한계. 속도/토크가 모자란다
    좌우 관절 이득이 다름     -> 한쪽 하드웨어 이상 (같은 모터·같은 게인인데 다르면)
    이득 낮은데 전류도 낮음   -> 토크가 안 걸린다 (접촉 불량·구동 이상)
    이득 낮은데 전류 천장     -> 힘이 모자란다 (부하·마찰)
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

LOG_PATH = os.path.expanduser("~/track_test_log.csv")
DT = 0.02          # 50 Hz — rl_walk 와 같은 제어 주기


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("joint", help="관절 이름. 쉼표로 여러 개 (예: left_knee,right_knee)")
    ap.add_argument("--amp", type=float, default=12.0,
                    help="사인파 진폭 (도, 기본 12). 보행 중 무릎 스윙이 편도 약 17도다")
    ap.add_argument("--freqs", default="0.5,1.0,1.5,2.0,3.0,4.0,5.0",
                    help="시험 주파수 (Hz), 쉼표 구분. 보행이 1.85 Hz 다")
    ap.add_argument("--cycles", type=float, default=4.0, help="주파수마다 몇 주기 돌릴지")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="사인 중심을 현재 위치에서 이만큼 옮긴다 (도)")
    ap.add_argument("--current", type=int, default=700, help="전류 상한 (틱)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    args = ap.parse_args()

    names = [n.strip() for n in args.joint.split(",")]
    for n in names:
        if n not in BY_NAME:
            sys.exit(f"모르는 관절: {n}  (가능: {', '.join(sorted(BY_NAME))})")
    ids = [BY_NAME[n][1] for n in names]
    freqs = [float(x) for x in args.freqs.split(",")]

    if not sys.stdin.isatty():
        sys.exit("TTY 가 아니다. ssh -t 로 붙어서 돌릴 것.")

    print(f"대상: {', '.join(f'{n}(ID{i})' for n, i in zip(names, ids))}")
    print(f"사인 ±{args.amp}° · 주파수 {freqs} Hz · 주파수당 {args.cycles} 주기")
    print(f"전류 상한 {args.current} 틱 ({args.current * CURRENT_UNIT_MA / 1000:.2f} A)")
    # 지령 최대 속도를 미리 알려준다 — 무부하 한계 4.82 rad/s 를 넘기면 애초에
    # 못 따라가는 게 정상이므로, 그걸 모르고 "고장" 으로 읽지 않게.
    for f in freqs:
        w = 2 * math.pi * f * math.radians(args.amp)
        mark = "  <- 무부하 한계 4.82 초과" if w > 4.82 else ""
        print(f"    {f:4.1f} Hz -> 지령 최대속도 {w:5.2f} rad/s{mark}")
    print("\n⚠️  로봇이 **매달려 있는지** 확인할 것. 이 축만 토크가 켜진다.")
    if input("계속하려면 go 입력: ").strip() != "go":
        sys.exit("취소")

    hwi = HWI(port=args.port, current_limit=args.current)
    hwi.io.sync_write_torque_enable(ids, [0] * len(ids))
    hwi.io.sync_write_operating_mode(ids, [MODE_CURRENT_POSITION] * len(ids))
    hwi.io.sync_write_current_limit(ids, [args.current] * len(ids))

    center = [rad_of(n, hwi.io.sync_read_present_position([i])[0]) + math.radians(args.offset)
              for n, i in zip(names, ids)]
    print("중심 위치: " + ", ".join(f"{n} {math.degrees(c):+.1f}°" for n, c in zip(names, center)))

    # 사인의 일부가 URDF 한계에 잘리면 이득이 무의미해진다 — 잘린 만큼 진폭이
    # 줄어 "모터가 못 따라간다" 로 오독하게 된다. 2026-08-12 에 실제로 겪었다:
    # left_hip_pitch 를 한계(+70°) 근처에서 재서 이득 0.13 이 나왔는데, 모터는
    # 멀쩡했고 목표의 위쪽 절반이 통째로 잘린 것이었다.
    clipped = []
    for n, c in zip(names, center):
        for edge in (+1, -1):
            want = c + edge * math.radians(args.amp)
            sent = rad_of(n, tick_of(n, want))
            if abs(sent - want) > math.radians(0.5):
                clipped.append((n, math.degrees(want), math.degrees(sent)))
    if clipped:
        print("\n!! 사인이 URDF 한계에 잘린다 — 이 조건의 이득은 못 믿는다:")
        for n, want, sent in clipped:
            print(f"     {n:16} 목표 {want:+7.1f}° -> 실제 전송 {sent:+7.1f}°")
        print("   --offset 으로 중심을 한계에서 떨어뜨리거나 --amp 를 줄일 것.")
        if input("   그래도 계속하려면 yes 입력: ").strip() != "yes":
            hwi.io.sync_write_torque_enable(ids, [0] * len(ids))
            sys.exit("취소")

    hwi.io.sync_write_torque_enable(ids, [1] * len(ids))
    log_f = open(LOG_PATH, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["freq", "t"] + [f"{p}_{n}" for n in names
                                    for p in ("goal_deg", "pos_deg", "cur_A", "vel_rads", "tq")])
    results: dict[str, list] = {n: [] for n in names}
    bad_reads = 0

    try:
        for f in freqs:
            w = 2 * math.pi * f
            dur = args.cycles / f
            # 상관 적분 누적기 (주파수마다 초기화)
            acc = {n: {"s": 0.0, "c": 0.0, "n": 0, "vmax": 0.0, "cmax": 0.0,
                       "gs": 0.0, "gc": 0.0} for n in names}
            t0 = time.time()
            while True:
                t = time.time() - t0
                if t >= dur:
                    break
                goal = [c + math.radians(args.amp) * math.sin(w * t) for c in center]
                hwi.io.sync_write_goal_position(ids, [tick_of(n, g) for n, g in zip(names, goal)])
                # SyncRead 가 짧게 돌아오는 일이 있다 (버스 충돌·타이밍). 그때
                # struct.unpack 이 죽으면 시험 전체가 날아가므로, 한 번 다시 읽고
                # 그래도 안 되면 그 샘플만 버린다.
                raws = hwi.io.sync_read_raw_data(ids, 126, 10)
                if any(len(b) != 10 for b in raws):
                    bad_reads += 1
                    raws = hwi.io.sync_read_raw_data(ids, 126, 10)
                    if any(len(b) != 10 for b in raws):
                        time.sleep(max(0.0, DT - (time.time() - t0 - t)))
                        continue
                try:
                    tqs = [b[0] for b in hwi.io.sync_read_raw_data(ids, 64, 1)]
                except (struct.error, IndexError):
                    tqs = [1] * len(ids)
                row = [f"{f:.3f}", f"{t:.4f}"]
                for k, n in enumerate(names):
                    c_raw, v_raw, p_raw = struct.unpack("<hii", raws[k])
                    pos = rad_of(n, p_raw) - center[k]
                    cur = abs(c_raw * CURRENT_UNIT_MA / 1000.0)
                    vel = abs(BY_NAME[n][2] * v_raw * 0.229 * 2.0 * math.pi / 60.0)
                    a = acc[n]
                    # 첫 주기는 과도구간이라 버린다
                    if t > 1.0 / f:
                        a["s"] += pos * math.sin(w * t)
                        a["c"] += pos * math.cos(w * t)
                        a["gs"] += math.radians(args.amp) * math.sin(w * t) * math.sin(w * t)
                        a["gc"] += math.radians(args.amp) * math.sin(w * t) * math.cos(w * t)
                        a["n"] += 1
                        a["vmax"] = max(a["vmax"], vel)
                        a["cmax"] = max(a["cmax"], cur)
                    row += [f"{math.degrees(goal[k] - center[k]):.3f}",
                            f"{math.degrees(pos):.3f}", f"{cur:.4f}", f"{vel:.4f}", tqs[k]]
                log_w.writerow(row)
                time.sleep(max(0.0, DT - (time.time() - t0 - t)))
            log_f.flush()

            print(f"\n  === {f:.1f} Hz ===" + (f"   (읽기 재시도 {bad_reads})" if bad_reads else ""))
            for n in names:
                a = acc[n]
                if a["n"] == 0:
                    continue
                As, Ac = 2.0 * a["s"] / a["n"], 2.0 * a["c"] / a["n"]
                Gs, Gc = 2.0 * a["gs"] / a["n"], 2.0 * a["gc"] / a["n"]
                amp_act = math.hypot(As, Ac)
                amp_cmd = math.hypot(Gs, Gc)
                gain = amp_act / max(amp_cmd, 1e-9)
                phase = math.degrees(math.atan2(-Ac, As) - math.atan2(-Gc, Gs))
                while phase > 180:
                    phase -= 360
                while phase < -180:
                    phase += 360
                results[n].append((f, gain, phase, a["vmax"], a["cmax"]))
                print(f"    {n:16} 이득 {gain:5.2f}  위상 {phase:+7.1f}°  "
                      f"실측최대속도 {a['vmax']:5.2f} rad/s  전류 {a['cmax']:5.2f} A")
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        log_f.close()
        hwi.io.sync_write_torque_enable(ids, [0] * len(ids))
        print("\n토크 끔.")

    print("\n" + "=" * 78)
    print("요약 — 이득이 0.71 아래로 내려가는 곳이 대역폭 (-3 dB)")
    print(f"  {'Hz':>5}" + "".join(f"{n[:14]:>17}" for n in names))
    for i, f in enumerate(freqs):
        cells = ""
        for n in names:
            if i < len(results[n]):
                _, g, ph, _, _ = results[n][i]
                cells += f"{g:9.2f}/{ph:+6.0f}°"
            else:
                cells += f"{'-':>17}"
        print(f"  {f:5.1f}" + cells)
    print("  (이득 / 위상)")
    if len(names) == 2:
        print("\n좌우 비교: 같은 모터·같은 게인이면 이득 차이가 10 % 안이어야 한다.")
        for i, f in enumerate(freqs):
            if i < len(results[names[0]]) and i < len(results[names[1]]):
                g0 = results[names[0]][i][1]
                g1 = results[names[1]][i][1]
                diff = (g1 - g0) / max(g0, 1e-9) * 100
                flag = "  <<< 이상" if abs(diff) > 20 else ""
                print(f"    {f:4.1f} Hz  {g0:.2f} vs {g1:.2f}   차 {diff:+.0f} %{flag}")
    print(f"\n로그: {LOG_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
