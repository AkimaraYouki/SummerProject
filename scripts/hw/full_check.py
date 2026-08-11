#!/usr/bin/env python3
"""실기 14축 전체 점검 — 한 번에 (Jetson).

    ssh -t parksuho@192.168.137.7 'python3 ~/full_check.py --policy ~/policy_v42'

────────────────────────────────────────────────────────────────────────
⚠️  로봇을 **매달아 놓고** 돌릴 것.
    **14축 전부 토크를 걸어 READY 자세로 잡아 둔 채**, 한 축씩만 사인을 넣는다.
    (예전엔 대상 축만 토크를 켜서 나머지가 흐물거렸다 — 사람이 손으로 들고
     있어야 했고, 그러면 부하 조건이 매번 달라져 좌우 비교가 안 됐다.)
────────────────────────────────────────────────────────────────────────

## 왜

2026-08-12 에 관절을 하나씩 손으로 재다가 두 번 헛짚었다.
  * left_knee 가 보행에서 지령의 21% 만 내길래 고장으로 봤는데, 주파수 응답을
    재니 좌우 이득이 1% 차이였다 — 진범은 READY 10.3° 불일치였다.
  * left_hip_pitch 이득이 0.13 이 나와 또 고장인 줄 알았는데, 시작 위치가
    URDF 한계 근처라 사인의 절반이 잘린 것이었다.

두 번 다 **비교 대상과 조건**이 없어서 생긴 오독이다. 전 축을 같은 조건에서
한 번에 재고 좌우를 나란히 놓으면 이런 오독이 안 난다.

## 무엇을 하는가

축마다 (다리 10축을 순서대로. 머리 4축은 재지 않지만 READY 로 함께 잡는다):

  1. **한계 여유** — READY 자세에서 URDF 한계까지 남은 각도. 보행이 여기
     붙으면 클램프에 잘려 정책이 의도한 동작이 안 나온다.
  2. **주파수 응답** — 보행 주파수(1.85 Hz) 부근 세 점에서 이득·위상.
     사인이 한계에 잘리지 않도록 진폭을 자동으로 줄인다.
  3. **에러·온도** — 시험 전후.

마지막에 **좌우 쌍**을 비교해 이득 차이가 20 % 넘는 축만 표시한다.

## 읽는 법

    이득 ~1.0            그 축은 보행 주파수를 따라간다
    좌우 이득 차 >20 %   한쪽 하드웨어 이상
    한계 여유 <5°        보행 중 클램프에 걸릴 위험
    이득 낮은데 전류 낮음  토크가 안 걸린다
    이득 낮은데 전류 높음  힘이 모자란다
"""
from __future__ import annotations

import argparse
import json
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

#: 다리 10축만 본다. 머리는 lock_head_joints 라 보행에 안 쓰인다.
LEGS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
PAIRS = [(f"left_{b}", f"right_{b}")
         for b in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
DT = 0.02
#: 결과를 파일로도 남긴다. 화면에만 찍으면 원격(랩PC)에서 볼 수가 없다
#: (2026-08-12 에 실제로 한 번 날렸다).
OUT_TXT = os.path.expanduser("~/full_check_result.txt")
OUT_CSV = os.path.expanduser("~/full_check_raw.csv")
_LINES: list[str] = []


def w(msg: str = "") -> None:
    print(msg, flush=True)
    _LINES.append(msg)


def flush_out() -> None:
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(_LINES) + "\n")


def measure(hwi, name, idv, center, amp, freqs, cycles, hold_ticks, csv_w=None):
    """한 축의 주파수 응답. [(f, gain, phase, vmax, cmax), ...]"""
    out = []
    for f in freqs:
        w = 2 * math.pi * f
        dur = cycles / f
        s = c = gs = gc = 0.0
        cnt = 0
        vmax = cmax = 0.0
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= dur:
                break
            goal = center + math.radians(amp) * math.sin(w * t)
            # 대상 축은 사인, 나머지는 READY 로 계속 붙잡는다. 한 번의 SyncWrite 로
            # 같이 보내야 나머지 축이 그 사이 흘러내리지 않는다.
            goals = dict(hold_ticks)
            goals[idv] = tick_of(name, goal)
            hwi.io.sync_write_goal_position(list(goals), list(goals.values()))
            raw = hwi.io.sync_read_raw_data([idv], 126, 10)
            if not raw or len(raw[0]) != 10:
                time.sleep(max(0.0, DT - (time.time() - t0 - t)))
                continue
            c_raw, v_raw, p_raw = struct.unpack("<hii", raw[0])
            pos = rad_of(name, p_raw) - center
            if csv_w is not None:
                csv_w.writerow([name, f"{f:.3f}", f"{t:.4f}",
                                f"{math.degrees(goal - center):.3f}",
                                f"{math.degrees(pos):.3f}",
                                f"{c_raw * CURRENT_UNIT_MA / 1000:.4f}"])
            if t > 1.0 / f:                       # 첫 주기는 과도구간
                s += pos * math.sin(w * t)
                c += pos * math.cos(w * t)
                gs += math.radians(amp) * math.sin(w * t) ** 2
                gc += math.radians(amp) * math.sin(w * t) * math.cos(w * t)
                cnt += 1
                vmax = max(vmax, abs(BY_NAME[name][2] * v_raw * 0.229 * 2 * math.pi / 60))
                cmax = max(cmax, abs(c_raw * CURRENT_UNIT_MA / 1000))
            time.sleep(max(0.0, DT - (time.time() - t0 - t)))
        if cnt == 0:
            out.append((f, float("nan"), float("nan"), 0.0, 0.0))
            continue
        As, Ac = 2 * s / cnt, 2 * c / cnt
        Gs, Gc = 2 * gs / cnt, 2 * gc / cnt
        gain = math.hypot(As, Ac) / max(math.hypot(Gs, Gc), 1e-9)
        ph = math.degrees(math.atan2(-Ac, As) - math.atan2(-Gc, Gs))
        while ph > 180:
            ph -= 360
        while ph < -180:
            ph += 360
        out.append((f, gain, ph, vmax, cmax))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None,
                    help="정책 폴더. policy.meta.json 의 ready_joint_pos 를 중심으로 쓴다")
    ap.add_argument("--amp", type=float, default=10.0, help="사인 진폭 상한 (도)")
    ap.add_argument("--freqs", default="1.0,1.85,3.0", help="시험 주파수 (Hz)")
    ap.add_argument("--cycles", type=float, default=3.0)
    ap.add_argument("--current", type=int, default=700)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    args = ap.parse_args()
    freqs = [float(x) for x in args.freqs.split(",")]

    if not sys.stdin.isatty():
        sys.exit("TTY 가 아니다. ssh -t 로 붙어서 돌릴 것.")

    ready = None
    if args.policy:
        mp = os.path.join(os.path.expanduser(args.policy), "policy.meta.json")
        ready = (json.load(open(mp)).get("ready_joint_pos") or None) if os.path.exists(mp) else None
        w(f"READY 출처: {mp}" if ready else f"!! {mp} 에 ready_joint_pos 없음 — 현재 위치를 중심으로")

    w(f"\n다리 10축 · 주파수 {freqs} Hz · 진폭 최대 ±{args.amp}° · "
          f"전류 상한 {args.current * CURRENT_UNIT_MA / 1000:.2f} A")
    w("14축 전부 READY 로 잡아 둔 채 한 축씩 흔든다 — 손으로 들 필요 없다.")
    w(f"총 소요 약 {len(LEGS) * sum(args.cycles / f for f in freqs) / 60 + 0.2:.0f} 분")
    w("\n⚠️  로봇이 **매달려 있는지** 확인할 것.")
    if input("계속하려면 go 입력: ").strip() != "go":
        sys.exit("취소")

    hwi = HWI(port=args.port, current_limit=args.current)
    # **14축 전부** 잡는다. 대상 축만 켜면 나머지가 흐물거려 사람이 들고 있어야
    # 하고, 그러면 부하 조건이 매번 달라져 좌우 비교가 성립하지 않는다.
    all_names = list(BY_NAME)
    all_ids = [BY_NAME[n][1] for n in all_names]
    hwi.io.sync_write_torque_enable(all_ids, [0] * len(all_ids))
    hwi.io.sync_write_operating_mode(all_ids, [MODE_CURRENT_POSITION] * len(all_ids))
    hwi.io.sync_write_current_limit(all_ids, [args.current] * len(all_ids))

    # 각 축의 유지 목표 = READY (메타가 있으면 그 값, 없으면 현재 위치)
    hold_rad = {}
    for n in all_names:
        i = BY_NAME[n][1]
        hold_rad[i] = (ready[n] if ready and n in ready
                       else rad_of(n, hwi.io.sync_read_present_position([i])[0]))
    hold_ticks = {i: tick_of(n, hold_rad[i]) for n, i in zip(all_names, all_ids)}

    # 현재 자세에서 READY 로 천천히 (smoothstep 5초). 급출발 금지.
    start = {i: rad_of(n, hwi.io.sync_read_present_position([i])[0])
             for n, i in zip(all_names, all_ids)}
    hwi.io.sync_write_torque_enable(all_ids, [1] * len(all_ids))
    w("\n14축 토크 켬. READY 로 5초 램프인...")
    N_RAMP = 250
    for k in range(N_RAMP):
        u = (k + 1) / N_RAMP
        u = u * u * (3 - 2 * u)
        hwi.io.sync_write_goal_position(
            all_ids, [tick_of(n, start[i] + (hold_rad[i] - start[i]) * u)
                      for n, i in zip(all_names, all_ids)])
        time.sleep(0.02)
    time.sleep(0.5)
    w("READY 도달. 이제 한 축씩 잰다 (나머지는 계속 잡고 있다).\n")

    import csv as _csv
    csv_f = open(OUT_CSV, 'w', newline='')
    csv_w = _csv.writer(csv_f)
    csv_w.writerow(['joint', 'freq', 't', 'goal_deg', 'pos_deg', 'cur_A'])
    t_start = time.time()
    res: dict[str, dict] = {}
    for name in LEGS:
        idv = BY_NAME[name][1]
        center = hold_rad[idv]

        # 한계 여유: 중심에서 양쪽으로 얼마나 갈 수 있나 (tick_of 가 자르는 지점)
        margin = []
        for edge in (+1, -1):
            a = args.amp
            while a > 0.5:
                want = center + edge * math.radians(a)
                if abs(rad_of(name, tick_of(name, want)) - want) < math.radians(0.5):
                    break
                a -= 0.5
            margin.append(a)
        amp = min(args.amp, min(margin))
        temp0 = hwi.io.sync_read_raw_data([idv], 146, 1)[0][0]

        w(f"\n[{name}]  중심 {math.degrees(center):+7.2f}°  "
              f"한계여유 +{margin[0]:.1f}/-{margin[1]:.1f}°  진폭 ±{amp:.1f}°"
              + ("   <<< 한계에 붙어 진폭을 줄임" if amp < args.amp - 0.1 else ""))
        if amp < 2.0:
            w("   진폭이 너무 작다 — 건너뛴다 (중심이 한계에 붙어 있다)")
            res[name] = dict(skip=True, margin=margin, center=center)
            continue

        r = measure(hwi, name, idv, center, amp, freqs, args.cycles, hold_ticks, csv_w)
        # 대상 축을 READY 로 되돌린 뒤 다음 축으로 (토크는 계속 켜 둔다)
        hwi.io.sync_write_goal_position(list(hold_ticks), list(hold_ticks.values()))
        time.sleep(0.4)
        temp1 = hwi.io.sync_read_raw_data([idv], 146, 1)[0][0]
        err = hwi.io.sync_read_hardware_error_status([idv])[0]
        res[name] = dict(r=r, margin=margin, amp=amp, center=center,
                         temp=(temp0, temp1), err=err, skip=False)
        for f, g, ph, vm, cm in r:
            w(f"     {f:5.2f} Hz  이득 {g:5.2f}  위상 {ph:+6.1f}°  "
                  f"vmax {vm:5.2f} rad/s  전류 {cm:5.2f} A")
        w(f"     온도 {temp0}->{temp1}C  에러 {err}")

    w("\n" + "=" * 84)
    w(f"전체 점검 결과  ({(time.time() - t_start) / 60:.1f} 분)")
    w("=" * 84)
    w(f"  {'관절':16}{'한계여유':>12}" + "".join(f"{f'{f}Hz 이득':>12}" for f in freqs) + f"{'온도':>7}{'에러':>6}")
    for n in LEGS:
        d = res.get(n, {})
        if d.get("skip"):
            w(f"  {n:16}{'+%.0f/-%.0f' % tuple(d['margin']):>12}   (한계에 붙어 건너뜀)")
            continue
        gains = "".join(f"{g:12.2f}" for _, g, _, _, _ in d["r"])
        flag = "  <<<" if any(g < 0.8 for _, g, _, _, _ in d["r"]) else ""
        w(f"  {n:16}{'+%.0f/-%.0f' % tuple(d['margin']):>12}{gains}"
              f"{d['temp'][1]:6}C{d['err']:6}{flag}")

    w("\n  좌우 비교 (같은 모터·같은 게인이면 20 % 안)")
    for a, b in PAIRS:
        da, db = res.get(a, {}), res.get(b, {})
        if da.get("skip") or db.get("skip") or "r" not in da or "r" not in db:
            w(f"    {a[5:]:12} (비교 불가 — 한쪽을 못 쟀다)")
            continue
        line = f"    {a[5:]:12}"
        worst = 0.0
        for i, (f, ga, *_ ) in enumerate(da["r"]):
            gb = db["r"][i][1]
            diff = (gb - ga) / max(ga, 1e-9) * 100
            worst = max(worst, abs(diff))
            line += f"  {f}Hz {ga:.2f}/{gb:.2f} ({diff:+.0f}%)"
        w(line + ("   <<< 이상" if worst > 20 else ""))
    w("=" * 84)
    csv_f.close()
    flush_out()
    print(f"\n결과 저장: {OUT_TXT}  (원시 샘플 {OUT_CSV})")
    w("\n토크는 켜 둔 채 READY 를 유지한다. 풀려면:")
    w("  python3 ~/home_position.py --release")


if __name__ == "__main__":
    # 중간에 끊겨도(Ctrl+C·에러) 거기까지의 결과는 남긴다. 2026-08-12 에 한 번
    # 통째로 날렸다 — 원격에서는 화면에 남은 게 없으면 그걸로 끝이다.
    try:
        main()
    finally:
        if _LINES:
            flush_out()
            print(f"[full_check] 결과 저장: {OUT_TXT}")
