#!/usr/bin/env python3
"""여러 정책의 **걸음 품질**을 한 표로 비교한다.

    python3 scripts/diag/gait_quality.py a.csv b.csv c.csv --labels v48 v49 v50

입력은 `play_fixed_cmd --track-csv` 또는 실기 `~/rl_walk_log.csv`.

## 왜 보상 대신 이걸 보나

환경이나 레퍼런스를 바꾼 런은 **보상을 비교할 수 없다.** 다른 시험을 친
것이기 때문이다 — 2026-08-13 에 v50(지면 마찰 0.5)이 351.0 으로 v48(318.8)을
크게 앞섰지만, 마찰을 바꾸면 물리가 달라지므로 그 숫자로 "더 잘 걷는다" 고
말할 수 없다.

그래서 환경과 무관하게 **걸음 자체의 성질**을 잰다. 각 항목은 실기에서
관측된 증상과 하나씩 대응한다:

    목표각 속도 p95   정책이 모터 속도 클램프를 무는가.
                      실기에서 클램프(4.82 rad/s)에 상시 붙어 있었고, 그
                      상태에서는 토크-속도 직선상 남는 토크가 없어 착지
                      직전에 감속하지 못한다 (착지 피크 2.4 g).
    roll 진폭         좌우 덜컹거림. 실기 20.4°, 심 21.0° 로 같았다 —
                      배포 문제가 아니라 학습된 걸음이라는 근거였다.
    발 들림           몸통 기준 발 하나의 z 범위. ⚠️ 몸통 출렁임이 섞이므로
                      **발 여유가 아니다** (2026-08-14 정정). 접지 발 기준으로
                      다시 재면 중앙값 5.2 mm(다리의 3.3 %)로 사람에 가깝다 —
                      발 높이는 애초에 문제가 아니었다. 좌우가 문제다.
    접지 전환         걸음 빈도. 너무 높으면 종종거리는 것이고 너무 낮으면
                      발을 끄는 것이다.
    명령 추종         명령 속도 대비 실제 몸통 속도. 2026-08-14 에 사용자가
                      우선순위를 **1 추종 · 2 부드러움 · 3 진동** 으로 못박았다.
                      그런데 이 표에는 추종이 없어서, "부드러워졌지만 안 간다"
                      를 걸러낼 수가 없었다. 부드러움 항을 걸 때마다 이 줄을
                      먼저 볼 것 — 여기가 무너지면 나머지는 의미가 없다.
                      실기 로그에는 오도메트리가 없어 빈다.
    관절 추종         목표각 대비 실제 관절각의 진폭비. 이쪽은 정책이 아니라
                      PD 게인과 부하의 문제다. 실기 0.76~0.83 / 심 0.69~0.71.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from leg_fk import foot_in_trunk  # noqa: E402

NAMES = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
LEG = [n for n in NAMES if not n.startswith(("neck", "head"))]
#: 다리 마디 길이 합 (URDF: hip_pitch->knee 78.65 mm, knee->ankle 78.65 mm).
LEG_LEN_MM = 157.3


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def analyse(path, skip=4.0):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    t0 = f(rows[0], "t")
    rows = [r for r in rows if f(r, "t") - t0 >= skip]
    # 실기 로그면 명령이 있는 구간만
    if "cmd_vx" in rows[0]:
        moving = [r for r in rows if abs(f(r, "cmd_vx")) > 0.05]
        if len(moving) > 60:
            rows = moving
    if len(rows) < 60:
        return None
    T = [f(r, "t") for r in rows]
    dur = T[-1] - T[0]
    hz = (len(rows) - 1) / dur if dur > 0 else 50.0
    out = {"n": len(rows), "hz": hz}

    # 목표각 변화 속도 (정책이 요구하는 관절 속도)
    p95 = []
    for n in LEG:
        v = [f(r, "goal_" + n) for r in rows]
        d = [abs(b - a) * hz for a, b in zip(v, v[1:])]
        d = sorted(x for x in d if x == x)
        if d:
            p95.append(d[int(0.95 * (len(d) - 1))])
    out["tgt_p95"] = max(p95) if p95 else float("nan")
    out["tgt_p95_med"] = statistics.median(p95) if p95 else float("nan")

    # 몸통 자세
    if "proj_grav_z" in rows[0]:
        rol = [math.degrees(math.atan2(-f(r, "proj_grav_y"), -f(r, "proj_grav_z"))) for r in rows]
        pit = [math.degrees(math.atan2(f(r, "proj_grav_x"), -f(r, "proj_grav_z"))) for r in rows]
        out["roll_pp"] = max(rol) - min(rol)
        out["roll_rms"] = statistics.pstdev(rol)
        out["pitch_mean"] = sum(pit) / len(pit)
    # 접지
    if "contact_l" in rows[0]:
        cl = [f(r, "contact_l") for r in rows]
        cr = [f(r, "contact_r") for r in rows]
        tr = (sum(1 for a, b in zip(cl, cl[1:]) if a != b)
              + sum(1 for a, b in zip(cr, cr[1:]) if a != b))
        out["contact_hz"] = tr / dur

    # 스윙(발이 떠 있는) 구간의 궤적 모양 — 수직으로 드는가.
    #
    # 2026-08-14, 사용자가 다른 빌더의 로봇 영상을 보고 "발을 거의 수직으로
    # 들고 높이도 사람처럼 매우 작은데 앞뒤양옆 다 잘 간다" 고 했다. 재 보니
    # 우리는 한 번 스윙에 위로 22 mm 뜨는 동안 **옆으로 33~39 mm** 움직인다.
    # 수직이 아니라 옆으로 호를 그리는 것이고, 그만큼 몸통이 좌우로 무게를
    # 옮겨야 해서 roll 이 흔들린다.
    if "contact_l" in rows[0]:
        for side, ck in (("left", "contact_l"), ("right", "contact_r")):
            segs, cur = [], []
            for r in rows:
                ja = {n: f(r, "pos_" + n) for n in NAMES}
                if any(v != v for v in ja.values()):
                    continue
                pt = foot_in_trunk(ja)[side]
                if f(r, ck) < 0.5:
                    cur.append(pt)
                elif cur:
                    if len(cur) >= 4:
                        segs.append(cur)
                    cur = []
            if len(segs) < 3:
                continue
            rise = statistics.median(max(p[2] for p in g) - min(p[2] for p in g) for g in segs) * 1000
            lat = statistics.median(max(p[1] for p in g) - min(p[1] for p in g) for g in segs) * 1000
            out.setdefault("rise", []).append(rise)
            out.setdefault("lat", []).append(lat)
        if "rise" in out:
            out["swing_rise"] = sum(out.pop("rise")) / 2
            out["swing_lat"] = sum(out.pop("lat")) / 2
            out["vertical"] = out["swing_rise"] / max(out["swing_lat"], 1e-6)

    # 명령 추종 — 우선순위 1 위. 심 로그에만 vel_* 가 있다.
    if "vel_x" in rows[0]:
        vx = [f(r, "vel_x") for r in rows]
        vy = [f(r, "vel_y") for r in rows]
        cx = [f(r, "cmd_vx") for r in rows]
        cy = [f(r, "cmd_vy") for r in rows]
        mcx = sum(cx) / len(cx)
        if abs(mcx) > 1e-3:
            out["track_x"] = (sum(vx) / len(vx)) / mcx
        out["track_err"] = math.sqrt(sum((a - b) ** 2 for a, b in zip(vx, cx)) / len(vx))
        # 옆으로 새는 양. 전진 명령에서는 0 이어야 한다.
        out["drift_y"] = abs(sum(vy) / len(vy) - sum(cy) / len(cy))
    if "gyro_z" in rows[0]:
        wz = [f(r, "gyro_z") for r in rows]
        cw = [f(r, "cmd_wz") for r in rows]
        out["yaw_err"] = math.sqrt(sum((a - b) ** 2 for a, b in zip(wz, cw)) / len(wz))

    # 관절 추종 — 목표각 진폭 대비 실제 진폭. 심·실기 모두 있다.
    gains = []
    for n in LEG:
        tg = [f(r, "goal_" + n) for r in rows]
        ps = [f(r, "pos_" + n) for r in rows]
        if any(v != v for v in tg) or any(v != v for v in ps):
            continue
        st = statistics.pstdev(tg)
        if st > 1e-6:
            gains.append(statistics.pstdev(ps) / st)
    if gains:
        out["joint_gain"] = statistics.median(gains)

    # 발 궤적 (FK)
    lifts, strides = [], []
    for side in ("left", "right"):
        z, x = [], []
        for r in rows:
            try:
                ja = {n: f(r, "pos_" + n) for n in NAMES}
            except Exception:
                continue
            if any(v != v for v in ja.values()):
                continue
            p = foot_in_trunk(ja)[side]
            z.append(p[2]); x.append(p[0])
        if len(z) > 40:
            z.sort(); x.sort()
            q = lambda a, p: a[int(p * (len(a) - 1))]
            lifts.append((q(z, .95) - q(z, .05)) * 1000)
            strides.append((q(x, .95) - q(x, .05)) * 1000)
    if lifts:
        out["lift"] = sum(lifts) / len(lifts)
        out["stride"] = sum(strides) / len(strides)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--skip", type=float, default=4.0)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(p).split(".")[0] for p in args.csv]

    res = []
    for p, lab in zip(args.csv, labels):
        a = analyse(p, args.skip)
        if a is None:
            print(f"  {lab}: 읽을 수 없거나 샘플 부족 ({p})")
            continue
        res.append((lab, a))
    if not res:
        return

    def row(name, key, fmt, note=""):
        cells = "".join(f"{fmt.format(a[key]):>12}" if key in a and a[key] == a[key]
                        else f"{'—':>12}" for _, a in res)
        print(f"  {name:22}{cells}   {note}")

    print("\n  " + " " * 22 + "".join(f"{lab:>12}" for lab, _ in res))
    print("  " + "-" * (22 + 12 * len(res) + 3))
    print("  [1] 추종 — 여기가 무너지면 나머지는 의미 없다")
    row("전진 추종 (실제/명령)", "track_x", "{:.2f}", "1.0 이 완벽")
    row("전진 추종 오차 RMS", "track_err", "{:.3f}", "m/s")
    row("좌우 드리프트", "drift_y", "{:.3f}", "m/s · 0 이어야")
    row("요 추종 오차 RMS", "yaw_err", "{:.3f}", "rad/s")
    row("관절 추종 이득", "joint_gain", "{:.2f}", "실제진폭/목표진폭")
    print("  [2] 부드러움 · [3] 진동")
    row("목표각 속도 p95(최대)", "tgt_p95", "{:.2f}", "rad/s · 클램프 4.82")
    row("  같은 값 (중앙값)", "tgt_p95_med", "{:.2f}", "rad/s")
    row("roll 진폭 (p-p)", "roll_pp", "{:.1f}", "도 · 낮을수록 안 덜컹")
    row("roll RMS", "roll_rms", "{:.2f}", "도")
    row("pitch 평균", "pitch_mean", "{:+.1f}", "도")
    row("발 들림", "lift", "{:.1f}", f"mm · 다리 {LEG_LEN_MM:.0f} mm")
    row("  다리 길이 대비", "lift", "{:.0f}", "")
    row("보폭", "stride", "{:.1f}", "mm")
    row("접지 전환", "contact_hz", "{:.1f}", "/s")
    row("스윙 상승", "swing_rise", "{:.1f}", "mm · 한 번 스윙에")
    row("스윙 좌우", "swing_lat", "{:.1f}", "mm · 작을수록 수직")
    row("수직성 (상승/좌우)", "vertical", "{:.2f}", "1.0 이상이면 위로 더 간다")
    print()
    for lab, a in res:
        if "lift" in a:
            print(f"  {lab}: 발 들림 {a['lift']:.1f} mm = 다리 길이의 "
                  f"{100*a['lift']/LEG_LEN_MM:.0f} %")


if __name__ == "__main__":
    main()
