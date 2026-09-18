#!/usr/bin/env python3
"""보행 중 몸통 앞뒤 기울기가 **시간이 갈수록 앞으로 쏠리는지** 본다. 2026-09-17.

교수님 지적: "보행 초반 -> 후반으로 가면서 무게중심이 앞으로 쏠리는 것 같다.
자이로 적분과 관련 있나?"

우리 코드의 자이로 적분(`rl_walk --path-imu`)은 **요(좌우 회전)** 오차만 만든다.
앞뒤 기울기 관측(projected_gravity)은 BNO055 의 NDOF 융합값(GRV_DATA)에서 온다 —
우리가 적분하지 않는다. 그러나 **BNO055 내부 융합도 자이로를 적분**하고 가속도로
보정하므로, 그쪽이 흐르면 같은 증상이 난다. 그래서 두 가지를 갈라 본다.

    융합 기울기   = projected_gravity (로그 proj_grav_*) 에서 계산 — 정책이 본 값
    가속도 기울기 = 생 가속도(accel_*)를 구간 평균해서 계산 — 보행이 정상 속도면
                    구간 평균 가속도는 중력 방향에 가깝다

둘의 **차이가 시간에 따라 커지면 융합이 흐른 것**이다 (정책은 실제보다 뒤로
기운 줄 알고 앞으로 숙인다). 둘이 **같이 커지면 로봇이 실제로 앞으로 숙인 것**이다
(전압 강하·모터 발열·무게중심 자체 문제).

    python3 scripts/diag/pitch_drift.py ~/rl_walk_log.csv [구간초=2.0]
"""
from __future__ import annotations

import csv
import math
import statistics
import sys


def pitch_from(gx: float, gy: float, gz: float) -> float:
    """중력 방향 벡터에서 앞뒤 기울기(도). + 가 앞으로 숙임."""
    return math.degrees(math.atan2(gx, -gz))


def main(path: str, win_s: float = 2.0) -> None:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit("빈 로그")
    f = lambda r, k: float(r.get(k) or 0.0)                          # noqa: E731
    walk = [r for r in rows if abs(f(r, "cmd_vx")) > 0.05]
    if len(walk) < 50:
        raise SystemExit("보행 구간(|cmd_vx| > 0.05)이 너무 짧다")
    volts = [k for k in rows[0] if k.startswith("volt_")]
    temps = [k for k in rows[0] if k.startswith("temp_")]

    print(f"{path}: 전체 {len(rows)} 스텝 / 보행 {len(walk)} 스텝 "
          f"({f(walk[-1], 't') - f(walk[0], 't'):.0f} 초)")
    print(f"{'t(s)':>7} {'vx':>6} {'융합°':>7} {'가속도°':>8} {'차이°':>7} "
          f"{'자이로적분°':>11} {'전압V':>6} {'온도°C':>6}")

    out, t0 = [], f(walk[0], "t")
    i = 0
    gyro_int = 0.0
    while i < len(walk):
        j = i
        while j < len(walk) and f(walk[j], "t") - f(walk[i], "t") < win_s:
            j += 1
        w = walk[i:j]
        if len(w) < 10:
            break
        # 융합: 정책이 실제로 본 값
        fu = statistics.fmean(pitch_from(f(r, "proj_grav_x"), f(r, "proj_grav_y"),
                                         f(r, "proj_grav_z")) for r in w)
        # 가속도: 구간 평균 벡터의 방향. projected_gravity 와 부호를 맞춘다(-a/|a|).
        ax = statistics.fmean(f(r, "accel_x") for r in w)
        ay = statistics.fmean(f(r, "accel_y") for r in w)
        az = statistics.fmean(f(r, "accel_z") for r in w)
        n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ac = pitch_from(-ax / n, -ay / n, -az / n)
        # 참고: 자이로 y 를 그냥 적분하면 얼마나 흐르는지 (보정 없는 경우)
        for k in range(1, len(w)):
            gyro_int += f(w[k], "gyro_y") * max(0.0, f(w[k], "t") - f(w[k - 1], "t"))
        vv = statistics.fmean(f(r, k) for r in w for k in volts) if volts else 0.0
        tt = statistics.fmean(f(r, k) for r in w for k in temps) if temps else 0.0
        vx = statistics.fmean(f(r, "cmd_vx") for r in w)
        print(f"{f(w[0], 't') - t0:7.1f} {vx:+6.3f} {fu:+7.2f} {ac:+8.2f} {fu - ac:+7.2f} "
              f"{math.degrees(gyro_int):+11.1f} {vv:6.2f} {tt:6.1f}")
        out.append((fu, ac, fu - ac, vv, tt, (f(w[0], "t") - t0) / 60.0, vx))
        i = j

    if len(out) < 4:
        print("\n구간이 너무 적어 추세를 못 낸다.")
        return

    # 시간 추세와 속도 효과를 **함께** 푼다. 기울기는 걷는 속도에 따라 달라지므로
    # (빠를수록 더 숙인다) 초반·후반의 속도가 다르면 속도 차이가 시간 추세로
    # 둔갑한다. pitch = a + b*t + c*vx 로 놓고 b 만 본다.
    import numpy as np

    A = np.array([[1.0, r[5], r[6]] for r in out])       # 1, t(분), vx
    names = ("융합 기울기", "가속도 기울기", "둘의 차이(융합-가속도)")
    slopes = []
    for k, nm in enumerate(names):
        y = np.array([r[k] for r in out])
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ b
        slopes.append(b[1])
        print(f"  {nm:22s} {b[1]:+6.2f} 도/분   (속도 1 m/s 당 {b[2]:+.1f}도, 잔차 {resid.std():.2f}도)")
    vv = np.array([r[3] for r in out]); tt = np.array([r[4] for r in out])
    t_min = np.array([r[5] for r in out])
    print(f"  {'전압':22s} {np.linalg.lstsq(A[:, :2], vv, rcond=None)[0][1]:+6.2f} V/분"
          f"      {'온도':4s} {np.linalg.lstsq(A[:, :2], tt, rcond=None)[0][1]:+.2f} °C/분"
          f"   (관측 {t_min[-1] - t_min[0]:.1f}분)")

    print("\n판정:")
    fu, ac, df = slopes
    if abs(df) > 0.5 and abs(df) > abs(ac):
        print(f"  IMU 융합이 시간당 {df:+.2f}도/분 흘렀다 — BNO055 내부 자이로 적분 쪽이다.")
        print("  정책은 실제와 다른 기울기를 보고 그만큼 반대로 버틴다.")
    elif abs(ac) > 0.5:
        print(f"  실제 몸통이 {ac:+.2f}도/분 기울었고 융합은 그것을 따라갔다 —")
        print("  IMU 가 아니라 전압 강하·모터 발열·무게중심 쪽을 본다.")
    else:
        print("  이 로그에서는 시간에 따른 쏠림이 뚜렷하지 않다 (둘 다 0.5도/분 미만).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rl_walk_log.csv",
         float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
