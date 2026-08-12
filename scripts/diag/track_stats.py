#!/usr/bin/env python3
"""관절추종을 **심과 실기에 같은 자로** 재고 나란히 놓는다.

    # 실기 로그 하나만
    python3 scripts/diag/track_stats.py ~/rl_walk_log.csv

    # 심 vs 실기
    python3 scripts/diag/track_stats.py sim.csv --vs real.csv

    # 특정 명령 구간만 (전진만 본다)
    python3 scripts/diag/track_stats.py sim.csv --vs real.csv --vx 0.15

## 무엇을 재는가

관절마다 **목표각(goal_\\*) 대비 실제각(pos_\\*)** 을 본다.

    이득    실제진폭 / 목표진폭.  1.0 이면 명령한 만큼 움직인다.
            낮으면 못 따라간다 — 강성(P)이 모자라거나 토크가 잘린 것이다.
    RMS오차 목표와 실제의 차이. 이득이 1 이어도 위상이 밀리면 커진다.
    지연    상관이 최대가 되는 시간차. 버스·제어주기·필터가 만드는 순수 지연.
    오프셋  평균의 차이. 중력에 눌려 한쪽으로 처져 있으면 여기 나온다.

**이득과 지연을 나눠 보는 게 핵심이다.** 둘 다 RMS 를 키우지만 처방이 다르다
— 이득이 낮으면 P 를 올려야 하고, 지연은 P 로 못 고친다 (오히려 지연 + 높은
게인은 진동을 만든다).

## 왜 같은 자여야 하는가

심은 `play_fixed_cmd.py --track-csv`, 실기는 `rl_walk.py` 가 남기는
`~/rl_walk_log.csv` 인데 **열 이름과 정의를 일부러 똑같이 맞춰 놨다**
(goal_* = 실제로 관절에 나간 목표각 rad, pos_* = 실측 관절각 rad).
정의가 조금이라도 다르면 갭인지 측정 방법 차이인지 못 가린다 — 이 프로젝트
에서 이미 여러 번 그걸로 오진했다.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys

LEG = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
       "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
SHORT = {n: n.replace("left_", "L.").replace("right_", "R.") for n in LEG}


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"빈 파일: {path}")
    for need in ("t", "goal_left_knee", "pos_left_knee"):
        if need not in rows[0]:
            sys.exit(f"{path} 에 '{need}' 열이 없다 — 관절추종 로그가 아니다.\n"
                     f"  심:   play_fixed_cmd.py --track-csv <경로>\n"
                     f"  실기: rl_walk.py 가 ~/rl_walk_log.csv 로 남긴다")
    t = [_f(r, "t") for r in rows]
    span = t[-1] - t[0]
    hz = (len(rows) - 1) / span if span > 0 else 50.0
    return rows, hz


def select(rows, vx=None, wz=None, moving=False, tol=0.02):
    """명령으로 구간을 고른다. 둘 다 None 이고 moving 이면 '명령이 있는 구간'."""
    def ok(r):
        if vx is not None and abs(_f(r, "cmd_vx") - vx) > tol:
            return False
        if wz is not None and abs(_f(r, "cmd_wz") - wz) > tol:
            return False
        if moving and abs(_f(r, "cmd_vx")) < 0.05 and abs(_f(r, "cmd_wz")) < 0.1:
            return False
        return True
    return [r for r in rows if ok(r)]


def stats(rows, hz, max_lag_s=0.20):
    """관절별 (이득, RMS오차, 지연ms, 오프셋, 목표진폭). 각도 단위는 도."""
    d = math.degrees
    out = {}
    max_lag = max(1, int(max_lag_s * hz))
    for n in LEG:
        tg = [d(_f(r, "goal_" + n)) for r in rows]
        ps = [d(_f(r, "pos_" + n)) for r in rows]
        if len(tg) < 40 or any(x != x for x in tg[:5]):
            out[n] = None
            continue
        mt, mp = sum(tg) / len(tg), sum(ps) / len(ps)
        tc = [x - mt for x in tg]
        pc = [x - mp for x in ps]
        st, sp = statistics.pstdev(tc), statistics.pstdev(pc)
        err = [a - b for a, b in zip(tg, ps)]
        rms = math.sqrt(sum(e * e for e in err) / len(err))
        # 지연: 실제를 앞으로 당겨 목표와 가장 잘 겹치는 lag.
        bl, bc = 0, -9.0
        for lag in range(0, max_lag + 1):
            a = tc[:len(tc) - lag] if lag else tc
            b = pc[lag:] if lag else pc
            if len(a) < 30:
                continue
            try:
                c = statistics.correlation(a, b)
            except statistics.StatisticsError:
                continue
            if c > bc:
                bc, bl = c, lag
        out[n] = {"gain": (sp / st) if st > 1e-6 else float("nan"),
                  "rms": rms, "lag_ms": bl * 1000.0 / hz,
                  "offset": mp - mt, "amp": st, "corr": bc}
    return out


def print_one(title, s, n_rows, hz):
    print(f"\n{title}   ({n_rows} 샘플 · {n_rows/hz:.1f}s · {hz:.1f} Hz)")
    print(f"  {'관절':14}{'목표진폭':>9}{'이득':>7}{'RMS':>8}{'지연':>8}{'오프셋':>9}")
    for n in LEG:
        v = s.get(n)
        if not v:
            print(f"  {SHORT[n]:14}{'—':>9}")
            continue
        print(f"  {SHORT[n]:14}{v['amp']:9.2f}{v['gain']:7.2f}{v['rms']:8.2f}"
              f"{v['lag_ms']:6.0f}ms{v['offset']:+9.2f}")


def print_compare(a, b, la, lb):
    print(f"\n{'관절':14}"
          f"{'이득':>17}{'RMS오차':>17}{'지연(ms)':>17}{'오프셋':>17}")
    print(f"{'':14}" + "".join(f"{la[:6]:>8}{lb[:6]:>8}{'차':>1}" for _ in range(4)))
    print("-" * 82)
    worst = []
    for n in LEG:
        va, vb = a.get(n), b.get(n)
        if not va or not vb:
            continue
        cells = ""
        for k in ("gain", "rms", "lag_ms", "offset"):
            fa, fb = va[k], vb[k]
            cells += f"{fa:8.2f}{fb:8.2f} "
        print(f"{SHORT[n]:14}{cells}")
        worst.append((abs(vb["gain"] - va["gain"]), n, va["gain"], vb["gain"]))
    print("-" * 82)
    worst.sort(reverse=True)
    print(f"\n이득 차이가 큰 순 ({la} -> {lb})")
    for dg, n, ga, gb in worst[:5]:
        arrow = "낮다" if gb < ga else "높다"
        print(f"  {SHORT[n]:14} {ga:.2f} -> {gb:.2f}   ({dg:+.2f}, {lb} 가 {arrow})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="관절추종 CSV (심 또는 실기)")
    ap.add_argument("--vs", default=None, help="비교 대상 CSV")
    ap.add_argument("--label", default=None, help="첫 CSV 이름표")
    ap.add_argument("--vs-label", default=None, help="비교 CSV 이름표")
    ap.add_argument("--vx", type=float, default=None, help="이 vx 명령 구간만")
    ap.add_argument("--wz", type=float, default=None, help="이 wz 명령 구간만")
    ap.add_argument("--skip", type=float, default=0.0, metavar="SEC",
                    help="앞 몇 초를 버릴지. 심은 리셋 직후 과도구간이 있다")
    ap.add_argument("--all", action="store_true",
                    help="명령이 없는 구간도 포함 (기본은 움직이는 구간만)")
    args = ap.parse_args()

    def prep(path, label):
        rows, hz = load(path)
        if args.skip > 0:
            t0 = _f(rows[0], "t")
            rows = [r for r in rows if _f(r, "t") - t0 >= args.skip]
        sel = select(rows, args.vx, args.wz, moving=not args.all)
        if len(sel) < 40:
            sys.exit(f"{path}: 조건에 맞는 샘플이 {len(sel)}개뿐이다 — "
                     f"--vx/--wz 를 빼거나 --all 을 쓸 것")
        return sel, hz, (label or path.split("/")[-1])

    sa, hza, la = prep(args.csv, args.label)
    A = stats(sa, hza)
    print_one(f"[{la}]", A, len(sa), hza)
    if not args.vs:
        print("\n  이득 1.0 = 명령한 만큼 움직인다. 낮으면 P 를 올린다.")
        print("  지연은 P 로 못 고친다 — 지연 + 높은 게인은 진동을 만든다.")
        return
    sb, hzb, lb = prep(args.vs, args.vs_label)
    B = stats(sb, hzb)
    print_one(f"[{lb}]", B, len(sb), hzb)
    print("\n" + "=" * 82)
    print(f"심-실기 갭   ({la} vs {lb})")
    print("=" * 82)
    print_compare(A, B, la, lb)


if __name__ == "__main__":
    main()
