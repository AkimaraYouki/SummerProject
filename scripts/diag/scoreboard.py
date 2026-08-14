#!/usr/bin/env python3
"""모든 정책을 **합격선에 대고** 한 표로 채점한다. 자동 실험 루프의 판정기.

    python3 scripts/diag/scoreboard.py                 # 전부
    python3 scripts/diag/scoreboard.py v47 v59 v60     # 일부만
    python3 scripts/diag/scoreboard.py --top 8

`~/odm_out/gait_<ver>.npz` (= `odm measure` 산출물) 을 읽는다.

## 왜 별도 도구인가

2026-08-15, 사용자가 "1 순위 2 순위 둘 다 잡을 때까지 개입 없이 자동으로
돌려" 라고 했다. 사람이 매번 판단하지 않으려면 **합격 여부가 숫자로 결정**
돼야 한다. `verdict.py` 는 값을 보여 주지만 합격을 말하지 않는다.

## 합격선

    1 순위  우선순위 점수 <= 0.0180
            (앞뒤 3 : 회전 2 : 옆 1 가중. v59 가 0.0165 를 냈고, 측정 잡음이
             단일 방향 +-25 % 이므로 그 위로 약간의 여유를 뒀다.)
    2 순위  roll RMS 평균 <= 5.50 도      (v47 5.80 · v57 5.56 · v59 6.66)
            넘어짐 비율 최대 <= 0.5 %     (|roll| > 40 도 인 표본의 비율)

roll 진폭(p-p)은 합격선에 넣지 않는다. 드문 넘어짐 한 건이 값을 지배해서
잡음이 크다 — 대신 넘어짐 비율을 따로 본다.

## 측정 잡음

같은 체크포인트(v48)를 같은 날 두 번 재니 단일 방향이 +-25 % 흔들렸다
(전진 0.0231 vs 0.0287). 종합 점수는 그보다 안정적이지만, **0.002 이하
차이는 읽지 말 것.**
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

#: 6 방향 가중치. 사용자 순위: 앞뒤 > 회전 > 완전 옆걸음.
PRIO_W = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}
#: 합격선.
PASS_SCORE = 0.0180
PASS_ROLL_RMS = 5.50
PASS_FALL_PCT = 0.5
#: 넘어졌다고 볼 몸통 기울기 (도).
FALL_DEG = 40.0
SKIP = 100


def analyse(path):
    z = np.load(path, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    e, roll_rms, fall = {}, [], []
    for c in conds:
        v = z[f"{c}__v_base"][SKIP:]
        w = z[f"{c}__w_base"][SKIP:]
        cmd = z[f"{c}__cmd"]
        # 회전 명령의 추종은 요레이트로 본다.
        e[c] = (float(abs(w.mean() - cmd[2])) if c == "turn"
                else float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2])))
        key = f"{c}__grav"
        if key in z:
            g = z[key][SKIP:]
            r = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
            if c != "stop":
                roll_rms.append(float(r.std()))
            fall.append(float(100.0 * (np.abs(r) > FALL_DEG).mean()))
    if not all(k in e for k in PRIO_W):
        return None
    o = {
        "score": sum(PRIO_W[k] * e[k] for k in PRIO_W) / sum(PRIO_W.values()),
        "fb": (e["forward"] + e["backward"]) / 2,
        "turn": e["turn"],
        "lr": (e["left"] + e["right"]) / 2,
        "stop": e.get("stop", float("nan")),
    }
    if roll_rms:
        o["roll"] = float(np.mean(roll_rms))
    if fall:
        o["fall"] = float(max(fall))
    m = re.search(r"model_(\d+)\.pt", str(z["checkpoint"]))
    o["iter"] = int(m.group(1)) + 1 if m else 0
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vers", nargs="*", help="비우면 전부")
    ap.add_argument("--top", type=int, default=0, help="점수 상위 N 개만")
    args = ap.parse_args()

    paths = ([f"{os.path.expanduser('~')}/odm_out/gait_{v}.npz" for v in args.vers]
             if args.vers else sorted(glob.glob(os.path.expanduser("~/odm_out/gait_*.npz"))))
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  {os.path.basename(p)}: 없음")
            continue
        try:
            a = analyse(p)
        except Exception as ex:                                   # noqa: BLE001
            print(f"  {os.path.basename(p)}: 읽기 실패 ({ex})")
            continue
        if a:
            rows.append((os.path.basename(p)[5:-4], a))
    if not rows:
        return
    rows.sort(key=lambda r: r[1]["score"])
    if args.top:
        rows = rows[:args.top]

    print()
    print(f"  합격선   1순위 점수 <= {PASS_SCORE:.4f}   ·   "
          f"2순위 roll RMS <= {PASS_ROLL_RMS:.2f}도, 넘어짐 <= {PASS_FALL_PCT:.1f}%")
    print("  " + "=" * 92)
    print(f"  {'버전':<10}{'iter':>6}{'점수':>9}{'앞뒤':>9}{'회전':>9}{'옆':>9}"
          f"{'rollRMS':>9}{'넘어짐':>8}   판정")
    print("  " + "-" * 92)
    winner = None
    for v, a in rows:
        p1 = a["score"] <= PASS_SCORE
        p2 = (a.get("roll", 99) <= PASS_ROLL_RMS
              and a.get("fall", 99) <= PASS_FALL_PCT)
        if "roll" not in a:
            mark = "2순위 미측정"
        elif p1 and p2:
            mark = "★ 합격 (1·2순위)"
            winner = winner or v
        elif p1:
            mark = "1순위만"
        elif p2:
            mark = "2순위만"
        else:
            mark = "—"
        r = f"{a['roll']:>9.2f}" if "roll" in a else f"{'—':>9}"
        f = f"{a['fall']:>8.1f}" if "fall" in a else f"{'—':>8}"
        print(f"  {v:<10}{a['iter']:>6}{a['score']:>9.4f}{a['fb']:>9.4f}"
              f"{a['turn']:>9.4f}{a['lr']:>9.4f}{r}{f}   {mark}")
    print("  " + "=" * 92)
    if winner:
        print(f"  → 합격: {winner}")
    else:
        best1 = min(rows, key=lambda r: r[1]["score"])
        cand = [r for r in rows if "roll" in r[1]]
        best2 = min(cand, key=lambda r: r[1]["roll"]) if cand else None
        print(f"  → 아직 없음. 1순위 최고 {best1[0]} ({best1[1]['score']:.4f})", end="")
        if best2:
            print(f" · 2순위 최고 {best2[0]} ({best2[1]['roll']:.2f}도)")
        else:
            print()
    print("  0.002 이하 점수 차이는 측정 잡음이다 — 읽지 말 것.\n")


if __name__ == "__main__":
    main()
