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

## ⚠️ 이 점수가 실기 순위와 어긋난 적이 있다

2026-08-18, 사용자 확인: "지금 실기에서 잘 되는 건 v61 이다. 전 것들은 전부
잘 안 되거나 넘어진다." 그런데 이 표는 반대로 매겼다:

    v59  점수 0.0165 · roll 6.66 · 옆걸음 0.0263   <- 여기서 1 등
    v61  점수 0.0221 · roll 4.97 · 옆걸음 0.0534   <- 실기에서 1 등

둘은 앞뒤·회전이 사실상 같고 옆걸음만 갈리는데 v61 은 흔들림이 25 % 적다.
**실기에서는 넘어지면 그 주행이 끝난다** — 추종 0.003 차이보다 흔들림
25 % 가 성패를 가른다. 이 점수는 낙상 위험을 거의 반영하지 못한다.

그러니 **점수로 후보를 좁히되, 순위는 실기로 정한다.** roll RMS 와 낙상률
열을 점수만큼 비중 있게 볼 것.

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
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "source"))
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX   # noqa: E402

#: 6 방향 가중치. 사용자 순위: 앞뒤 > 회전 > 완전 옆걸음.
PRIO_W = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}
#: 실사용 속도(사용자가 실제로 쓰는 명령 크기). 2026-08-18 추가.
#: 옛 npz 에는 없으므로 있을 때만 별도 열로 보여 준다.
HALF = ("fwd_half", "turn_half")
#: 합격선. 2026-08-24 에 **측정 잡음을 실측하고 다시 세웠다.** 같은 체크포인트를
#: 4 회 재니 지표마다 흩어짐이 크게 달랐다 (폭/평균):
#:
#:     일률W    3 %     <- 유일하게 믿을 만하다
#:     rollRMS 20 %
#:     포화%   22 %
#:     roll속  34 %
#:     옆      41 %
#:     점수    89 %
#:     낙상%  150 %     <- 같은 정책에서 0.00~0.56 이 나온다
#:     앞뒤   146 %
#:     회전   223 %     <- 0.0038~0.0414, 11 배
#:
#: 그래서 **낙상률과 추종은 문턱으로 쓰지 않는다.** 1 회 측정으로 판정하면
#: 동전던지기다. 예전 기준(낙상 <= 0.5 %)이 정확히 그랬다 — v75 를 0.00 으로
#: 합격시켰는데 4 회 평균은 0.375 였다.
#:
#: 남기는 문턱은 잡음 대비 여유가 있는 셋뿐이고, 값은 v75 4 회 평균 근처로 잡았다.
PASS_SAT_PCT = 0.25      # 포화율 [%]   v75 4회 0.179 (잡음 22 %)
PASS_WATT = 6.5          # 일률 [W]     v75 4회 6.23  (잡음 3 %)
PASS_ROLL_RMS = 2.90     # rollRMS [도]  v75 4회 2.35  (잡음 20 %)
#: 추종은 정확도가 아니라 **방향이 맞는지**만 본다 (사용자: 천천히라도 명령대로).
#: 이것만은 잡음에 강하다 — 부호와 크기 50 % 는 11 배 흔들려도 뒤집히지 않는다.
PASS_TRACK_FRAC = 0.5
#: 아래 둘은 **참고용으로만** 찍는다. 잡음이 커서 판정에 못 쓴다.
PASS_FALL_PCT = 0.5
#: 참고용으로 남기는 옛 추종 합격선. 판정에는 더 이상 쓰지 않는다.
PASS_SCORE = 0.0180
#: 토크 실효 한계 [N·m]. effort_limit 4.1 이 아니라 토크-속도 모델이 여기서 자른다.
TAU_SAT = 3.16
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
    half = [e[k] for k in HALF if k in e]
    o = {
        "score": sum(PRIO_W[k] * e[k] for k in PRIO_W) / sum(PRIO_W.values()),
        "fb": (e["forward"] + e["backward"]) / 2,
        "turn": e["turn"],
        "lr": (e["left"] + e["right"]) / 2,
        "stop": e.get("stop", float("nan")),
    }
    if half:
        o["half"] = float(np.mean(half))
    if roll_rms:
        o["roll"] = float(np.mean(roll_rms))
    if fall:
        o["fall"] = float(max(fall))
    # 2026-08-20, 목표가 바뀌었다: 피크 토크 최소화 · 보행 효율 최대화 ·
    # 토르소 좌우 흔들림 최소화. 재지 않으면 달성 여부를 알 수 없어 세 열을 더한다.
    dt = float(z["ctrl_dt"]) if "ctrl_dt" in z else 0.02
    peaks, sats, works, rrates = [], [], [], []
    for c in conds:
        if c == "stop":
            continue
        tk, dk = f"{c}__tau", f"{c}__dq"
        if tk in z:
            t = z[tk][SKIP:][..., ACT_LEG_JOINT_IDX]
            # 피크 |tau| 의 p99, 그리고 **포화율**. 실효 한계는 3.16 N·m 다
            # (effort_limit 4.1 이 아니라 토크-속도 모델이 먼저 자른다). 모든
            # 정책이 p99 에서 한계에 물리는 일이 흔해 값만으로는 안 갈린다 —
            # 얼마나 자주 한계에 붙어 있는지가 실제 차이다 (v68 0.35 % vs
            # v73 1.26 %).
            aa = np.abs(t)
            peaks.append(float(np.percentile(aa.max(axis=-1), 99)))
            sats.append(float(100.0 * (aa > TAU_SAT * 0.995).mean()))
            if dk in z:
                w = z[dk][SKIP:]
                n = min(t.shape[0], w.shape[0])
                # 기계적 일률 |tau . omega| [W]. 회생을 인정하지 않는 절대값 합 --
                # 서보는 역구동으로 에너지를 되돌려받지 못한다.
                works.append(float(np.abs(t[:n] * w[:n]).sum(-1).mean()))
        gk = f"{c}__grav"
        if gk in z:
            g = z[gk][SKIP:]
            rr = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
            rrates.append(float(np.diff(rr, axis=0).std() / dt))
    # 방향 준수: 6 방향 각각 부호가 맞고 크기가 명령의 PASS_TRACK_FRAC 이상인가.
    # "덜 움직이는 쪽" 으로 도망가는 정책을 잡는 유일한 문턱이다.
    obey = True
    for c in conds:
        if c == "stop":
            continue
        cmd = z[f"{c}__cmd"]
        v = z[f"{c}__v_base"][SKIP:].mean(axis=(0, 1))
        wz = float(z[f"{c}__w_base"][SKIP:].mean())
        for tgt, act in ((cmd[0], v[0]), (cmd[1], v[1]), (cmd[2], wz)):
            if abs(tgt) < 1e-6:
                continue
            if np.sign(tgt) != np.sign(act) or abs(act) < PASS_TRACK_FRAC * abs(tgt):
                obey = False
    o["obey"] = obey
    if peaks:
        o["peak"] = float(np.mean(peaks))
    if sats:
        o["sat"] = float(np.mean(sats))
    if works:
        o["watt"] = float(np.mean(works))
    if rrates:
        o["rrate"] = float(np.mean(rrates))
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
    print(f"  합격선   포화 <= {PASS_SAT_PCT:.2f}% · 일률 <= {PASS_WATT:.1f}W · "
          f"rollRMS <= {PASS_ROLL_RMS:.2f}도 · 6방향 부호 OK")
    print("  (낙상률·추종 정확도·roll속은 잡음이 커서 문턱으로 쓰지 않는다 — "
          "표에는 참고로 찍는다)")
    print("  " + "=" * 110)
    print(f"  {'버전':<10}{'iter':>6}{'점수':>9}{'앞뒤':>9}{'회전':>9}{'옆':>9}"
          f"{'rollRMS':>9}{'넘어짐':>8}{'p99t':>7}{'포화%':>7}{'일률W':>7}{'roll속':>8}   판정")
    print("  " + "-" * 110)
    winner = None
    for v, a in rows:
        checks = {
            "포화": a.get("sat", 99) <= PASS_SAT_PCT,
            "일률": a.get("watt", 99) <= PASS_WATT,
            "rollRMS": a.get("roll", 99) <= PASS_ROLL_RMS,
            "방향": a.get("obey", False),
        }
        if "sat" not in a or "roll" not in a:
            mark = "미측정"
        elif all(checks.values()):
            mark = "★ 합격"
            winner = winner or v
        elif not checks["방향"]:
            mark = "방향 실패"
        else:
            mark = "미달: " + ",".join(k for k, ok in checks.items() if not ok)
        r = f"{a['roll']:>9.2f}" if "roll" in a else f"{'—':>9}"
        f = f"{a['fall']:>8.1f}" if "fall" in a else f"{'—':>8}"
        pk = f"{a['peak']:>7.2f}" if "peak" in a else f"{'—':>7}"
        st = f"{a['sat']:>7.2f}" if "sat" in a else f"{'—':>7}"
        wt = f"{a['watt']:>7.1f}" if "watt" in a else f"{'—':>7}"
        rt = f"{a['rrate']:>8.1f}" if "rrate" in a else f"{'—':>8}"
        print(f"  {v:<10}{a['iter']:>6}{a['score']:>9.4f}{a['fb']:>9.4f}"
              f"{a['turn']:>9.4f}{a['lr']:>9.4f}{r}{f}{pk}{st}{wt}{rt}   {mark}")
    print("  " + "=" * 110)
    if winner:
        print(f"  → 합격: {winner}")
    else:
        print("  → 합격 없음. 항목별 최고:", end="")
        for key, lbl, fmt in (("sat", "포화", "{:.2f}%"), ("watt", "일률", "{:.1f}W"),
                              ("rrate", "roll속", "{:.1f}"), ("fall", "낙상", "{:.1f}%")):
            cand = [r for r in rows if key in r[1]]
            if cand:
                b = min(cand, key=lambda r: r[1][key])
                print(f"  {lbl} {b[0]} " + fmt.format(b[1][key]), end="")
        print()
    print(f"  p99t/포화% = 다리 |tau| 의 p99 [N·m] 와 한계 {TAU_SAT} 에 붙어 있는 비율")
    print("  일률W = |tau·omega| 합의 평균 [W] — 낮을수록 효율적")
    print("  roll속 = roll 각속도 RMS [deg/s] — 좌우 휘청거림")
    print("  판정은 목표(토크·효율·토르소·안정성) 기준이다. 점수 열은 참고용 —")
    print("  추종은 '천천히라도 명령대로' 이므로 방향 준수만 문턱으로 쓴다.")
    print("  ⚠ 1회 측정으로 추종을 비교하지 말 것 — 회전은 같은 정책에서 11배 흔들린다.")
    print("     `odm measure <ver> --repeat=4` 로 재고 noise_spread.py 로 볼 것.\n")


if __name__ == "__main__":
    main()
