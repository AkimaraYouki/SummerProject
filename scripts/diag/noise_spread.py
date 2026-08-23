#!/usr/bin/env python3
"""같은 체크포인트를 여러 번 잰 npz 들의 **흩어짐**을 본다. 잡음 문턱을 정한다.

    python3 scripts/diag/noise_spread.py ~/odm_out/noise/v75_*.npz

## 왜

2026-08-23 현재, 정책 비교가 소수점 둘째 자리에서 갈린다 (v75 낙상 0.00 % vs
v81 0.88 %). 그런데 **그 차이가 잡음인지 아닌지 모른다.** 추종 점수는 같은
체크포인트를 두 번 재니 단일 방향이 +-25 % 흔들린 전례가 있는데(v48, 0.0231 vs
0.0287), 새로 넣은 지표(포화율·일률·roll속·낙상률)는 아예 재 본 적이 없다.

같은 정책을 N 번 재서 표준편차와 최대-최소를 내면, **그보다 작은 차이는 읽지
않는다**는 규칙을 숫자로 세울 수 있다.

흔들리는 원인은 무작위 초기자세·도메인 무작위화(마찰·질량·강성)·외란 타이밍
이다. 정책은 같아도 매번 다른 조건에서 걷는다.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source"))
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX   # noqa: E402

TAU_SAT, SKIP = 3.16, 100
PRIO = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}


def metrics(p):
    z = np.load(p, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    dt = float(z["ctrl_dt"]) if "ctrl_dt" in z else 0.02
    e, rr, fa, st, wt, rt = {}, [], [], [], [], []
    for c in conds:
        v = z[f"{c}__v_base"][SKIP:]; w = z[f"{c}__w_base"][SKIP:]; cmd = z[f"{c}__cmd"]
        e[c] = (float(abs(w.mean() - cmd[2])) if c == "turn"
                else float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2])))
        g = z[f"{c}__grav"][SKIP:]
        r = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
        fa.append(float(100 * (np.abs(r) > 40).mean()))
        if c == "stop":
            continue
        rr.append(float(r.std()))
        t = z[f"{c}__tau"][SKIP:][..., ACT_LEG_JOINT_IDX]; a = np.abs(t)
        st.append(float(100 * (a > TAU_SAT * 0.995).mean()))
        d = z[f"{c}__dq"][SKIP:]; n = min(len(t), len(d))
        wt.append(float(np.abs(t[:n] * d[:n]).sum(-1).mean()))
        rt.append(float(np.diff(r, axis=0).std() / dt))
    return {
        "점수": sum(PRIO[k] * e[k] for k in PRIO) / sum(PRIO.values()),
        "앞뒤": (e["forward"] + e["backward"]) / 2, "회전": e["turn"],
        "옆": (e["left"] + e["right"]) / 2,
        "포화%": float(np.mean(st)), "일률W": float(np.mean(wt)),
        "roll속": float(np.mean(rt)), "rollRMS": float(np.mean(rr)),
        "낙상%": float(max(fa)),
    }


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        raise SystemExit("npz 를 2 개 이상 주세요")
    rows = [metrics(p) for p in paths]
    keys = list(rows[0])
    print(f"\n  같은 체크포인트 {len(rows)} 회 반복 측정")
    print(f"  {'지표':<9}{'평균':>10}{'표준편차':>10}{'최소':>10}{'최대':>10}{'폭/평균':>10}")
    print("  " + "-" * 59)
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        rng = v.max() - v.min()
        rel = f"{100*rng/abs(v.mean()):.0f}%" if abs(v.mean()) > 1e-9 else "—"
        print(f"  {k:<9}{v.mean():>10.4f}{v.std():>10.4f}{v.min():>10.4f}{v.max():>10.4f}{rel:>10}")
    print("\n  → 이 폭보다 작은 차이는 정책 차이로 읽지 말 것.\n")


if __name__ == "__main__":
    main()
