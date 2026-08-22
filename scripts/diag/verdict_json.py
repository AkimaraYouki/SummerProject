#!/usr/bin/env python3
"""한 버전을 채점해 **한 줄 JSON**으로 뱉는다. 자동 큐가 분기 판단에 쓴다.

    python3 scripts/diag/verdict_json.py v79

`scoreboard.py` 는 사람이 읽는 표를 그린다. 셸 스크립트가 그 표를 파싱하면
서식이 바뀔 때마다 조용히 깨지므로, 기계가 읽을 출력을 따로 낸다.

`pass` 는 목표 지표 기준이다 (2026-08-20 에 목표가 토크·효율·토르소로 바뀌었다).
`obey` 는 6 방향 명령을 부호와 크기(명령의 50 % 이상)로 따라가는지 — 벌점을
세게 걸면 정책이 "안 움직이는 쪽" 으로 도망갈 수 있어 이것만은 문턱으로 쓴다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source"))
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX   # noqa: E402

TAU_SAT, SKIP, TRACK_FRAC = 3.16, 100, 0.5
PRIO = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}


def main():
    ver = sys.argv[1]
    p = os.path.expanduser(f"~/odm_out/gait_{ver}.npz")
    if not os.path.exists(p):
        print(json.dumps({"ver": ver, "ok": False, "why": "npz 없음"})); return
    z = np.load(p, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    dt = float(z["ctrl_dt"]) if "ctrl_dt" in z else 0.02
    e, rr, fa, st, wt, rt, bad = {}, [], [], [], [], [], []
    for c in conds:
        v = z[f"{c}__v_base"][SKIP:]; w = z[f"{c}__w_base"][SKIP:]; cmd = z[f"{c}__cmd"]
        e[c] = (float(abs(w.mean() - cmd[2])) if c == "turn"
                else float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2])))
        vb = v.mean(axis=(0, 1)); wz = float(w.mean())
        for tgt, act, nm in ((cmd[0], vb[0], "vx"), (cmd[1], vb[1], "vy"), (cmd[2], wz, "wz")):
            if abs(tgt) < 1e-6:
                continue
            if np.sign(tgt) != np.sign(act) or abs(act) < TRACK_FRAC * abs(tgt):
                bad.append(f"{c}.{nm}")
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
    o = {
        "ver": ver, "ok": True,
        "ck": os.path.basename(str(z["checkpoint"])),
        "score": round(sum(PRIO[k] * e[k] for k in PRIO) / sum(PRIO.values()), 4),
        "fb": round((e["forward"] + e["backward"]) / 2, 4),
        "turn": round(e["turn"], 4), "lr": round((e["left"] + e["right"]) / 2, 4),
        "sat": round(float(np.mean(st)), 3), "watt": round(float(np.mean(wt)), 2),
        "rrate": round(float(np.mean(rt)), 1), "roll": round(float(np.mean(rr)), 2),
        "fall": round(float(max(fa)), 2), "obey": not bad, "bad": bad,
    }
    print(json.dumps(o, ensure_ascii=False))


if __name__ == "__main__":
    main()
