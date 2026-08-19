#!/usr/bin/env python3
"""cfg 독스트링에서 **버전 색인**을 만든다 (`docs/versions.md`).

    python3 scripts/diag/versions.py

## 왜

2026-08-20, 사용자: "v 말고 이름에 좀 주석 달아줘 알아보기 쉽게."
v25 를 넘어가면서 번호만으로는 무엇을 시험했는지 알 수 없게 됐다. 각 cfg 는
독스트링 첫 줄에 이미 한 줄 요약을 갖고 있으므로 그걸 모아 표로 만든다 —
따로 관리하는 목록은 반드시 어긋나기 때문에 **생성**한다.

점수·roll·낙상은 `scoreboard.py` 와 **같은 정의**를 쓴다. 다르면 두 표를
나란히 놓을 수 없다 (처음엔 roll 을 전진만으로 재서 어긋났다).
"""
from __future__ import annotations

import os
import pathlib
import re

import numpy as np

PRIO_W = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}
FALL_DEG = 40.0
SKIP = 100
ROOT = pathlib.Path(__file__).resolve().parents[2]


def metrics(ver: str):
    p = os.path.expanduser(f"~/odm_out/gait_{ver.lower()}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    e, roll, fall = {}, [], []
    for c in conds:
        v = z[f"{c}__v_base"][SKIP:]
        w = z[f"{c}__w_base"][SKIP:]
        cmd = z[f"{c}__cmd"]
        e[c] = (float(abs(w.mean() - cmd[2])) if c == "turn"
                else float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2])))
        k = f"{c}__grav"
        if k in z:
            g = z[k][SKIP:]
            r = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
            if c != "stop":
                roll.append(float(r.std()))
            fall.append(float(100.0 * (np.abs(r) > FALL_DEG).mean()))
    if not all(k in e for k in PRIO_W):
        return None
    return (sum(PRIO_W[k] * e[k] for k in PRIO_W) / sum(PRIO_W.values()),
            float(np.mean(roll)) if roll else None,
            max(fall) if fall else None)


def main():
    src = (ROOT / "source/open_duck_mini_isaaclab/tasks/velocity/joystick_env_cfg.py").read_text()
    rows = []
    for m in re.finditer(
            r'^class JoystickEnvCfg_(V\d+\w*)\(JoystickEnvCfg_(\w+)\):\s*\n\s*"""(.+?)$',
            src, re.M):
        ver, par, first = m.group(1), m.group(2), m.group(3).strip()
        t = re.sub(r"^imitation_\w+\s*[—-]\s*", "", first)
        t = t.replace("**", "").rstrip('."')
        rows.append((ver, par, t))
    rows.sort(key=lambda r: (len(r[0]), r[0]))

    out = [
        "# 버전 색인",
        "",
        "`scripts/diag/versions.py` 가 cfg 독스트링에서 생성한다. 손으로 고치지 말 것.",
        "",
        "점수는 6 방향 우선순위 가중(앞뒤 3 : 회전 2 : 옆 1), 낮을수록 좋다.",
        "roll RMS 는 이동 5 방향 평균, 낙상은 |roll| > 40 도 표본 비율의 최댓값 —",
        "`scoreboard.py` 와 같은 정의다. 빈칸은 아직 측정하지 않은 버전이다.",
        "",
        "| 버전 | 부모 | 무엇을 바꿨나 | 점수 | rollRMS | 낙상 |",
        "|---|---|---|---|---|---|",
    ]
    for ver, par, t in rows:
        r = metrics(ver)
        a = f"{r[0]:.4f}" if r else ""
        b = f"{r[1]:.2f}" if r and r[1] is not None else ""
        c = f"{r[2]:.1f}%" if r and r[2] is not None else ""
        out.append(f"| **{ver}** | {par} | {t[:80]} | {a} | {b} | {c} |")
    (ROOT / "docs/versions.md").write_text("\n".join(out) + "\n")
    print(f"docs/versions.md 갱신 — {len(rows)} 개 버전")


if __name__ == "__main__":
    main()
