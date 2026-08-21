#!/usr/bin/env python3
"""6 방향 **토크·일률** 그림. 목표가 토크·효율로 바뀐 뒤 판정에 쓰는 그래프.

    python3 scripts/diag/plot_torque_power.py v75            # 단독
    python3 scripts/diag/plot_torque_power.py v75 --vs v61   # 실기 기준선과 비교

`~/odm_out/gait_<ver>.npz` (= `odm measure` 산출물) 를 읽는다.

## 왜 이 그림인가

2026-08-20 에 목표가 추종 정확도에서 **피크 토크·보행 효율**로 바뀌었다.
그런데 기존 그래프(`plot_best.py`)는 속도 추종과 관절 궤적만 그린다. 판정에
쓰는 숫자를 눈으로 볼 수단이 없으면 표만 쌓인다.

## 무엇을 그리나

    좌상  방향별 **포화율** — 한계 3.16 N·m 에 붙어 있는 시간 비율. p99 는 숫자로
    우상  방향별 일률 |tau . omega| [W]
    좌하  다리 10 축별 p99 |tau| — 어느 관절이 한계에 물리는지
    우하  **왼발 접지 순간에 맞춰 접은** 일률 — 접지에서 어떻게 뛰는지

**피크 값만으로는 정책이 안 갈린다** — 정책 일곱 개가 전부 p99 3.16 이었다.
p99 를 막대로 그리면 여섯 방향이 전부 같은 높이가 되어 규약 1 의 "신호가
납작해진다" 에 정확히 걸린다. 그래서 **막대는 포화율**로 그리고 p99 는 숫자로만
적는다.

우하도 처음엔 시간축을 그대로 겹쳤는데 6 방향이 스파게티가 됐다. 고정 주기로
접어 봤더니 이번엔 **위상이 안 맞아 주기 구조가 사라졌다** — 측정 시작 시점이
걸음의 어디인지 모르기 때문이다. 그래서 **왼발 접지가 시작되는 순간(상승
에지)에 맞춰** 한 주기를 잘라 겹친다. 이러면 0 ms 가 항상 착지 순간이다.

레퍼런스 궤적에는 토크가 없다. 그래서 `--vs` 로 **다른 정책**을 파선/연한
막대로 겹쳐 "좋아졌는가" 를 볼 수 있게 했다 (규약 3 의 취지).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source"))
from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    ACTUATOR_JOINT_NAMES, ACT_LEG_JOINT_IDX,
)

for _f in ("NanumGothic", "AppleGothic", "Apple SD Gothic Neo"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

#: 스폰 직후 자세를 잡는 과도구간. 축 계산에서도 뺀다 (규약 9).
WARM = 100
#: 실효 토크 한계 [N·m]. effort_limit 4.1 이 아니라 토크-속도 모델이 여기서 자른다.
TAU_SAT = 3.16
DIRS = [("forward", "전진"), ("backward", "후진"), ("left", "좌"),
        ("right", "우"), ("turn", "회전"), ("stop", "정지")]
JOINTS = [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX]
_JKO = {"hip_yaw": "고관절 yaw", "hip_roll": "고관절 roll", "hip_pitch": "고관절 pitch",
        "knee": "무릎", "ankle": "발목"}
JLABEL = [("왼 " if j.startswith("left_") else "오른 ")
          + _JKO[j.split("_", 1)[1]] for j in JOINTS]


def load(ver):
    p = os.path.expanduser(f"~/odm_out/gait_{ver}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"없다: {p}")
    z = np.load(p, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    dt = float(z["ctrl_dt"]) if "ctrl_dt" in z else 0.02
    out = {}
    for c, _ in DIRS:
        if c not in conds:
            continue
        t = z[f"{c}__tau"][WARM:][..., ACT_LEG_JOINT_IDX]
        d = z[f"{c}__dq"][WARM:]
        n = min(len(t), len(d))
        a = np.abs(t)
        out[c] = dict(
            p99=float(np.percentile(a.max(-1), 99)),
            mean=float(a.sum(-1).mean()),
            sat=float(100.0 * (a > TAU_SAT * 0.995).mean()),
            watt=float(np.abs(t[:n] * d[:n]).sum(-1).mean()),
            watt_env=np.abs(t[:n] * d[:n]).sum(-1),          # [T, env]
            contact=z[f"{c}__feet"][WARM:][:n, :, 0],          # 왼발 접지 [T, env]
            per_joint=np.percentile(a, 99, axis=(0, 1)),
        )
    ck = os.path.basename(str(z["checkpoint"]))
    T = int(z["gait_period_steps"]) if "gait_period_steps" in z else 27
    return out, dt, ck, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ver")
    ap.add_argument("--vs", default=None, help="비교로 겹칠 다른 버전")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    A, dt, ckA, gait_T = load(a.ver)
    B, ckB = (load(a.vs)[0], load(a.vs)[2]) if a.vs else (None, None)
    keys = [(k, ko) for k, ko in DIRS if k in A]
    x = np.arange(len(keys))
    w = 0.38 if B else 0.55

    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

    # ── 좌상: 방향별 포화율 ──────────────────────────────────────────────
    p = ax[0, 0]
    p.bar(x - (w / 2 if B else 0), [A[k]["sat"] for k, _ in keys], w,
          color="#2b6cb0", label=a.ver)
    if B:
        p.bar(x + w / 2, [B[k]["sat"] for k, _ in keys], w,
              color="#2b6cb0", alpha=0.35, hatch="//", label=a.vs)
    mv = [A[k]["sat"] for k, _ in keys] + ([B[k]["sat"] for k, _ in keys] if B else [])
    top = max(mv) * 1.30
    for i, (k, _) in enumerate(keys):
        p.text(i - (w / 2 if B else 0), A[k]["sat"] + top * 0.02,
               f"{A[k]['sat']:.2f}%\np99 {A[k]['p99']:.2f}", ha="center",
               fontsize=8, color="#2b6cb0", linespacing=1.3)
        if B:
            p.text(i + w / 2, B[k]["sat"] + top * 0.02,
                   f"{B[k]['sat']:.2f}%\np99 {B[k]['p99']:.2f}", ha="center",
                   fontsize=8, color="#718096", linespacing=1.3)
    p.set_xticks(x); p.set_xticklabels([ko for _, ko in keys])
    p.set_ylabel(f"|tau| 가 한계 {TAU_SAT} N·m 에 붙어 있는 시간 비율  [%]")
    p.set_title("방향별 토크 포화율 — p99 는 모든 정책이 한계라 안 갈린다", fontsize=11)
    p.set_ylim(0, top)
    p.grid(axis="y", alpha=0.25); p.legend(fontsize=9, loc="upper right")

    # ── 우상: 방향별 일률 ────────────────────────────────────────────────
    p = ax[0, 1]
    p.bar(x - (w / 2 if B else 0), [A[k]["watt"] for k, _ in keys], w,
          color="#2f855a", label=a.ver)
    if B:
        p.bar(x + w / 2, [B[k]["watt"] for k, _ in keys], w,
              color="#2f855a", alpha=0.35, hatch="//", label=a.vs)
    for i, (k, _) in enumerate(keys):
        p.text(i - (w / 2 if B else 0), A[k]["watt"] + 0.15,
               f"{A[k]['watt']:.1f}", ha="center", fontsize=8, color="#2f855a")
    p.set_xticks(x); p.set_xticklabels([ko for _, ko in keys])
    p.set_ylabel("기계적 일률  |tau · omega| 합의 평균  [W]")
    mv = [A[k]["watt"] for k, _ in keys] + ([B[k]["watt"] for k, _ in keys] if B else [])
    p.set_ylim(0, max(mv) * 1.22)
    p.set_title("방향별 보행 효율 — 낮을수록 좋다", fontsize=11)
    p.grid(axis="y", alpha=0.25); p.legend(fontsize=9)

    # ── 좌하: 관절별 p99 토크 ────────────────────────────────────────────
    p = ax[1, 0]
    xs = np.arange(len(JOINTS))
    mv = [k for k, _ in keys if k != "stop"]
    for k, ko in [(k, ko) for k, ko in keys if k != "stop"]:
        p.plot(xs, A[k]["per_joint"], marker="o", ms=4, lw=1.6, label=ko)
    p.axhline(TAU_SAT, color="#c53030", ls="--", lw=1.2)
    p.set_xticks(xs); p.set_xticklabels(JLABEL, rotation=40, ha="right", fontsize=8)
    p.set_ylabel("관절별 |tau| p99  [N·m]")
    p.set_title("어느 관절이 한계에 물리는가 (정지 제외)", fontsize=11)
    p.set_ylim(0, TAU_SAT * 1.12)
    p.grid(alpha=0.25); p.legend(fontsize=8, ncol=3)

    # ── 우하: 왼발 접지에 맞춰 접은 일률 ─────────────────────────────────
    p = ax[1, 1]
    T = gait_T
    for k, ko in keys:
        if k == "stop":
            continue                       # 정지는 접지 전환이 없어 위상이 없다
        wv, ct = A[k]["watt_env"], A[k]["contact"]
        cycles = []
        for e in range(wv.shape[1]):
            c = ct[:, e]
            rise = np.flatnonzero((c[1:] > 0.5) & (c[:-1] <= 0.5)) + 1
            for i0 in rise:
                if i0 + T <= len(wv):
                    cycles.append(wv[i0:i0 + T, e])
        if len(cycles) < 5:
            continue
        f = np.array(cycles)
        ph = np.arange(T) * dt * 1000.0
        ln, = p.plot(ph, np.median(f, axis=0), lw=1.9, label=f"{ko} (n={len(f)})")
        p.fill_between(ph, np.percentile(f, 25, axis=0), np.percentile(f, 75, axis=0),
                       color=ln.get_color(), alpha=0.15, lw=0)
    p.axvline(0, color="#4a5568", lw=1.0, ls=":")
    p.text(T * dt * 1000 * 0.012, p.get_ylim()[1] * 0.96, "왼발 착지",
           fontsize=8, color="#4a5568", va="top")
    p.set_xlabel(f"왼발 착지 이후 경과 [ms]   (한 걸음 {T*dt*1000:.0f} ms)")
    p.set_ylabel("순시 일률  [W]")
    p.set_title("착지에 맞춰 접은 일률 — 굵은 선 중앙값, 띠 25~75 백분위", fontsize=11)
    p.grid(alpha=0.25); p.legend(fontsize=8, ncol=2)

    tot_w = np.mean([A[k]["watt"] for k, _ in keys if k != "stop"])
    tot_s = np.mean([A[k]["sat"] for k, _ in keys if k != "stop"])
    sub = (f"{a.ver}  ({ckA})   이동 5방향 평균: 일률 {tot_w:.1f} W · 포화 {tot_s:.2f} %")
    if B:
        bw = np.mean([B[k]["watt"] for k, _ in keys if k != "stop"])
        bs = np.mean([B[k]["sat"] for k, _ in keys if k != "stop"])
        sub += f"      비교 {a.vs} ({ckB}): {bw:.1f} W · {bs:.2f} %"
    fig.suptitle("6방향 토크·일률\n" + sub, fontsize=13)

    out = a.out or os.path.expanduser(f"~/odm_out/{a.ver}_6방향_토크일률.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
