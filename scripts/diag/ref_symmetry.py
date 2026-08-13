#!/usr/bin/env python3
"""레퍼런스 보행 pkl 의 **좌우 대칭성**과 **격자 충전율**을 잰다.

2026-08-10 에 실기 좌우 비대칭을 쫓다가, 원인이 정책이 아니라 레퍼런스에
있다는 걸 찾고 만들었다. 두 가지를 본다.

## 1. 좌우 대칭성

대칭 보행이면 오른다리 궤적은 왼다리를 반주기 민 것의 거울상이다. 진폭
좌우차와 중립 자세 어긋남으로 본다. 진폭차는 위상 시프트와 무관해서
(주기 27 이 홀수라 정수 반주기가 없는 문제를 안 탄다) 가장 믿을 만하다.

## 2. 격자 충전율 — 이쪽이 실제로 더 심각했다

pkl 은 (dx, dy, dθ) 격자마다 다항식을 갖는데, **녹화는 그 격자의 절반만
있다.** `auto_waddle.py` 5 단계의 속도 필터가 medium 프리셋에서 총속도가
slow(0.05) 이하이거나 fast(0.15) 초과인 녹화를 지우기 때문이다. 그런데
스윕은 ±0.222 m/s 까지 돈다.

빠진 칸은 런타임에서 `nearest_coeffs` 로 **가장 가까운 녹화로 폴백**된다
(poly_reference_motion.py). 조용히 일어나서 눈치채기 어렵다. 실측:

    ref_g125.pkl   녹화 119 / 격자 216 (55%)
    ref_g125s.pkl  녹화 143 / 격자 270 (53%)

그리고 하필 **직진 슬라이스**(dy≈0, dθ=0)에서 dx = 0.0 / +0.148 / +0.222 가
전부 빠져 있다. 그래서 **전진 속도를 바꿔도 레퍼런스가 안 바뀐다** — 전부
dx=+0.074 하나로 폴백된다. 저속에서 제자리걸음처럼 보이는 원인이다.

## 사용

    isaaclab.sh -p scripts/diag/ref_symmetry.py            # data/ 의 pkl 전부
    isaaclab.sh -p scripts/diag/ref_symmetry.py ref_g125.pkl ref_g125s.pkl
"""
from __future__ import annotations

import math
import os
import pickle
import sys

import torch

sys.path.insert(0, "source")
from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    LEG_MIRROR_SIGN,
    REF_LEG_JOINT_IDX,
)
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import (  # noqa: E402
    PolyReferenceMotion,
)

DATA = "source/open_duck_mini_isaaclab/reference_motion/data"
NAMES = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
LI, RI = REF_LEG_JOINT_IDX[:5], REF_LEG_JOINT_IDX[5:]
d = math.degrees


def grid_report(path: str) -> None:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    pts = {tuple(round(float(x), 6) for x in k.split("_")) for k in raw}
    dxs = sorted({p[0] for p in pts})
    dys = sorted({p[1] for p in pts})
    dts = sorted({p[2] for p in pts})
    total = len(dxs) * len(dys) * len(dts)
    fill = len(pts) / total * 100
    print(f"  격자 충전율 {len(pts)}/{total} = {fill:.0f} %" + ("   <<< 절반이 폴백" if fill < 90 else ""))
    print(f"    dx {dxs}")
    print(f"    dy {dys}" + ("" if any(abs(v) < 1e-9 for v in dys)
                             else "   <<< dy=0 이 없다 — 직진이 옆걸음으로 스냅된다"))
    # 직진 슬라이스: vy=0 / dθ=0 이 스냅되는 칸
    dy0 = min(dys, key=abs)
    dt0 = min(dts, key=abs)
    have = [x for x in dxs if (x, dy0, dt0) in pts]
    miss = [x for x in dxs if x not in have]
    print(f"    직진 슬라이스 (dy={dy0}, dθ={dt0}):  녹화 {have}")
    if miss:
        print(f"      빠짐 {miss}   <<< 전부 최근접 폴백 — 이 속도들은 레퍼런스가 같아진다")


def symmetry_report(path: str) -> None:
    prm = PolyReferenceMotion(path, device="cpu")
    n = prm.nb_steps_in_period
    # 격자 위 전진 속도 하나를 골라 본다 (스냅 때문에 격자 밖 값은 의미가 흐려진다).
    vx = max([x for x in prm.dxs if x > 0] or [0.0])
    fr = prm.get_reference_motion(
        torch.full((n,), float(vx)), torch.zeros(n), torch.zeros(n), torch.arange(n))
    jp = fr[:, 0:14]
    print(f"  대칭성 @ vx={vx:+.3f}, vy=0, dθ=0")
    print(f"    {'관절':10}{'좌 진폭':>10}{'우 진폭':>10}{'진폭차':>9}{'중립 어긋남':>12}")
    for k, nm in enumerate(NAMES):
        L, R = jp[:, LI[k]], jp[:, RI[k]]
        s = LEG_MIRROR_SIGN[nm]
        aL = float(L.max() - L.min())
        aR = float(R.max() - R.min())
        pct = (aR - aL) / max(aL, 1e-9) * 100
        off = d(float(R.mean() - s * L.mean()))
        flag = "  <<<" if abs(pct) > 10 or abs(off) > 1.5 else ""
        print(f"    {nm:10}{d(aL):9.2f}°{d(aR):9.2f}°{pct:+8.1f}%{off:+11.2f}°{flag}")


def main() -> None:
    args = sys.argv[1:]
    files = args or sorted(f for f in os.listdir(DATA) if f.endswith(".pkl"))
    for fn in files:
        path = fn if os.path.sep in fn else os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"\n=== {fn} ===  (없음)")
            continue
        print(f"\n=== {os.path.basename(path)} ===")
        grid_report(path)
        symmetry_report(path)


if __name__ == "__main__":
    main()
