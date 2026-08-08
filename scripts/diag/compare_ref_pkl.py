#!/usr/bin/env python3
"""레퍼런스 보행 pkl 두 개를 같은 명령 격자 위에서 수치로 비교한다.

왜 필요한가: 레퍼런스를 다시 만들 이유(질량 변경, 높이 변경, 격자 수정)는 자주
생기는데, "새로 만든 게 실제로 얼마나 달라졌나" 를 잴 방법이 없었다. 눈으로
재생해 보는 것으로는 1 도짜리 차이를 못 잡는다. 차이가 무시할 만하면 재학습을
안 해도 되고, 크면 어느 관절이 움직였는지 바로 나온다.

    python3 scripts/diag/compare_ref_pkl.py \
        --a source/.../data/ref_g115.pkl --b source/.../data/ref_g115m.pkl

`~/.placo-env` 가 아니라 torch 가 있는 환경(~/.odm-tools 또는 IsaacLab)에서 돈다.
"""

import argparse
import sys

sys.path.insert(0, "source")

import torch  # noqa: E402

from open_duck_mini_isaaclab.joint_order import REF_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import (  # noqa: E402
    PolyReferenceMotion,
)

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="기준 pkl")
ap.add_argument("--b", required=True, help="비교 pkl")
ap.add_argument("--top", type=int, default=6, help="차이 큰 관절 몇 개를 볼지")
args = ap.parse_args()

A = PolyReferenceMotion(args.a, device="cpu")
B = PolyReferenceMotion(args.b, device="cpu")
if A.nb_steps_in_period != B.nb_steps_in_period:
    raise SystemExit(
        f"주기 길이가 다르다: {A.nb_steps_in_period} vs {B.nb_steps_in_period} — 비교 불가"
    )
N = A.nb_steps_in_period

# calc_home.py 와 같은 명령 집합. 정지·전후·좌우·회전을 모두 훑는다.
CMDS = [
    (0.0, 0.0, 0.0), (0.10, 0.0, 0.0), (-0.10, 0.0, 0.0),
    (0.0, 0.10, 0.0), (0.0, -0.10, 0.0),
    (0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.10, 0.0, 0.5),
]

fa, fb = [], []
for dx, dy, dth in CMDS:
    idx = torch.arange(N)
    fa.append(A.get_reference_motion(torch.full((N,), dx), torch.full((N,), dy),
                                     torch.full((N,), dth), idx)[:, 0:14])
    fb.append(B.get_reference_motion(torch.full((N,), dx), torch.full((N,), dy),
                                     torch.full((N,), dth), idx)[:, 0:14])
Fa, Fb = torch.cat(fa), torch.cat(fb)
D = (Fb - Fa)
deg = 180.0 / torch.pi

print("=" * 74)
print(f"A  {args.a}")
print(f"B  {args.b}")
print(f"   {len(CMDS)} 명령 x {N} 프레임 = {Fa.shape[0]} 샘플, 관절 14")
print("-" * 74)
print(f"  전체 최대차   {float(D.abs().max()) * deg:7.3f}°")
print(f"  전체 RMS      {float(D.pow(2).mean().sqrt()) * deg:7.3f}°")
print(f"  A 자체 진폭   {float((Fa.max(0).values - Fa.min(0).values).mean()) * deg:7.3f}° (관절 평균 p-p)")
print("-" * 74)
names = REF_JOINT_NAMES[:14]
per = [(float(D[:, i].abs().max()) * deg, float(D[:, i].mean()) * deg, names[i]) for i in range(14)]
per.sort(reverse=True)
print(f"  {'관절':22} {'최대차':>9} {'평균 이동':>10}")
for mx, mn, nm in per[: args.top]:
    print(f"  {nm:22} {mx:8.3f}° {mn:+9.3f}°")
print("=" * 74)
