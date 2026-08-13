#!/usr/bin/env python3
"""직진 녹화(dy=0, dθ=0)를 좌우 대칭으로 만든다. **적합 전에** 돈다.

## 왜 필요한가

2026-08-11 측정: placo 원본 녹화는 거의 대칭인데(무릎 진폭 좌우차 +3.3 %,
hip_pitch +7.7 %) `fit_poly.py` 를 거치면 +11.1 % / **+41.0 %** 로 벌어진다.
격자점마다 36 개 차원을 **독립으로** 15 차 다항식에 맞추므로, 좌우 다리가 서로
다른 잔차를 갖는 것을 막을 구조가 없다.

정책은 이 레퍼런스를 모방하도록 학습하므로 비대칭이 그대로 새겨진다 (v35~v41).

## 왜 녹화 단계인가 — 적합 후 대칭화는 실패했다

먼저 적합된 pkl 을 사후 대칭화해 봤다:
`frame_sym(t) = ½(frame(t) + M[frame_mirror((t+½) mod 1)])`.
**진폭이 748° 로 폭주했다.** 원인은 `mod 1` 위상 감김이다 — 다항식이 완전
주기적이지 않아 t=½ 에서 계단 불연속이 생기고, 15 차 다항식이 그걸 맞추려다
깁스 링잉을 일으킨다.

녹화는 500 프레임(18.5 주기)의 연속 시계열이라 그 문제가 없다. 반주기만큼
민 것과 평균내고 끝부분만 버리면 된다.

## 무엇을 하는가

직진 보행은 **자기 자신이 거울 짝**이다 — 오른다리 궤적은 왼다리를 반주기 민
것의 거울상이어야 한다. 그래서 파일 하나로 대칭화가 닫힌다:

    frame_sym[t] = ½ ( frame[t] + M[ frame[t + T/2] ] )

M 은 좌우 다리를 맞바꾸고 시상면 부호를 적용한다. 반주기 T/2 는 13.5 프레임
(주기 27 이 홀수)이라 13/14 를 선형보간한다 — 프레임 간격 20 ms 에서 이 오차는
관절각 0.5° 수준으로, 지우려는 비대칭(수십 %)보다 두 자릿수 작다.

**dy≠0 이나 dθ≠0 인 녹화는 건드리지 않는다.** 게걸음·회전은 물리적으로
비대칭이 맞고, 그 거울 짝은 다른 명령(dx,−dy,−dθ)의 **다른 파일**이라 위상
정렬이 필요해 자기완결적으로 못 한다.

## 사용

    python3 scripts/setup/symmetrize_recordings.py <녹화디렉터리>
"""
from __future__ import annotations

import json
import math
import os
import sys

#: 녹화 JSON 의 "Joints" 순서와 같아야 한다.
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
#: joint_order.LEG_MIRROR_SIGN 과 같아야 한다 (FK 전수탐색으로 얻은 값).
LEG_MIRROR_SIGN = {"hip_yaw": -1.0, "hip_roll": +1.0, "hip_pitch": +1.0,
                   "knee": -1.0, "ankle": -1.0}


def joint_mirror() -> tuple[list[int], list[float]]:
    perm = list(range(14))
    sign = [1.0] * 14
    for base, s in LEG_MIRROR_SIGN.items():
        li = JOINTS.index("left_" + base)
        ri = JOINTS.index("right_" + base)
        perm[li], perm[ri] = ri, li
        sign[li] = sign[ri] = s
    sign[JOINTS.index("head_yaw")] = -1.0
    sign[JOINTS.index("head_roll")] = -1.0
    return perm, sign


PERM, SIGN = joint_mirror()


def mirror_frame(fr: list[float], off: dict, size: dict) -> list[float]:
    """한 프레임(55 칸)을 시상면 거울 반사한다."""
    out = list(fr)

    def mirror_joint_block(base: int) -> None:
        src = fr[base:base + 14]
        for i in range(14):
            out[base + i] = SIGN[i] * src[PERM[i]]

    mirror_joint_block(off["joints_pos"])
    mirror_joint_block(off["joints_vel"])
    # 발끝 위치·속도: 좌우 교환 + y 반전
    for a, b in (("left_toe_pos", "right_toe_pos"), ("left_toe_vel", "right_toe_vel")):
        ia, ib = off[a], off[b]
        for j, s in enumerate((1.0, -1.0, 1.0)):          # x, y, z
            out[ia + j], out[ib + j] = s * fr[ib + j], s * fr[ia + j]
    # 몸통 위치·자세: y 와 quat 의 x,z 성분 반전 (quat 은 [x,y,z,w] 가정)
    rp, rq = off["root_pos"], off["root_quat"]
    out[rp + 1] = -fr[rp + 1]
    out[rq + 0], out[rq + 2] = -fr[rq + 0], -fr[rq + 2]
    # 월드 선속도 y 반전 · 각속도 x,z 반전
    lv, av = off["world_linear_vel"], off["world_angular_vel"]
    out[lv + 1] = -fr[lv + 1]
    out[av + 0], out[av + 2] = -fr[av + 0], -fr[av + 2]
    # 발 접지 좌우 교환
    fc = off["foot_contacts"]
    out[fc], out[fc + 1] = fr[fc + 1], fr[fc]
    return out


def symmetrize(path: str) -> tuple[bool, str]:
    with open(path) as f:
        rec = json.load(f)
    placo = rec.get("Placo", {})
    dy, dtheta = placo.get("dy", 0.0), placo.get("dtheta", 0.0)
    if abs(dy) > 1e-9 or abs(dtheta) > 1e-9:
        return False, f"직진 아님 (dy={dy}, dθ={dtheta})"

    off = rec["Frame_offset"][0]
    size = rec["Frame_size"][0]
    F = rec["Frames"]
    period_frames = placo["period"] * rec["FPS"]         # 27.0
    half = period_frames / 2.0                            # 13.5
    lo, frac = int(math.floor(half)), half - math.floor(half)

    n = len(F) - (lo + 1)
    if n <= 0:
        return False, "프레임이 너무 짧다"

    # 대칭화 전 비대칭 (무릎 진폭 좌우차) — 보고용
    def knee_amp(frames, side):
        i = off["joints_pos"] + JOINTS.index(f"{side}_knee")
        v = [fr[i] for fr in frames]
        return max(v) - min(v)

    before = (knee_amp(F, "right") - knee_amp(F, "left")) / max(knee_amp(F, "left"), 1e-9) * 100

    out = []
    for t in range(n):
        a = mirror_frame(F[t + lo], off, size)
        b = mirror_frame(F[t + lo + 1], off, size)
        shifted = [a[k] * (1.0 - frac) + b[k] * frac for k in range(len(a))]
        out.append([0.5 * (F[t][k] + shifted[k]) for k in range(len(F[t]))])

    after = (knee_amp(out, "right") - knee_amp(out, "left")) / max(knee_amp(out, "left"), 1e-9) * 100
    rec["Frames"] = out
    with open(path, "w") as f:
        json.dump(rec, f)
    return True, f"프레임 {len(F)} -> {n}, 무릎 진폭차 {before:+.1f}% -> {after:+.1f}%"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("사용법: symmetrize_recordings.py <녹화디렉터리>")
    d = sys.argv[1]
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    n_sym = 0
    for fn in files:
        ok, msg = symmetrize(os.path.join(d, fn))
        if ok:
            n_sym += 1
            print(f"  [대칭화] {fn}  {msg}")
    print(f"직진 녹화 {n_sym}개 대칭화 · 나머지 {len(files) - n_sym}개는 그대로 "
          f"(게걸음·회전은 비대칭이 물리적으로 맞다)")


if __name__ == "__main__":
    main()
