#!/usr/bin/env python3
"""레퍼런스 보행 한 주기를 실기 재생용 JSON 으로 뽑는다 (데스크탑에서 실행).

Jetson 에는 torch 도 이 패키지도 없다. 그래서 다항식 레퍼런스를 여기서 풀어
**평범한 JSON** 으로 떨어뜨리고, Jetson 쪽 play_ref_gait.py 는 그걸 읽기만 한다.

    ~/Desktop/IsaacLab/isaaclab.sh -p scripts/hw/export_ref_gait.py --vx 0.15
    scp /tmp/ref_gait.json parksuho@192.168.137.7:~/

**URDF 한계로 클립한다.** 레퍼런스는 한계를 넘는다 -- 명령·위상을 훑어보면
표본의 55 % 가 어느 한 관절이라도 하드 리밋 밖이다. 다만 **넘는 양은 작아서**
실제 보행 명령에서는 최대 2.4° 다 (right_knee). 그래서 클립이 궤적을 거의
안 바꾸면서 기구 스토퍼를 밀어붙이는 것만 막아준다. 스토퍼에 밀어붙인 전례가
있다 (docs/handoff/project_hardware_bringup_2026-08-06.md §1).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, "source")

import torch  # noqa: E402

from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import (  # noqa: E402
    PolyReferenceMotion,
)
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.hardware_map import JOINT_LIMIT_RAD  # noqa: E402

DEFAULT_PKL = "source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--vx", type=float, default=0.0, help="전진 명령 m/s")
    ap.add_argument("--vy", type=float, default=0.0, help="횡보 명령 m/s")
    ap.add_argument("--wz", type=float, default=0.0, help="회전 명령 rad/s")
    ap.add_argument("--out", default="/tmp/ref_gait.json")
    # 한계에서 이만큼 안쪽까지만 허용한다. 기구 공차와 서보 추종 오차를 감안해
    # 한계에 정확히 붙이지 않는다.
    ap.add_argument("--margin-deg", type=float, default=1.0)
    args = ap.parse_args()

    prm = PolyReferenceMotion(args.pkl, device="cpu")
    n = prm.nb_steps_in_period
    jp = prm.get_reference_motion(
        torch.full((n,), args.vx), torch.full((n,), args.vy),
        torch.full((n,), args.wz), torch.arange(n),
    )[:, 0:14].tolist()

    margin = math.radians(args.margin_deg)
    frames, clipped = [], {}
    for row in jp:
        out = []
        for k, name in enumerate(ACTUATOR_JOINT_NAMES):
            lo, hi = JOINT_LIMIT_RAD[name]
            lo, hi = lo + margin, hi - margin
            v = row[k]
            c = max(lo, min(hi, v))
            if abs(c - v) > 1e-9:
                clipped[name] = max(clipped.get(name, 0.0), abs(c - v))
            out.append(c)
        frames.append(out)

    doc = {
        "joint_names": ACTUATOR_JOINT_NAMES,
        "command": {"vx": args.vx, "vy": args.vy, "wz": args.wz},
        "dt": 0.02,          # 레퍼런스 생성 시각 간격 (시뮬 정책 주기와 같다)
        "period": n,
        "margin_deg": args.margin_deg,
        "frames": frames,    # [period][14] rad, URDF 한계 - margin 으로 클립됨
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)

    print(f"[ok] {args.out}  주기 {n} 스텝 × dt 0.02 = {n * 0.02:.2f} s/사이클")
    print(f"     명령 vx={args.vx} vy={args.vy} wz={args.wz} · 여유 {args.margin_deg}°")
    if clipped:
        print("     클립된 관절 (한계-여유 를 넘던 최대량):")
        for k, v in sorted(clipped.items(), key=lambda x: -x[1]):
            print(f"       {k:16s} {math.degrees(v):5.2f}°")
    else:
        print("     클립 없음")

    lo_hi = [(min(r[k] for r in frames), max(r[k] for r in frames)) for k in range(14)]
    print("\n     관절별 재생 범위 (deg):")
    for k, name in enumerate(ACTUATOR_JOINT_NAMES):
        a, b = lo_hi[k]
        if abs(b - a) < 1e-6:
            continue
        print(f"       {name:16s} {math.degrees(a):+7.1f} .. {math.degrees(b):+7.1f}")


if __name__ == "__main__":
    main()
