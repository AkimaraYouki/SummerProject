"""좌우 미러 매핑을 **레퍼런스 데이터에서 유도한다** (가정하지 않는다).

왜 유도해야 하나. 순진하게 "왼쪽 관절 i <-> 오른쪽 관절 i, roll/yaw 는 부호
반전"으로 짜면 틀린다. 이 로봇의 URDF 는 좌우 관절 축이 대칭이 아니다 —
같은 서 있는 자세에서 실측값이 이렇다:

    left_hip_pitch  +0.991   right_hip_pitch  +1.011   (같은 부호)
    left_knee       -1.785   right_knee       +1.816   (반대 부호)
    left_ankle      +0.865   right_ankle      -0.875   (반대 부호)

부호 규칙이 관절마다 다르다. 잘못 짜면 학습이 조용히 망가진다 — 터지지 않고
그냥 나쁜 정책이 나오므로 알아채기가 특히 어렵다.

**유도 원리.** 레퍼런스 보행은 명령 (dx, dy, dtheta) 로 매개변수화돼 있다.
로봇을 좌우로 뒤집으면 "왼쪽으로 dy 만큼, 반시계로 dtheta 만큼" 가는 보행은
"오른쪽으로 dy 만큼, 시계로 dtheta 만큼" 가는 보행이 된다. 전진 dx 는 그대로다.
게다가 뒤집으면 디딤발과 흔드는 발이 바뀌므로 **반 주기만큼 위상이 어긋난다.**

    ref(dx, +dy, +dtheta)[t][j]  ==  s_j * ref(dx, -dy, -dtheta)[t + N/2][perm(j)]

이 식을 만족하는 perm 과 s 를 여러 명령·여러 위상에 대해 찾는다. 특정 명령
하나에서만 맞는 우연한 대응을 배제하려고 여러 명령으로 교차 검증한다.

Isaac Sim 도 GPU 도 쓰지 않는다 — 레퍼런스 다항식만 읽는다.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))
from open_duck_mini_isaaclab.joint_order import REF_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import (  # noqa: E402
    PolyReferenceMotion,
)

# 좌우가 뒤집혀도 그대로인 명령 성분은 dx, 뒤집히는 성분은 dy 와 dtheta.
CMD_PAIRS = [
    ((0.10, 0.10, 0.0), (0.10, -0.10, 0.0)),
    ((0.00, 0.15, 0.0), (0.00, -0.15, 0.0)),
    ((0.10, 0.00, 0.5), (0.10, 0.00, -0.5)),
    ((0.00, 0.00, 0.8), (0.00, 0.00, -0.8)),
    ((0.10, 0.10, 0.5), (0.10, -0.10, -0.5)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="일치로 볼 최대 평균 절대 오차 (rad)")
    args = ap.parse_args()

    prm = PolyReferenceMotion(args.pkl, device="cpu")
    N = prm.nb_steps_in_period
    nj = len(REF_JOINT_NAMES)

    def frames(cmd, shift=0.0):
        """shift 는 프레임 단위. 소수면 이웃 두 프레임을 선형보간한다.

        보간이 필요한 이유: 주기가 홀수(27)라 '반 주기'가 13.5 프레임이다.
        정수 13 으로 자르면 관절각이 반 스텝만큼 어긋나고, 그 크기가
        진폭 0.3 rad x 2pi/27 / 2 ~ 0.035 rad 로 무시할 수 없다 — 실제로
        정수 이동만 썼을 때 잔차가 딱 그 정도(0.02~0.05)로 남았다.
        """
        lo, hi = int(np.floor(shift)), int(np.ceil(shift))
        w = shift - lo

        def at(sh):
            idx = (torch.arange(N) + sh) % N
            f = prm.get_reference_motion(
                torch.full((N,), cmd[0]), torch.full((N,), cmd[1]), torch.full((N,), cmd[2]), idx
            )
            return f[:, 0:nj].numpy()

        return at(lo) if lo == hi else (1.0 - w) * at(lo) + w * at(hi)

    # 각 (원본관절 a, 후보관절 b, 부호 s) 조합의 오차를 모든 명령쌍에 대해 누적
    err = np.full((nj, nj, 2), 0.0)
    for cmd_p, cmd_m in CMD_PAIRS:
        A = frames(cmd_p)                    # 원본
        B = frames(cmd_m, shift=N / 2.0)     # 뒤집힌 명령 + 반 주기 (홀수 주기라 보간)
        for a in range(nj):
            for b in range(nj):
                err[a, b, 0] += np.abs(A[:, a] - B[:, b]).mean()   # 부호 +
                err[a, b, 1] += np.abs(A[:, a] + B[:, b]).mean()   # 부호 -
    err /= len(CMD_PAIRS)

    print("=" * 78)
    print(f"미러 매핑 유도  ({len(CMD_PAIRS)}개 명령쌍 x {N}프레임, 반주기 {N/2.0} 이동)")
    print(f"레퍼런스: {args.pkl}")
    print("=" * 78)
    print(f"{'관절':<22}{'-> 대응 관절':<22}{'부호':>5}{'평균오차(rad)':>15}{'판정':>8}")
    print("-" * 78)

    perm, signs, ok_all = [], [], True
    for a in range(nj):
        flat = err[a].reshape(-1)
        k = int(np.argmin(flat))
        b, si = k // 2, k % 2
        s = 1.0 if si == 0 else -1.0
        e = err[a, b, si]
        ok = e <= args.tol
        ok_all &= ok
        perm.append(b)
        signs.append(s)
        print(f"{REF_JOINT_NAMES[a]:<22}{REF_JOINT_NAMES[b]:<22}{s:>+5.0f}{e:>15.4f}{'OK' if ok else '  안맞음':>8}")

    print("-" * 78)
    # 미러는 스스로의 역이어야 한다. perm 이 대합(involution)이 아니면 유도가 틀린 것이다.
    involution = all(perm[perm[a]] == a for a in range(nj))
    sign_ok = all(abs(signs[a] * signs[perm[a]] - 1.0) < 1e-9 for a in range(nj))
    print(f"대합(perm[perm[j]] == j)          : {'OK' if involution else '실패'}")
    print(f"부호 정합(s_j * s_perm(j) == 1)   : {'OK' if sign_ok else '실패'}")
    print(f"모든 관절이 허용오차 이내          : {'OK' if ok_all else '실패'}")

    if involution and sign_ok and ok_all:
        print("\n유도 성공. 아래를 그대로 쓰면 된다:\n")
        print(f"MIRROR_JOINT_PERM = {perm}")
        print(f"MIRROR_JOINT_SIGN = {[int(s) for s in signs]}")
    else:
        print("\n유도 실패. 이 매핑을 학습에 쓰면 안 된다.")
        print("레퍼런스가 좌우 대칭이 아니거나(placo 가 비대칭 보행을 냈거나),")
        print("반주기 위상 가정이 틀렸을 수 있다. 위 표의 '안맞음' 관절부터 볼 것.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
