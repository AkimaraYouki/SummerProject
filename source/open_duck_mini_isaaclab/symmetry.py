"""좌우 대칭 데이터 증강 (imitation_v29용).

rsl-rl 5.0.1 이 `symmetry_cfg` 로 대칭 학습을 지원한다. 미러 함수를 주면
"좌우가 같아야 하는 문제"로 학습 문제 자체를 바꾼다 — 리워드 계수가 아니라
구조 변경이고, 이번 rsl-rl 이식으로 새로 열린 선택지다.

**왜 쓰나.** 좌/우 명령의 추종 오차가 전진의 7배다 (v27@1500: 좌 0.037 / 우 0.033
vs 전진 0.005). v13 부터 계속 그랬고 인수인계 문서도 좌우 비대칭을 미해결로
적어놨다. 로봇은 물리적으로 좌우 대칭인데 정책이 그걸 모른다.

**관절 매핑은 가정하지 않고 유도했다** (scripts/diag/derive_mirror.py).
이 로봇 URDF 는 좌우 관절 축이 대칭이 아니라서, 순진하게 "roll/yaw 부호 반전"
으로 짜면 hip_roll / knee / ankle 세 관절의 부호가 틀린다. 터지지 않고 그냥
나쁜 정책이 나오므로 특히 위험하다.

**반사 규칙** (시상면 y=0 에 대한 반사).

    참벡터   (x, y, z) -> ( x, -y,  z)   가속도, 중력, 선속도
    유사벡터 (x, y, z) -> (-x,  y, -z)   각속도, 회전 명령
                                          (반사는 det = -1 이라 유사벡터는
                                           부호가 한 번 더 뒤집힌다)

이 구분을 놓치면 자이로와 가속도에 같은 부호를 먹여 조용히 틀린다.

**위상.** 좌우를 뒤집으면 디딤발과 흔드는 발이 바뀌므로 보행 위상이 반 주기
어긋난다. imitation_phase = (cos p, sin p) 는 p -> p + pi 가 되어 둘 다 부호가
뒤집힌다.
"""

from __future__ import annotations

import torch

# scripts/diag/derive_mirror.py 가 레퍼런스 데이터에서 유도한 값 (2026-07-30).
# 5개 명령쌍 x 27프레임 교차검증, 대합·부호정합·전관절 오차<=0.05rad 통과.
#   perm: 관절 j 의 좌우 짝
#   sign: 뒤집을 때 곱할 부호
MIRROR_JOINT_PERM = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
MIRROR_JOINT_SIGN = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]

NUM_JOINTS = 14

#: 참벡터 (반사에서 y 만 뒤집힘)
_TRUE_VEC = (1.0, -1.0, 1.0)
#: 유사벡터 (반사에서 x, z 가 뒤집힘)
_PSEUDO_VEC = (-1.0, 1.0, -1.0)

#: 명령 7차원: vx, vy, yaw_rate, neck_pitch, head_pitch, head_yaw, head_roll
_CMD_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0)

#: path 오차 3차원: 횡방향 오차, cos(방향오차), sin(방향오차)
#: 횡방향과 방향각이 뒤집히므로 cos 는 그대로, sin 만 뒤집힌다.
_PATH_SIGN = (-1.0, 1.0, -1.0)


def mirror_joint_vector(x: torch.Tensor) -> torch.Tensor:
    """[..., 14] 관절 벡터를 좌우로 뒤집는다."""
    perm = torch.as_tensor(MIRROR_JOINT_PERM, device=x.device)
    sign = torch.as_tensor(MIRROR_JOINT_SIGN, device=x.device, dtype=x.dtype)
    return x[..., perm] * sign


def _scaled(x: torch.Tensor, signs) -> torch.Tensor:
    return x * torch.as_tensor(signs, device=x.device, dtype=x.dtype)


def build_obs_layout(use_path_frame: bool, use_gravity_obs: bool):
    """관측 벡터의 구간 목록. joystick_env.py 의 `state` 조립 순서와 같아야 한다.

    (이름, 폭, 종류) 로 돌려준다. 종류는 아래 mirror_observation 이 해석한다.
    순서가 어긋나면 조용히 틀리므로, 총 길이를 호출부에서 반드시 검증할 것.
    """
    layout = [
        ("gyro", 3, "pseudo"),
        ("accel", 3, "true"),
        ("command", 7, "command"),
        ("joint_pos_rel", NUM_JOINTS, "joint"),
        ("joint_vel", NUM_JOINTS, "joint"),
        ("last_act", NUM_JOINTS, "joint"),
        ("last_last_act", NUM_JOINTS, "joint"),
        ("last_last_last_act", NUM_JOINTS, "joint"),
        ("motor_targets", NUM_JOINTS, "joint"),
        ("contact", 2, "feet"),
        ("imitation_phase", 2, "phase"),
    ]
    if use_path_frame:
        layout.append(("path_error", 3, "path"))
    if use_gravity_obs:
        layout.append(("gravity", 3, "true"))
    return layout


def mirror_observation(obs: torch.Tensor, layout) -> torch.Tensor:
    """[N, D] 관측을 좌우로 뒤집는다. D 는 layout 의 폭 합과 같아야 한다."""
    total = sum(w for _, w, _ in layout)
    if obs.shape[-1] != total:
        raise ValueError(
            f"관측 폭이 layout 과 다릅니다: obs {obs.shape[-1]} vs layout {total}. "
            "joystick_env.py 의 state 조립 순서가 바뀌었는지 확인하세요 — "
            "여기가 어긋나면 학습이 터지지 않고 조용히 망가집니다."
        )
    out, i = [], 0
    for _name, width, kind in layout:
        chunk = obs[..., i : i + width]
        if kind == "joint":
            out.append(mirror_joint_vector(chunk))
        elif kind == "pseudo":
            out.append(_scaled(chunk, _PSEUDO_VEC))
        elif kind == "true":
            out.append(_scaled(chunk, _TRUE_VEC))
        elif kind == "command":
            out.append(_scaled(chunk, _CMD_SIGN))
        elif kind == "path":
            out.append(_scaled(chunk, _PATH_SIGN))
        elif kind == "feet":
            out.append(chunk.flip(-1))  # 왼발 <-> 오른발
        elif kind == "phase":
            # 반 주기 어긋나므로 p -> p + pi, 즉 (cos, sin) 둘 다 부호 반전
            out.append(-chunk)
        else:
            raise ValueError(f"모르는 구간 종류: {kind}")
        i += width
    return torch.cat(out, dim=-1)
