"""좌우 미러 검증. Isaac Sim 없이 돈다 (torch 만 필요).

미러가 틀리면 학습이 **터지지 않고 조용히** 나빠진다. 그래서 눈으로 못 보는
성질들을 여기서 못 박는다.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.symmetry import (  # noqa: E402
    MIRROR_JOINT_PERM,
    MIRROR_JOINT_SIGN,
    NUM_JOINTS,
    build_obs_layout,
    mirror_joint_vector,
    mirror_observation,
)


def test_perm_is_an_involution():
    """뒤집기를 두 번 하면 제자리여야 한다. 아니면 미러가 아니다."""
    for j in range(NUM_JOINTS):
        assert MIRROR_JOINT_PERM[MIRROR_JOINT_PERM[j]] == j


def test_signs_are_consistent_both_ways():
    """j -> perm(j) 로 갈 때와 돌아올 때의 부호를 곱하면 +1 이어야 한다."""
    for j in range(NUM_JOINTS):
        assert MIRROR_JOINT_SIGN[j] * MIRROR_JOINT_SIGN[MIRROR_JOINT_PERM[j]] == 1.0


def test_perm_maps_left_to_right_by_name():
    """유도된 매핑이 실제로 왼쪽<->오른쪽을 짝지었는지 이름으로 확인한다.
    숫자만 맞고 엉뚱한 관절끼리 짝지어졌을 수도 있다."""
    for j, name in enumerate(ACTUATOR_JOINT_NAMES):
        partner = ACTUATOR_JOINT_NAMES[MIRROR_JOINT_PERM[j]]
        if name.startswith("left_"):
            assert partner == name.replace("left_", "right_", 1), f"{name} -> {partner}"
        elif name.startswith("right_"):
            assert partner == name.replace("right_", "left_", 1), f"{name} -> {partner}"
        else:  # 머리/목은 가운데라 자기 자신
            assert partner == name, f"{name} -> {partner}"


def test_joint_mirror_is_an_involution():
    x = torch.randn(5, NUM_JOINTS)
    assert torch.allclose(mirror_joint_vector(mirror_joint_vector(x)), x, atol=1e-6)


def test_head_yaw_roll_flip_but_pitch_does_not():
    """가운데 관절은 자기 자신으로 가되 부호 규칙이 갈린다 —
    yaw/roll 은 좌우를 뒤집으면 반대가 되고, pitch 는 그대로다."""
    idx = {n: i for i, n in enumerate(ACTUATOR_JOINT_NAMES)}
    x = torch.zeros(1, NUM_JOINTS)
    for name, expect_flip in [("head_yaw", True), ("head_roll", True),
                              ("head_pitch", False), ("neck_pitch", False)]:
        x.zero_()
        x[0, idx[name]] = 1.0
        got = mirror_joint_vector(x)[0, idx[name]].item()
        assert got == (-1.0 if expect_flip else 1.0), f"{name}: {got}"


def test_observation_mirror_is_an_involution():
    layout = build_obs_layout(use_path_frame=True, use_gravity_obs=True)
    width = sum(w for _, w, _ in layout)
    assert width == 107, f"Grav/Tall 태스크의 관측은 107 이어야 한다 (계산 {width})"
    obs = torch.randn(8, width)
    assert torch.allclose(mirror_observation(mirror_observation(obs, layout), layout), obs, atol=1e-6)


def test_observation_width_matches_each_task_variant():
    """path/gravity 조합마다 폭이 실제 태스크와 맞는지."""
    assert sum(w for _, w, _ in build_obs_layout(False, False)) == 101  # 기본
    assert sum(w for _, w, _ in build_obs_layout(True, False)) == 104   # Path (v25/v26)
    assert sum(w for _, w, _ in build_obs_layout(True, True)) == 107    # Grav/Tall (v27/v28)


def test_wrong_width_is_rejected_loudly():
    """관측 조립 순서가 바뀌면 조용히 틀리는 대신 즉시 실패해야 한다."""
    layout = build_obs_layout(True, True)
    try:
        mirror_observation(torch.randn(2, 106), layout)
    except ValueError as exc:
        assert "107" in str(exc)
    else:
        raise AssertionError("폭이 틀린데 통과했습니다")


def test_gyro_and_accel_use_different_rules():
    """자이로는 유사벡터, 가속도는 참벡터 — 같은 부호를 먹이면 조용히 틀린다."""
    layout = build_obs_layout(True, True)
    width = sum(w for _, w, _ in layout)
    obs = torch.zeros(1, width)
    obs[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])   # gyro
    obs[0, 3:6] = torch.tensor([1.0, 2.0, 3.0])   # accel
    m = mirror_observation(obs, layout)[0]
    assert torch.allclose(m[0:3], torch.tensor([-1.0, 2.0, -3.0])), f"gyro {m[0:3]}"
    assert torch.allclose(m[3:6], torch.tensor([1.0, -2.0, 3.0])), f"accel {m[3:6]}"


def test_lateral_and_yaw_commands_flip_but_forward_does_not():
    layout = build_obs_layout(True, True)
    obs = torch.zeros(1, sum(w for _, w, _ in layout))
    obs[0, 6:13] = torch.tensor([0.15, 0.20, 1.0, 0.1, 0.2, 0.3, 0.4])
    m = mirror_observation(obs, layout)[0, 6:13]
    # float32 라 == 로 비교하면 0.15 가 0.150000005 로 어긋난다
    assert abs(m[0].item() - 0.15) < 1e-6, "전진 명령은 그대로여야 한다"
    assert abs(m[1].item() + 0.20) < 1e-6, "횡방향 명령은 뒤집혀야 한다"
    assert abs(m[2].item() + 1.0) < 1e-6, "회전 명령은 뒤집혀야 한다"


def test_feet_contact_swaps():
    layout = build_obs_layout(True, True)
    width = sum(w for _, w, _ in layout)
    off = sum(w for _, w, _ in layout[:9])  # contact 구간 시작
    obs = torch.zeros(1, width)
    obs[0, off : off + 2] = torch.tensor([1.0, 0.0])  # 왼발만 접지
    m = mirror_observation(obs, layout)[0, off : off + 2]
    assert m.tolist() == [0.0, 1.0], "왼발 접지는 뒤집으면 오른발이 되어야 한다"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")


# ── 관측 조립 순서가 어긋나는 것을 잡는다 ──────────────────────────────────
# symmetry.build_obs_layout 은 joystick_env.py 의 `state = torch.cat([...])`
# 순서를 **복제**하고 있다. 누가 환경 쪽 순서를 바꾸면 총 폭은 그대로여도
# 미러가 조용히 틀린다 — 폭 검사로는 절대 못 잡는 종류의 오류다.
# 그래서 소스를 실제로 읽어 대조한다.

_ENV_SRC = os.path.join(
    os.path.dirname(__file__), "..", "source", "open_duck_mini_isaaclab",
    "tasks", "velocity", "joystick_env.py",
)

# 환경 쪽 표현식 -> layout 이름
_EXPR_TO_LAYOUT = {
    "gyro": "gyro",
    "accel": "accel",
    "self._command": "command",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_scaled": "joint_vel",
    "self._last_act": "last_act",
    "self._last_last_act": "last_last_act",
    "self._last_last_last_act": "last_last_last_act",
    "self._motor_targets": "motor_targets",
    "contact": "contact",
    "imitation_phase": "imitation_phase",
    "self._path_error()": "path_error",
    "projected_gravity_b": "gravity",
}


def _env_state_order():
    """joystick_env.py 의 state 조립 순서를 읽어 layout 이름 목록으로 돌려준다."""
    with open(_ENV_SRC) as f:
        src = f.read()
    start = src.index("state = torch.cat(")
    end = src.index("dim=-1,", start)
    block = src[start:end]

    order = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for expr, name in _EXPR_TO_LAYOUT.items():
            if expr in line and name not in order:
                order.append(name)
                break
    return order


def test_layout_matches_env_state_assembly_order():
    env_order = _env_state_order()
    layout_order = [n for n, _, _ in build_obs_layout(use_path_frame=True, use_gravity_obs=True)]
    assert env_order == layout_order, (
        "관측 조립 순서가 어긋났다.\n"
        f"  joystick_env.py : {env_order}\n"
        f"  symmetry.py     : {layout_order}\n"
        "미러가 조용히 틀리게 된다 — symmetry.build_obs_layout 을 맞추세요."
    )
