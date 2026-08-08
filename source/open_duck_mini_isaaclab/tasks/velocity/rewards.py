"""Torch port of Open_Duck_Playground's playground/common/rewards.py
(the subset actually active in joystick.py's reward dict) plus
playground/open_duck_mini_v2/custom_rewards.py::reward_imitation.

All functions are batched: every tensor argument is [N, ...] and every
function returns an [N] tensor. Every formula/constant below is copied
verbatim from the JAX source — see docs/decisions.md for the one place
(reward_imitation's leg-joint index alignment) that needed real derivation
rather than direct transcription.
"""

from __future__ import annotations

import torch

from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX


def reward_tracking_lin_vel(commands: torch.Tensor, local_lin_vel: torch.Tensor, tracking_sigma: float) -> torch.Tensor:
    """commands: [N,>=2], local_lin_vel: [N,>=2] (base-frame x,y linear velocity)."""
    y_tol = 0.1
    error_x = (commands[:, 0] - local_lin_vel[:, 0]) ** 2
    error_y = torch.clamp(torch.abs(local_lin_vel[:, 1] - commands[:, 1]) - y_tol, min=0.0)
    lin_vel_error = error_x + error_y**2
    return torch.exp(-lin_vel_error / tracking_sigma)


def reward_tracking_ang_vel(commands: torch.Tensor, ang_vel: torch.Tensor, tracking_sigma: float) -> torch.Tensor:
    """commands: [N,>=3], ang_vel: [N,>=3] (base-frame gyro, z = yaw rate)."""
    ang_vel_error = (commands[:, 2] - ang_vel[:, 2]) ** 2
    return torch.exp(-ang_vel_error / tracking_sigma)


def cost_torques(torques: torch.Tensor) -> torch.Tensor:
    return torch.sum(torques**2, dim=-1)


def cost_action_rate(act: torch.Tensor, last_act: torch.Tensor) -> torch.Tensor:
    return torch.sum((act - last_act) ** 2, dim=-1)


def reward_alive(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.ones(num_envs, device=device)


def cost_stand_still(
    commands: torch.Tensor,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    default_pose: torch.Tensor,
) -> torch.Tensor:
    """ignore_head=False path only (the one joystick.py actually calls).

    `default_pose` 는 호출부가 정한다. 보통은 env 의 default_joint_pos 지만,
    cfg.standstill_joint_pos 가 있으면 그쪽이 들어온다 — 액션의 원점(보행 평균
    자세)과 정지 목표 자세는 목적이 달라서 분리했다 (joystick_env_cfg.py 주석).
    """
    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)
    pose_cost = torch.sum(torch.abs(qpos - default_pose), dim=-1)
    vel_cost = torch.sum(torch.abs(qvel), dim=-1)
    return (pose_cost + vel_cost) * (cmd_norm < 0.01).float()


def cost_upright_standstill(
    commands: torch.Tensor,
    projected_gravity_b: torch.Tensor,
) -> torch.Tensor:
    """정지 명령일 때 **몸통이 수직에서 벗어난 만큼** 벌한다.

    왜 관절각이 아니라 중력벡터인가. v34u 는 "정지 목표 관절각" 을 몸통이 수직이
    되도록 FK 로 풀어서 넣었는데, 실측 몸통 피치가 오히려 +8.19 -> +20.93 도로
    나빠졌다. 관절각은 몸통 기울기의 **간접** 손잡이다 — 발바닥이 지면에 눕도록
    물리가 몸통을 돌려버리면 목표 관절각을 맞춰도 몸통은 안 선다. 게다가
    `cost_stand_still` 계수(-0.2)는 imitation 의 1/30 이라 자세를 강제할 힘도 없다.

    `projected_gravity_b` 는 직립일 때 정확히 (0,0,-1) 이므로 x,y 성분이 그대로
    기울기다 (g_x = sin(pitch), g_y = -sin(roll)cos(pitch)). 이걸 직접 벌하면
    어느 관절로 세우든 정책이 알아서 고른다.

    제곱을 쓰는 이유: 수직 근처에서 기울기가 0 으로 죽어 마지막 1 도를 두고
    다른 항과 싸우지 않는다. 8.19 도에서 0.0203, 20.93 도에서 0.1276 이다.
    """
    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)
    tilt = projected_gravity_b[:, 0] ** 2 + projected_gravity_b[:, 1] ** 2
    return tilt * (cmd_norm < 0.01).float()


def reward_imitation(
    base_lin_vel_w: torch.Tensor,  # [N,3] — see docs/decisions.md frame note below
    base_ang_vel_w: torch.Tensor,  # [N,3]
    joints_qpos: torch.Tensor,  # [N,14] actuator order (joint_order.ACTUATOR_JOINT_NAMES)
    joints_qvel: torch.Tensor,  # [N,14]
    contacts: torch.Tensor,  # [N,2] bool/float, left then right
    reference_frame: torch.Tensor,  # [N,36], see poly_reference_motion.py docstring
    commands: torch.Tensor,  # [N,7]
    w_joint_pos: float = 15.0,
    bounded_joint_pos: bool = False,
    swing_only_contact: bool = False,
    k_lin_vel_xy: float = 8.0,
    w_lin_vel_z: float = 1.0,
    w_ang_vel_xy: float = 0.5,
    w_contact: float = 1.0,
    w_stance_violation: float = 0.0,
    w_joint_pos_amp: float = 1.0,
) -> torch.Tensor:
    """Direct port of custom_rewards.py::reward_imitation.

    Frame note: the reference frame's linear/angular velocity slices were
    recorded in the WORLD frame by the reference-motion generator
    (gait_generator.py's `world_linear_vel`/`world_angular_vel`). We compare
    against Isaac Lab's `root_lin_vel_w`/`root_ang_vel_w` (also world frame)
    for that reason. Playground's own MJX version instead reads MuJoCo's
    raw floating-base qvel, whose angular component is expressed in the
    body's local frame per MuJoCo's freejoint convention — a likely
    frame mismatch already present in the original code, not something
    reproduced here. If reward curves look off during the Stage 3 Ubuntu
    smoke test, this is the first place to check.
    """
    # 2026-07-28 (imitation_v12): w_lin_vel_z / w_ang_vel_xy / w_contact and
    # the lin_vel_xy exp sharpness became arguments. scripts/imit_internals2.py
    # measured imitation_v11's final policy standing still (base speed
    # 0.064 m/s against the reference's 0.265) and still collecting:
    #   lin_vel_z  0.954/1.0   ang_vel_xy 0.220/0.5   lin_vel_xy 0.556/1.0
    # i.e. ~92% of the raw imitation total was reachable without walking.
    # lin_vel_xy is the term that is *supposed* to price walking, and at the
    # default sharpness k=8 being wrong by the entire reference speed
    # (err^2 = 0.265^2 = 0.070) still pays exp(-0.56) = 0.57. The term simply
    # could not tell standing from walking. Defaults below are unchanged so
    # v1-v11 stay reproducible; JoystickEnvCfg_Walk2 supplies the new values.
    w_lin_vel_xy = 1.0
    w_ang_vel_z = 0.5
    w_joint_vel = 1.0e-3

    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)

    ref_joint_pos = reference_frame[:, 0:14][:, REF_LEG_JOINT_IDX]  # [N,10]
    ref_joint_vel = reference_frame[:, 14:28][:, REF_LEG_JOINT_IDX]  # [N,10]
    ref_foot_contacts = reference_frame[:, 28:30]  # [N,2]
    ref_lin_vel = reference_frame[:, 30:33]  # [N,3]
    ref_ang_vel = reference_frame[:, 33:36]  # [N,3]

    joint_pos = joints_qpos[:, ACT_LEG_JOINT_IDX]  # [N,10]
    joint_vel = joints_qvel[:, ACT_LEG_JOINT_IDX]  # [N,10]

    lin_vel_xy_rew = (
        torch.exp(-k_lin_vel_xy * torch.sum((base_lin_vel_w[:, :2] - ref_lin_vel[:, :2]) ** 2, dim=-1)) * w_lin_vel_xy
    )
    lin_vel_z_rew = torch.exp(-8.0 * (base_lin_vel_w[:, 2] - ref_lin_vel[:, 2]) ** 2) * w_lin_vel_z
    ang_vel_xy_rew = torch.exp(-2.0 * torch.sum((base_ang_vel_w[:, :2] - ref_ang_vel[:, :2]) ** 2, dim=-1)) * w_ang_vel_xy
    ang_vel_z_rew = torch.exp(-2.0 * (base_ang_vel_w[:, 2] - ref_ang_vel[:, 2]) ** 2) * w_ang_vel_z

    # 2026-07-28: joint_pos was the ONE unbounded term here — every other
    # tracking term above is exp(-err), bounded to [0, 1], while this was a
    # raw negative quadratic that grows without limit. reward_breakdown_v2.py
    # measured the consequence on imitation_v8's final policy: imitation
    # contributed -1.01/step on average (vs alive's +0.40, the largest
    # positive term), driving the summed reward negative on 81.6% of steps —
    # and `_get_rewards` clamps to [0, ...], so those steps delivered a
    # constant 0 and no gradient. The policy was effectively learning from
    # under a fifth of its experience. `bounded_joint_pos` switches this to
    # the same exp form as the velocity terms so it can't dominate; the old
    # quadratic stays reachable for direct comparison.
    if bounded_joint_pos:
        # w_joint_pos is the exp SHARPNESS; w_joint_pos_amp is how much the
        # bounded term is worth relative to the other six imitation terms.
        # v15 proved these are not interchangeable: dropping sharpness 4.0->1.5
        # raised joint_pos_rew 0.148->0.425 at matched checkpoints while the
        # actual joint error stayed put (14.0deg -> 14.3deg). Pose tracking was
        # structurally underpriced -- halving the error is worth only ~+0.02/step
        # against the fall risk of moving precisely -- and no sharpness setting
        # can fix that, because the term is bounded to [0,1] either way.
        joint_pos_rew = torch.exp(-w_joint_pos * torch.sum((joint_pos - ref_joint_pos) ** 2, dim=-1)) * w_joint_pos_amp
    else:
        joint_pos_rew = -torch.sum((joint_pos - ref_joint_pos) ** 2, dim=-1) * w_joint_pos
    joint_vel_rew = -torch.sum((joint_vel - ref_joint_vel) ** 2, dim=-1) * w_joint_vel

    ref_contacts_bool = (ref_foot_contacts > 0.5).float()
    if swing_only_contact and w_stance_violation > 0.0:
        # 2026-07-28 (v13): the plain swing_only form below pays for lifting a
        # foot the reference wants lifted, but costs NOTHING for lifting one the
        # reference wants planted. Flickering both feet therefore raises the
        # chance of overlapping a swing phase and gets paid for it -- and that is
        # exactly what imitation_v12 produced: the user watched it and reported
        # the feet "진동하는것마냥" chattering against the ground, with the
        # measured contact toggle rate at 144-319/10s against v6's 29.4 best.
        # Adding the stance-violation penalty keeps standing at 0 (feet planted
        # during stance costs nothing, and the swing term pays nothing) while
        # making chatter strictly negative.
        swing = torch.sum((1.0 - ref_contacts_bool) * (1.0 - contacts), dim=-1)
        stance_violation = torch.sum(ref_contacts_bool * (1.0 - contacts), dim=-1)
        contact_rew = (swing - w_stance_violation * stance_violation) * w_contact
    elif swing_only_contact:
        # Credit only for lifting a foot the reference says should be lifted.
        # The plain agreement form below counts a planted foot as "matching"
        # whenever the reference also has it planted -- and scripts/ref_stats.py
        # measured the reference's stance duty at 0.692/0.652, so a robot that
        # simply keeps both feet on the ground collects ~1.34 of the available
        # 2.0 for doing nothing at all. That is exactly the behavior observed on
        # imitation_v10 ("발이 붙여진 상태" -- trembling in place, feet planted,
        # never stepping). Scoring swing agreement instead makes standing worth
        # 0 here and forces actual foot alternation to earn the term.
        contact_rew = torch.sum((1.0 - ref_contacts_bool) * (1.0 - contacts), dim=-1) * w_contact
    else:
        contact_rew = torch.sum((contacts == ref_contacts_bool).float(), dim=-1) * w_contact

    reward = lin_vel_xy_rew + lin_vel_z_rew + ang_vel_xy_rew + ang_vel_z_rew + joint_pos_rew + joint_vel_rew + contact_rew
    reward = reward * (cmd_norm > 0.01).float()
    return torch.nan_to_num(reward)


def reward_path_tracking(
    path_err: torch.Tensor,  # [N,3] — lateral, cos(yaw_err), sin(yaw_err)
    k_lateral: float,
    k_yaw: float,
    w_yaw: float,
) -> torch.Tensor:
    """경로에서 벗어난 정도를 벌한다 (Disney BD-X의 path frame).

    속도 명령은 순수 rate라 yaw_rate=0이 "원래 방향으로 돌아와라"를 뜻하지
    않는다. 그래서 한 번 휘면 정책이 그 사실 자체를 관측하지 못하고 되돌릴
    이유도 없다. 적분된 경로 기준의 횡방향·방향 오차를 관측에 넣고 여기서
    보상해야 비로소 "일자로 걷기"가 학습 목표가 된다.
    """
    lateral = path_err[:, 0]
    yaw_err = torch.atan2(path_err[:, 2], path_err[:, 1])
    return torch.exp(-k_lateral * lateral**2) + torch.exp(-k_yaw * yaw_err**2) * w_yaw
