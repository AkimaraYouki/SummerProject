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
    """ignore_head=False path only (the one joystick.py actually calls)."""
    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)
    pose_cost = torch.sum(torch.abs(qpos - default_pose), dim=-1)
    vel_cost = torch.sum(torch.abs(qvel), dim=-1)
    return (pose_cost + vel_cost) * (cmd_norm < 0.01).float()


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
        joint_pos_rew = torch.exp(-w_joint_pos * torch.sum((joint_pos - ref_joint_pos) ** 2, dim=-1))
    else:
        joint_pos_rew = -torch.sum((joint_pos - ref_joint_pos) ** 2, dim=-1) * w_joint_pos
    joint_vel_rew = -torch.sum((joint_vel - ref_joint_vel) ** 2, dim=-1) * w_joint_vel

    ref_contacts_bool = (ref_foot_contacts > 0.5).float()
    if swing_only_contact:
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
