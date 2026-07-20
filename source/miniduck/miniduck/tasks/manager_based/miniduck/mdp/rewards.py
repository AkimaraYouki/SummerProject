from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------
# reset
# ---------------------------
def reset_default_state(env: "ManagerBasedRLEnv", env_ids, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None or len(env_ids) == 0:
        env_ids = asset._ALL_INDICES

    root = asset.data.default_root_state[env_ids].clone()
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]

    asset.write_root_pose_to_sim(root[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:], env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    if hasattr(env, "_prev_action"):
        env._prev_action[env_ids] = 0.0


# ---------------------------
# observations
# ---------------------------
def joint_pos_rel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return wrap_to_pi(asset.data.joint_pos)


def joint_vel_rel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def root_up_vector_obs(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    up = torch.zeros((env.num_envs, 3), device=env.device)
    up[:, 2] = 1.0
    return quat_apply(asset.data.root_quat_w, up)


def root_ang_vel_obs(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


def root_lin_vel_obs(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def base_pos_z(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def last_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = getattr(env.action_manager, "action", None)
    if a is None:
        return torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    return a


def projected_gravity(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


# ---------------------------
# rewards
# ---------------------------
def is_alive(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return (~env.reset_terminated).float()


def is_terminated(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.reset_terminated.float()


def forward_velocity(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """X축 전진 속도 보상"""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_x = asset.data.root_lin_vel_w[:, 0]
    return torch.clamp(vel_x, min=0.0, max=2.0)


def target_forward_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_vel: float = 0.8,
    sigma: float = 0.25,
) -> torch.Tensor:
    """목표 속도에 가까울수록 높은 보상 (gaussian). 빠를수록 좋다는 incentive 제거."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_x = asset.data.root_lin_vel_w[:, 0]
    error = vel_x - target_vel
    return torch.exp(-error * error / (2.0 * sigma * sigma))


def base_height_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.33,
) -> torch.Tensor:
    """베이스 높이가 목표에서 벗어나는 패널티 — 점프/웅크림 방지."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    dz = z - target_height
    return dz * dz


def root_upright_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    local_up = torch.zeros((env.num_envs, 3), device=env.device)
    local_up[:, 2] = 1.0
    world_up = quat_apply(asset.data.root_quat_w, local_up)
    target = torch.zeros_like(world_up)
    target[:, 2] = 1.0
    diff = world_up - target
    return torch.sum(diff * diff, dim=1)


def root_ang_vel_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_w
    return torch.sum(w * w, dim=1)


def root_axis_vel_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, axis: int = 1) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_w[:, axis]
    return v * v


def root_axis_ang_vel_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, axis: int = 1) -> torch.Tensor:
    """특정 축의 각속도 패널티 — axis=1이 pitch"""
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_w[:, axis]
    return w * w


def action_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = getattr(env.action_manager, "action", None)
    if a is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(a * a, dim=1)


def stride_symmetry(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """좌우 보폭 크기가 같도록 강제 — (|left| - |right|)^2 패널티."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    pos = asset.data.joint_pos - asset.data.default_joint_pos
    pairs = [
        ("dof_left_knee_0",  "dof_right_knee_0"),
        ("dof_left_ankle_0", "dof_right_ankle_0"),
    ]
    loss = torch.zeros(env.num_envs, device=env.device)
    for l_name, r_name in pairs:
        try:
            l_i = joint_names.index(l_name)
            r_i = joint_names.index(r_name)
            diff = torch.abs(pos[:, l_i]) - torch.abs(pos[:, r_i])
            loss += diff * diff
        except ValueError:
            pass
    return loss


def root_lin_vel_x_rate_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """X축 선속도 변화율 패널티 — 몸체 전후 흔들림 억제."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_x = asset.data.root_lin_vel_w[:, 0]
    if not hasattr(env, "_prev_vel_x"):
        env._prev_vel_x = vel_x.clone()
    dv = vel_x - env._prev_vel_x
    env._prev_vel_x = vel_x.detach().clone()
    return dv * dv


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = getattr(env.action_manager, "action", None)
    if a is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_prev_action"):
        env._prev_action = torch.zeros_like(a)
    da = a - env._prev_action
    env._prev_action = a.detach().clone()
    return torch.sum(da * da, dim=1)


def joint_vel_l1(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(vel), dim=1)


def joint_pos_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """관절이 기본 자세에서 벗어나는 패널티 — 보폭 억제."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos - asset.data.default_joint_pos
    return torch.sum(diff * diff, dim=1)


def leg_antiphase(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """좌우 다리가 반대 부호로 움직일 때 보상 (L * R < 0 → 양수 보상).
    두 다리 모두 정지해도 L*R=0이므로 정지 해법을 허용하지 않음."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names

    pairs = [
        ("dof_left_knee_0",  "dof_right_knee_0"),
        ("dof_left_ankle_0", "dof_right_ankle_0"),
    ]
    reward = torch.zeros(env.num_envs, device=env.device)
    pos = asset.data.joint_pos - asset.data.default_joint_pos  # 기본자세 기준 편차
    for l_name, r_name in pairs:
        try:
            l_i = joint_names.index(l_name)
            r_i = joint_names.index(r_name)
            reward += -pos[:, l_i] * pos[:, r_i]  # 반대 부호면 양수
        except ValueError:
            pass
    return reward


def feet_regulation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    desired_body_height: float = 0.14,
) -> torch.Tensor:
    """발이 지면 근처일 때 수평 속도 패널티 — 발 슬라이딩/끌기 방지.
    발 높이가 낮을수록 수평 속도에 지수적으로 강한 패널티."""
    asset: Articulation = env.scene[asset_cfg.name]

    # 발 body ID 캐싱
    if not hasattr(env, "_foot_body_ids"):
        body_names = list(asset.data.body_names)
        ids = []
        for name in ["foot_bottom_pla", "foot_bottom_pla_1"]:
            try:
                ids.append(body_names.index(name))
            except ValueError:
                pass
        env._foot_body_ids = ids

    if not env._foot_body_ids:
        return torch.zeros(env.num_envs, device=env.device)

    # 발 높이 (world z)
    feet_pos_z = asset.data.body_pos_w[:, env._foot_body_ids, 2]        # (N, 2)
    # 발 수평 속도
    feet_vel_xy = asset.data.body_lin_vel_w[:, env._foot_body_ids, :2]  # (N, 2, 2)
    feet_vel_norm = torch.norm(feet_vel_xy, dim=-1)                      # (N, 2)

    # 높이가 낮을수록 패널티 강해짐
    exp_term = torch.exp(-feet_pos_z / (0.025 * desired_body_height))
    return torch.sum(feet_vel_norm ** 2 * exp_term, dim=-1)


def unbalance_feet_air_time(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "foot_contact",
) -> torch.Tensor:
    """좌우 발 공중 체류 시간 분산 패널티 — 한쪽 발만 오래 뜨는 비대칭 보행 억제."""
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    # last_air_time: (num_envs, num_bodies) — track_air_time=True 필요
    air_times = sensor.data.last_air_time
    return torch.var(air_times, dim=-1)


def foot_alternating_contact(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "foot_contact",
    threshold: float = 5.0,
) -> torch.Tensor:
    """교대 접지 보상: 한 발만 지면에 닿을 때(보행 중) 보상.
    두 발 다 닿거나 두 발 다 뜨면 0점."""
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    force_norms = torch.norm(forces, dim=-1)  # (num_envs, 2)
    left  = (force_norms[:, 0] > threshold).float()
    right = (force_norms[:, 1] > threshold).float()
    # XOR: 정확히 한 발만 닿을 때 1.0
    return torch.abs(left - right)


def no_fly(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """정확히 한 발만 접지할 때 보상 — 실제 걷기 패턴 직접 유도."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]  # (N, num_feet)
    contacts = forces_z > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1
    return single_contact.float()


def no_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """두 발 다 지면에서 떨어지면 패널티 — 점프/날기 억제."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]
    contacts = forces_z > threshold
    return (torch.sum(contacts.float(), dim=1) == 0).float()


def joint_powers_l1(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """관절 파워 패널티 (토크 × 속도) — 에너지 효율적인 보행 유도."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


def foot_clearance_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.04,
    std: float = 0.05,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """발이 목표 높이만큼 들릴 때 보상 — 발을 못 떼는 문제 해결.
    발이 target_height에 가까우면서 수평 속도가 있을 때 보상."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


# ---------------------------
# terminations
# ---------------------------
def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length - 1


def root_tilt_exceeded(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, max_tilt_deg: float = 30.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    local_up = torch.zeros((env.num_envs, 3), device=env.device)
    local_up[:, 2] = 1.0
    world_up = quat_apply(asset.data.root_quat_w, local_up)
    theta = torch.acos(world_up[:, 2].clamp(-1.0, 1.0))
    return theta > math.radians(max_tilt_deg)
