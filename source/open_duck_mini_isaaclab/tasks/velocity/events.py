"""Custom domain-randomization event not covered by IsaacLab's builtins.

Port of Open_Duck_Playground's playground/common/randomize.py's qpos0
jitter (`qpos0 = qpos0.at[joint_addr].set(qpos0[joint_addr] + U(-0.03,0.03))`).

This is NOT the same as IsaacLab's builtin `reset_joints_by_offset` (which
perturbs the joint position written into the simulation AT RESET time only).
Playground's qpos0 jitter instead perturbs the *persistent reference pose*
(MuJoCo's `qpos0`) that motor-target deltas and joint-position observations
are computed relative to for the rest of the episode — i.e. it mutates
`ArticulationData.default_joint_pos` itself, which this env reads every
single step (see joystick_env.py's `_pre_physics_step`/`_get_observations`).
No IsaacLab builtin does this, hence the custom event term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_default_joint_pos(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    position_range: tuple[float, float] = (-0.03, 0.03),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Add a fresh U(*position_range) offset to `default_joint_pos` for the
    given envs, applied at `mode="reset"` (re-jittered every episode, same
    as Playground's `reset_model()` behavior — it's inside `reset`, not
    `startup`, in the original).
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids != slice(None) else torch.arange(asset.num_joints, device=asset.device)
    num_joints = len(joint_ids) if hasattr(joint_ids, "__len__") else asset.num_joints

    offset = torch.empty(len(env_ids), num_joints, device=asset.device).uniform_(*position_range)

    # start from the articulation's ORIGINAL default (USD/robot_cfg home
    # pose), not the previous episode's already-jittered value, so jitter
    # doesn't compound across resets.
    if not hasattr(asset.data, "_original_default_joint_pos"):
        asset.data._original_default_joint_pos = asset.data.default_joint_pos.clone()

    original = asset.data._original_default_joint_pos
    if asset_cfg.joint_ids != slice(None):
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = original[env_ids[:, None], joint_ids] + offset
    else:
        asset.data.default_joint_pos[env_ids] = original[env_ids] + offset
