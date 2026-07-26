"""DirectRLEnv port of Open_Duck_Playground's
playground/open_duck_mini_v2/joystick.py::Joystick.

Structured like Isaac Lab's own anymal_c_env.py (a first-party DirectRLEnv
locomotion example) — see docs/decisions.md for why DirectRLEnv was chosen
over ManagerBasedRLEnv for this task.

Known simplification vs. Playground (see joystick_env_cfg.py's docstring):
no asymmetric-critic privileged observation in this v1 port.

Known simplification vs. Playground #2: Playground computes a delayed/noisy
"gravity" reading (`imu_history`) but never actually includes it in the
final 101-dim `state` observation it returns (the hstack that builds
`state` omits it) — it's dead code in the source. This port does not
reproduce that dead computation.

Staged training note: `cfg.use_imitation` gates the reference-motion
subsystem (see joystick_env_cfg.py). When False (Stage 1, current default),
`PolyReferenceMotion` is never instantiated — `cfg.reference_motion_pkl`
need not exist — and the "imitation" reward term is omitted from `_get_rewards`
entirely rather than zero-weighted. The 101-dim observation layout is
unchanged either way: the imitation_phase (cos, sin) channel still runs off
a periodic gait-phase counter, just driven by `cfg.gait_period_steps` instead
of the reference motion's own period.

Step ordering note: DirectRLEnv's base `step()` calls `_get_dones()`, then
`_get_rewards()`, then (for terminated envs) `_reset_idx()`, then interval
events, then `_get_observations()` last. This env's `_get_rewards` relies on
that ordering — it reads `self._last_act` (still holding the *previous*
step's action) before `_get_observations` shifts the last-act history chain
for the *next* step.
"""

from __future__ import annotations

import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.terrains import TerrainImporter

from open_duck_mini_isaaclab.joint_order import (
    ACTUATOR_JOINT_NAMES,
    HOME_BASE_HEIGHT,
    LEFT_FOOT_BODY_NAME,
    RIGHT_FOOT_BODY_NAME,
    ROOT_BODY_NAME,
)
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import REF_FRAME_DIM, PolyReferenceMotion

from .joystick_env_cfg import JoystickEnvCfg
from .observations import DelayBuffer, apply_uniform_noise
from .rewards import (
    cost_action_rate,
    cost_stand_still,
    cost_torques,
    reward_alive,
    reward_imitation,
    reward_tracking_ang_vel,
    reward_tracking_lin_vel,
)

# velocity/ -> tasks/ -> open_duck_mini_isaaclab/ -> source/ -> repo root:
# 5 dirname() calls, not 4 — a 4-call version silently resolved to .../source
# instead of the repo root, which only surfaced once cfg.reference_motion_pkl
# (a "source/..."-prefixed relative path) got joined onto it, producing a
# doubled "source/source/..." path (caught 2026-07-26 on Stage 2's first
# real run: FileNotFoundError joining the two).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

FOOT_CONTACT_FORCE_THRESHOLD = 1.0  # N


class JoystickEnv(DirectRLEnv):
    cfg: JoystickEnvCfg

    def __init__(self, cfg: JoystickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Canonical joint order (joint_order.ACTUATOR_JOINT_NAMES) may not
        # match the articulation's internal (USD-derived) joint order —
        # resolve once and reuse everywhere, per-Isaac Lab convention
        # (Articulation.find_joints(..., preserve_order=True)).
        self._joint_ids, _joint_names = self._robot.find_joints(ACTUATOR_JOINT_NAMES, preserve_order=True)
        assert _joint_names == ACTUATOR_JOINT_NAMES, (
            f"find_joints did not return joints in the requested order: {_joint_names} != {ACTUATOR_JOINT_NAMES}"
        )
        self._feet_ids, _feet_names = self._contact_sensor.find_bodies(
            [LEFT_FOOT_BODY_NAME, RIGHT_FOOT_BODY_NAME], preserve_order=True
        )
        # Disney BD-X paper (Grandia et al. 2024), V-B: terminate on
        # torso/head ground contact rather than (only) a height-ratio/flip
        # heuristic — a contact condition can't be evaded by folding into a
        # collapsed-but-technically-tall-enough, non-inverted heap the way
        # min_base_height_ratio could.
        # No rigid body is literally named "head" in the USD-converted
        # articulation (confirmed 2026-07-27: URDF has a "head" link, but
        # it has no inertia of its own and gets merged into its fixed
        # parent during import) — "head_pitch_assembly" is the actual
        # physics body closest to the head.
        self._trunk_head_ids, _trunk_head_names = self._contact_sensor.find_bodies(
            [ROOT_BODY_NAME, "head_pitch_assembly"], preserve_order=True
        )

        n = self.num_envs
        nj = len(ACTUATOR_JOINT_NAMES)
        dev = self.device

        self._actions = torch.zeros(n, nj, device=dev)
        # NOTE: Playground keeps a single `last_act` (previous step's raw
        # action) used both for the action-history observation stack AND
        # for the action_rate reward's "current vs previous" comparison —
        # we do the same with `_last_act` here rather than tracking a
        # redundant second copy, relying on DirectRLEnv's step() ordering
        # (_get_rewards runs before _get_observations, see joystick_env.py
        # module docstring) so `_last_act` still holds the pre-shift
        # (previous-step) value when `_get_rewards` reads it.
        self._last_act = torch.zeros(n, nj, device=dev)
        self._last_last_act = torch.zeros(n, nj, device=dev)
        self._last_last_last_act = torch.zeros(n, nj, device=dev)
        self._motor_targets = torch.zeros(n, nj, device=dev)

        self._action_delay_buf = DelayBuffer(n, nj, max(cfg.action_max_delay, 1), dev)

        self._command = torch.zeros(n, 7, device=dev)
        self._imitation_i = torch.zeros(n, dtype=torch.long, device=dev)

        # Per-joint position-noise scale (hip/knee/ankle/head), built once
        # from cfg — see joystick_env_cfg.py's noise_scale_* fields.
        self._joint_pos_noise_scale = torch.zeros(nj, device=dev)
        for idx, name in enumerate(ACTUATOR_JOINT_NAMES):
            if "hip" in name:
                self._joint_pos_noise_scale[idx] = cfg.noise_scale_hip_pos
            elif "knee" in name:
                self._joint_pos_noise_scale[idx] = cfg.noise_scale_knee_pos
            elif "ankle" in name:
                self._joint_pos_noise_scale[idx] = cfg.noise_scale_ankle_pos
            else:
                self._joint_pos_noise_scale[idx] = cfg.noise_scale_head_pos

        # Stage 1 (use_imitation=False): skip loading PolyReferenceMotion
        # entirely — reference_motion_pkl need not exist yet. The gait-phase
        # observation channel still needs *a* period to divide by, so it
        # falls back to cfg.gait_period_steps instead of prm.nb_steps_in_period.
        if cfg.use_imitation:
            pkl_path = cfg.reference_motion_pkl
            if not os.path.isabs(pkl_path):
                pkl_path = os.path.join(_REPO_ROOT, pkl_path)
            self._prm = PolyReferenceMotion(pkl_path, device=dev)
            self._gait_period_steps = self._prm.nb_steps_in_period
        else:
            self._prm = None
            self._gait_period_steps = cfg.gait_period_steps
        self._current_reference_motion = torch.zeros(n, REF_FRAME_DIM, device=dev)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── command sampling ────────────────────────────────────────────────

    def _sample_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = len(env_ids)
        dev = self.device
        cfg = self.cfg
        cmd = torch.zeros(n, 7, device=dev)
        cmd[:, 0] = math_utils.sample_uniform(*cfg.lin_vel_x_range, (n,), dev)
        cmd[:, 1] = math_utils.sample_uniform(*cfg.lin_vel_y_range, (n,), dev)
        cmd[:, 2] = math_utils.sample_uniform(*cfg.ang_vel_yaw_range, (n,), dev)
        cmd[:, 3] = math_utils.sample_uniform(*cfg.neck_pitch_range, (n,), dev)
        cmd[:, 4] = math_utils.sample_uniform(*cfg.head_pitch_range, (n,), dev)
        cmd[:, 5] = math_utils.sample_uniform(*cfg.head_yaw_range, (n,), dev)
        cmd[:, 6] = math_utils.sample_uniform(*cfg.head_roll_range, (n,), dev)
        zero_mask = torch.rand(n, device=dev) < cfg.zero_command_prob
        cmd[zero_mask] = 0.0
        return cmd

    # ── physics step ─────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._action_delay_buf.push(self._actions)
        action_w_delay = self._action_delay_buf.sample(self.cfg.action_min_delay, self.cfg.action_max_delay)

        default_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        target = default_pos + action_w_delay * self.cfg.action_scale

        prev = self._motor_targets
        max_delta = self.cfg.max_motor_velocity * self.step_dt
        self._motor_targets = torch.clamp(target, prev - max_delta, prev + max_delta)

        # advance gait phase (drives the imitation_phase observation always;
        # drives the reference-motion lookup only when use_imitation=True)
        self._imitation_i = (self._imitation_i + 1) % self._gait_period_steps
        if self.cfg.use_imitation:
            self._current_reference_motion = self._prm.get_reference_motion(
                self._command[:, 0], self._command[:, 1], self._command[:, 2], self._imitation_i
            )

        # command resampling every `command_resample_steps` env steps
        resample_mask = (self.episode_length_buf % self.cfg.command_resample_steps == 0) & (self.episode_length_buf > 0)
        resample_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._command[resample_ids] = self._sample_command(resample_ids)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._motor_targets, joint_ids=self._joint_ids)

    # ── observations ─────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        cfg = self.cfg
        level = cfg.noise_level

        gyro = apply_uniform_noise(self._imu.data.ang_vel_b, level, cfg.noise_scale_gyro)
        accel = apply_uniform_noise(self._imu.data.lin_acc_b, level, cfg.noise_scale_accelerometer)

        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids]

        joint_pos_rel = apply_uniform_noise(joint_pos - default_joint_pos, level, self._joint_pos_noise_scale)
        joint_vel_scaled = apply_uniform_noise(joint_vel * cfg.dof_vel_scale, level, cfg.noise_scale_joint_vel)

        contact = self._get_foot_contact()

        phase = 2.0 * torch.pi * self._imitation_i.float() / self._gait_period_steps
        imitation_phase = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)

        state = torch.cat(
            [
                gyro,
                accel,
                self._command,
                joint_pos_rel,
                joint_vel_scaled,
                self._last_act,
                self._last_last_act,
                self._last_last_last_act,
                self._motor_targets,
                contact,
                imitation_phase,
            ],
            dim=-1,
        )

        # history bookkeeping (matches Playground's step()-end ordering)
        self._last_last_last_act = self._last_last_act
        self._last_last_act = self._last_act
        self._last_act = self._actions

        return {"policy": state}

    def _get_foot_contact(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids, :]
        return (torch.norm(forces, dim=-1) > FOOT_CONTACT_FORCE_THRESHOLD).float()

    def _get_trunk_head_contact(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._trunk_head_ids, :]
        return (torch.norm(forces, dim=-1) > FOOT_CONTACT_FORCE_THRESHOLD).any(dim=-1)

    # ── rewards ──────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        dt = self.step_dt

        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        contact = self._get_foot_contact()

        terms = {
            "tracking_lin_vel": reward_tracking_lin_vel(self._command, self._robot.data.root_lin_vel_b, cfg.tracking_sigma)
            * cfg.tracking_lin_vel_scale,
            "tracking_ang_vel": reward_tracking_ang_vel(self._command, self._imu.data.ang_vel_b, cfg.tracking_sigma)
            * cfg.tracking_ang_vel_scale,
            "torques": cost_torques(self._robot.data.applied_torque[:, self._joint_ids]) * cfg.torques_scale,
            "action_rate": cost_action_rate(self._actions, self._last_act) * cfg.action_rate_scale,
            "alive": reward_alive(self.num_envs, self.device) * cfg.alive_scale,
            "stand_still": cost_stand_still(self._command, joint_pos, joint_vel, default_joint_pos) * cfg.stand_still_scale,
        }
        # Stage 1 (use_imitation=False): omitted entirely, not just
        # zero-weighted — reward_imitation would divide-by-nothing-useful
        # against an all-zero self._current_reference_motion otherwise.
        if cfg.use_imitation:
            terms["imitation"] = (
                reward_imitation(
                    self._robot.data.root_lin_vel_w,
                    self._robot.data.root_ang_vel_w,
                    joint_pos,
                    joint_vel,
                    contact,
                    self._current_reference_motion,
                    self._command,
                    w_joint_pos=cfg.imitation_w_joint_pos,
                )
                * cfg.imitation_scale
            )

        reward = torch.sum(torch.stack(list(terms.values())), dim=0) * dt
        return torch.clamp(reward, 0.0, 10000.0)

    # ── termination ──────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Isaac Lab's projected_gravity_b has the OPPOSITE sign convention
        # from Playground's "upvector" sensor (see rewards.py's docstring
        # note): z > 0 here means flipped over (Playground: upvector z < 0).
        gravity = self._robot.data.projected_gravity_b
        flipped = gravity[:, 2] > 0.0
        # Added 2026-07-26 alongside `flipped`: see joystick_env_cfg.py's
        # min_base_height_ratio docstring — `flipped` alone only catches
        # >90 deg tips, not a collapsed-but-not-inverted heap, which Stage 1
        # (use_imitation=False) has no other guard against.
        collapsed = self._robot.data.root_pos_w[:, 2] < HOME_BASE_HEIGHT * self.cfg.min_base_height_ratio
        has_nan = torch.isnan(self._robot.data.joint_pos).any(dim=-1) | torch.isnan(self._robot.data.joint_vel).any(dim=-1)
        # See _get_trunk_head_contact's docstring note (added alongside
        # `collapsed`/`flipped`, not replacing them) — this catches
        # collapsed-but-not-inverted-or-short-enough poses that both of
        # those miss.
        trunk_head_down = self._get_trunk_head_contact()
        terminated = flipped | collapsed | has_nan | trunk_head_down
        return terminated, time_out

    # ── reset ────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        dev = self.device

        self._actions[env_ids] = 0.0
        self._last_act[env_ids] = 0.0
        self._last_last_act[env_ids] = 0.0
        self._last_last_last_act[env_ids] = 0.0
        self._action_delay_buf.reset_idx(env_ids)
        self._imitation_i[env_ids] = 0
        self._command[env_ids] = self._sample_command(env_ids)

        # multiplicative joint-pos randomization (matches Playground's
        # `qpos_j = home_qpos * U(0.5, 1.5)` — applied to ALL joints in
        # native articulation order, order-independent since it's elementwise).
        factor = math_utils.sample_uniform(0.5, 1.5, (n, self._robot.num_joints), dev)
        joint_pos = self._robot.data.default_joint_pos[env_ids] * factor
        joint_vel = torch.zeros(n, self._robot.num_joints, device=dev)

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]  # local -> world
        default_root_state[:, :2] += math_utils.sample_uniform(-0.05, 0.05, (n, 2), dev)  # xy spawn jitter
        yaw = math_utils.sample_uniform(-3.14, 3.14, (n,), dev)
        yaw_quat = math_utils.quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        default_root_state[:, 3:7] = math_utils.quat_mul(default_root_state[:, 3:7], yaw_quat)
        default_root_state[:, 7:13] = math_utils.sample_uniform(-0.05, 0.05, (n, 6), dev)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._motor_targets[env_ids] = joint_pos[:, self._joint_ids]
        if self.cfg.use_imitation:
            self._current_reference_motion[env_ids] = self._prm.get_reference_motion(
                self._command[env_ids, 0], self._command[env_ids, 1], self._command[env_ids, 2], self._imitation_i[env_ids]
            )
