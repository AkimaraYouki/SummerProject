"""Measures every reward term while the robot simply holds the READY pose
(action=0), before any training.

This is the acceptance test for the READY-pose fix: at action=0 the robot sits
exactly on the reference gait's mean pose, so the joint_pos imitation term
should now read near its maximum instead of the +0.012/1.0 it was stuck at
when the robot initialized straight-legged. It also shows the reward landscape
the policy starts from, and whether the clamp is still eating steps.
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-OpenDuckMini-Joystick-A20J5Bounded-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity.rewards import (  # noqa: E402
    reward_tracking_lin_vel, reward_tracking_ang_vel, cost_torques,
    cost_action_rate, reward_alive, cost_stand_still, reward_imitation)

_MAP = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9Big-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9BigLE-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9MB16-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Path-v0": "JoystickEnvCfg_Path",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
cfg = getattr(_cm, _MAP[args_cli.task])()
cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=cfg)
u = env.unwrapped
u.reset()
print(f"[info] w_joint_pos={cfg.imitation_w_joint_pos} bounded={cfg.imitation_bounded_joint_pos} "
      f"imitation_scale={cfg.imitation_scale} alive_scale={cfg.alive_scale}", flush=True)

zero = torch.zeros(args_cli.num_envs, len(u._joint_ids), device=u.device)
acc, n_clamped, n = {}, 0, 0
jp_err_acc = jpr_acc = 0.0
for i in range(args_cli.num_steps):
    u.step(zero)
    j_pos = u._robot.data.joint_pos[:, u._joint_ids]
    j_vel = u._robot.data.joint_vel[:, u._joint_ids]
    d_pos = u._robot.data.default_joint_pos[:, u._joint_ids]
    contact = u._get_foot_contact()
    t = {
        "tracking_lin_vel": reward_tracking_lin_vel(u._command, u._robot.data.root_lin_vel_b, cfg.tracking_sigma) * cfg.tracking_lin_vel_scale,
        "tracking_ang_vel": reward_tracking_ang_vel(u._command, u._imu.data.ang_vel_b, cfg.tracking_sigma) * cfg.tracking_ang_vel_scale,
        "torques": cost_torques(u._robot.data.applied_torque[:, u._joint_ids]) * cfg.torques_scale,
        "action_rate": cost_action_rate(u._actions, u._last_act) * cfg.action_rate_scale,
        "alive": reward_alive(u.num_envs, u.device) * cfg.alive_scale,
        "stand_still": cost_stand_still(u._command, j_pos, j_vel, d_pos) * cfg.stand_still_scale,
    }
    if cfg.use_imitation:
        t["imitation"] = reward_imitation(
            u._robot.data.root_lin_vel_w, u._robot.data.root_ang_vel_w, j_pos, j_vel, contact,
            u._current_reference_motion, u._command,
            w_joint_pos=cfg.imitation_w_joint_pos,
            bounded_joint_pos=cfg.imitation_bounded_joint_pos,
            # 2026-07-29: these were missing, so every knob added after
            # bounded_joint_pos silently fell back to reward_imitation's
            # upstream defaults and this script reported numbers for a config
            # nobody was training. Caught when w_joint_pos_amp=3.0 produced no
            # change here despite being correctly wired into the env.
            swing_only_contact=cfg.imitation_swing_only_contact,
            k_lin_vel_xy=cfg.imitation_k_lin_vel_xy,
            w_lin_vel_z=cfg.imitation_w_lin_vel_z,
            w_ang_vel_xy=cfg.imitation_w_ang_vel_xy,
            w_contact=cfg.imitation_w_contact,
            w_stance_violation=cfg.imitation_w_stance_violation,
            w_joint_pos_amp=cfg.imitation_w_joint_pos_amp) * cfg.imitation_scale
        e2 = torch.sum((j_pos[:, ACT_LEG_JOINT_IDX] - u._current_reference_motion[:, 0:14][:, REF_LEG_JOINT_IDX]) ** 2, dim=-1)
        jp_err_acc += e2.mean().item()
        jpr_acc += (torch.exp(-cfg.imitation_w_joint_pos * e2) * cfg.imitation_w_joint_pos_amp).mean().item()
    s = torch.sum(torch.stack(list(t.values())), 0) * u.step_dt
    n_clamped += (s < 0).sum().item()
    for k, v in t.items():
        acc[k] = acc.get(k, 0.0) + (v * u.step_dt).mean().item()
    acc["__TOTAL__"] = acc.get("__TOTAL__", 0.0) + s.mean().item()
    n += 1

print("\n" + "=" * 72)
print(f"REWARD AT READY POSE (action=0, {n} steps x {args_cli.num_envs} envs)")
print("=" * 72)
for k, v in acc.items():
    if not k.startswith("__"):
        print(f"  {k:<20}{v/n:+9.5f}/step")
print(f"  {'SUM (pre-clamp)':<20}{acc['__TOTAL__']/n:+9.5f}/step")
print("-" * 72)
print(f"  clamped steps: {n_clamped}/{n*args_cli.num_envs} ({100*n_clamped/(n*args_cli.num_envs):.1f}%)")
print(f"  joint_pos sum_err^2 : {jp_err_acc/n:.4f} rad^2  -> per-joint {((jp_err_acc/n)/10)**0.5*57.3:.1f} deg")
print(f"  joint_pos_rew (raw) : {jpr_acc/n:.4f} / 1.0")
print("=" * 72, flush=True)
env.close(); simulation_app.close()
