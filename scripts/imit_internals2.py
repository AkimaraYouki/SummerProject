"""imit_internals + Walk task + swing_only_contact + direct foot-motion measurement.

Adds what imit_internals.py could not answer: is the policy actually STEPPING?
Reward sub-terms tell you what it is being paid for; foot height/toggle tells
you what it is doing. Both are needed to separate "reward is wrong" from
"reward is right but unlearnable".
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from open_duck_mini_isaaclab.agents.rsl_rl_compat import (  # noqa: E402
    build_runner,
    load_checkpoint,
)
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    JoystickPPORunnerCfg,
    JoystickPPORunnerCfg_Gamma097,
)

# The runner cfg must match the one the checkpoint was TRAINED with, not just
# the env cfg: Walk9 trains with the upstream network (512,256,128) while every
# other variant uses (256,128,64), and loading across them fails with a bare
# size-mismatch on actor.0.weight.
_TASK_TO_RUNNER = {
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": JoystickPPORunnerCfg_Gamma097,
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": JoystickPPORunnerCfg_Gamma097,
    "Isaac-OpenDuckMini-Joystick-Path-v0": JoystickPPORunnerCfg_Gamma097,
}

from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

_MAP = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Path-v0": "JoystickEnvCfg_Path",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = _TASK_TO_RUNNER.get(args_cli.task, JoystickPPORunnerCfg)()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
cfg = u.cfg
print(f"[info] w_joint_pos={cfg.imitation_w_joint_pos} bounded={cfg.imitation_bounded_joint_pos} "
      f"swing_only={cfg.imitation_swing_only_contact} scale={cfg.imitation_scale} "
      f"lock_head={cfg.lock_head_joints}", flush=True)

K = ["joint_pos_err2", "joint_pos_rew", "joint_vel_rew", "joint_vel_err2", "lin_xy", "lin_z",
     "ang_xy", "ang_z", "contact", "contact_ref", "cmd_active", "imit_total",
     "n_feet_down", "ref_n_feet_down", "base_speed", "ref_speed"]
acc = {k: 0.0 for k in K}
n = 0
toggles = None
prev_contact = None
foot_z_hist = []

obs = env.get_observations()
for step in range(args_cli.num_steps):
    with torch.inference_mode():
        actions = policy(obs)
    obs, rew, dones, infos = env.step(actions)

    jp = u._robot.data.joint_pos[:, u._joint_ids][:, ACT_LEG_JOINT_IDX]
    jv = u._robot.data.joint_vel[:, u._joint_ids][:, ACT_LEG_JOINT_IDX]
    rf = u._current_reference_motion
    ref_jp = rf[:, 0:14][:, REF_LEG_JOINT_IDX]
    ref_jv = rf[:, 14:28][:, REF_LEG_JOINT_IDX]
    ref_c = (rf[:, 28:30] > 0.5).float()
    ref_lv = rf[:, 30:33]
    ref_av = rf[:, 33:36]
    contacts = u._get_foot_contact().float()
    blv = u._robot.data.root_lin_vel_w
    bav = u._robot.data.root_ang_vel_w

    err2 = torch.sum((jp - ref_jp) ** 2, dim=-1)
    verr2 = torch.sum((jv - ref_jv) ** 2, dim=-1)
    w = cfg.imitation_w_joint_pos
    AMP = cfg.imitation_w_joint_pos_amp
    jpr = torch.exp(-w * err2) * AMP if cfg.imitation_bounded_joint_pos else -err2 * w
    jvr = -verr2 * 1.0e-3
    lxy = torch.exp(-cfg.imitation_k_lin_vel_xy * torch.sum((blv[:, :2] - ref_lv[:, :2]) ** 2, dim=-1)) * 1.0
    lz = torch.exp(-8.0 * (blv[:, 2] - ref_lv[:, 2]) ** 2) * cfg.imitation_w_lin_vel_z
    axy = torch.exp(-2.0 * torch.sum((bav[:, :2] - ref_av[:, :2]) ** 2, dim=-1)) * cfg.imitation_w_ang_vel_xy
    az = torch.exp(-2.0 * (bav[:, 2] - ref_av[:, 2]) ** 2) * 0.5
    if cfg.imitation_swing_only_contact:
        sw = torch.sum((1.0 - ref_c) * (1.0 - contacts), dim=-1)
        sv = torch.sum(ref_c * (1.0 - contacts), dim=-1)
        cr = (sw - cfg.imitation_w_stance_violation * sv) * cfg.imitation_w_contact
    else:
        cr = torch.sum((contacts == ref_c).float(), dim=-1) * cfg.imitation_w_contact

    acc["joint_pos_err2"] += err2.mean().item()
    acc["joint_pos_rew"] += jpr.mean().item()
    acc["joint_vel_rew"] += jvr.mean().item()
    acc["joint_vel_err2"] += verr2.mean().item()
    acc["lin_xy"] += lxy.mean().item()
    acc["lin_z"] += lz.mean().item()
    acc["ang_xy"] += axy.mean().item()
    acc["ang_z"] += az.mean().item()
    acc["contact"] += cr.mean().item()
    acc["contact_ref"] += ref_c.sum(dim=-1).mean().item()
    acc["imit_total"] += (jpr + jvr + lxy + lz + axy + az + cr).mean().item()
    acc["cmd_active"] += (torch.linalg.norm(u._command[:, :3], dim=-1) > 0.01).float().mean().item()
    acc["n_feet_down"] += contacts.sum(dim=-1).mean().item()
    acc["base_speed"] += torch.linalg.norm(blv[:, :2], dim=-1).mean().item()
    acc["ref_speed"] += torch.linalg.norm(ref_lv[:, :2], dim=-1).mean().item()

    if prev_contact is not None:
        t = (contacts != prev_contact).float().sum(dim=-1)
        toggles = t if toggles is None else toggles + t
    prev_contact = contacts.clone()
    n += 1

d = {k: v / n for k, v in acc.items()}
tog = (toggles / n).mean().item() if toggles is not None else 0.0
S = cfg.imitation_scale
print("\n" + "=" * 70)
print(f"IMITATION INTERNALS  ({n} steps, {args_cli.num_envs} envs)")
print("=" * 70)
print(f"  joint_pos sum_err^2 (rad^2)   : {d['joint_pos_err2']:9.4f}  -> {(d['joint_pos_err2']/10)**0.5*57.3:.1f} deg/joint")
print(f"  joint_pos_rew                 : {d['joint_pos_rew']:+9.4f}   (max +1.0)")
print(f"  joint_vel_rew                 : {d['joint_vel_rew']:+9.4f}   (UNBOUNDED neg, err2={d['joint_vel_err2']:.1f})")
print(f"  lin_vel_xy_rew                : {d['lin_xy']:+9.4f}   (max +1.0)")
print(f"  lin_vel_z_rew                 : {d['lin_z']:+9.4f}   (max +1.0)")
print(f"  ang_vel_xy_rew                : {d['ang_xy']:+9.4f}   (max +0.5)")
print(f"  ang_vel_z_rew                 : {d['ang_z']:+9.4f}   (max +0.5)")
print(f"  contact_rew                   : {d['contact']:+9.4f}   ({'swing-only, max +2.0' if cfg.imitation_swing_only_contact else 'agreement, max +2.0'})")
print("-" * 70)
print(f"  IMITATION TOTAL (raw)         : {d['imit_total']:+9.4f}")
print(f"  x imitation_scale({S})         : {d['imit_total']*S:+9.4f}  per step")
print("=" * 70)
print("BEHAVIOR (is it actually walking?)")
print(f"  feet on ground (0-2)          : {d['n_feet_down']:9.3f}   reference: {d['contact_ref']:.3f}")
print(f"  contact toggles / step        : {tog:9.4f}   (0 = feet never leave ground)")
print(f"  base speed (m/s)              : {d['base_speed']:9.4f}   reference: {d['ref_speed']:.4f}")
print(f"  active-command fraction       : {d['cmd_active']:9.3f}")
print("=" * 70, flush=True)
env.close()
simulation_app.close()
