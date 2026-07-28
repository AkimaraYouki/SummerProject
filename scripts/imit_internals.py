"""Breaks reward_imitation into its INTERNAL sub-terms for a trained policy.

reward_breakdown_v2.py only shows imitation's total; when the policy visibly
refuses to imitate, the question is which sub-term is actually dead -- e.g.
a joint_pos exp() saturated to ~0 gives a flat gradient, so the policy can't
feel any progress toward the reference and just optimizes `alive` instead.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

_MAP = {
    "Isaac-OpenDuckMini-Joystick-A20J5-v0": "JoystickEnvCfg_A20J5",
    "Isaac-OpenDuckMini-Joystick-A20J5NoRSI-v0": "JoystickEnvCfg_A20J5_NoRSI",
    "Isaac-OpenDuckMini-Joystick-A20J5Bounded-v0": "JoystickEnvCfg_A20J5_Bounded",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = JoystickPPORunnerCfg()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
cfg = u.cfg
print(f"[info] w_joint_pos={cfg.imitation_w_joint_pos} bounded={cfg.imitation_bounded_joint_pos} scale={cfg.imitation_scale}", flush=True)

obs, _ = env.get_observations()
acc = {k: 0.0 for k in ["joint_pos_err2", "joint_pos_rew", "joint_vel_rew", "lin_xy", "lin_z", "ang_xy", "ang_z", "contact", "cmd_active"]}
n = 0
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
    contacts = u._get_foot_contact()
    blv = u._robot.data.root_lin_vel_w
    bav = u._robot.data.root_ang_vel_w

    err2 = torch.sum((jp - ref_jp) ** 2, dim=-1)
    w = cfg.imitation_w_joint_pos
    jpr = torch.exp(-w * err2) if cfg.imitation_bounded_joint_pos else -err2 * w
    acc["joint_pos_err2"] += err2.mean().item()
    acc["joint_pos_rew"] += jpr.mean().item()
    acc["joint_vel_rew"] += (-torch.sum((jv - ref_jv) ** 2, dim=-1) * 1.0e-3).mean().item()
    acc["lin_xy"] += (torch.exp(-8.0 * torch.sum((blv[:, :2] - ref_lv[:, :2]) ** 2, dim=-1)) * 1.0).mean().item()
    acc["lin_z"] += (torch.exp(-8.0 * (blv[:, 2] - ref_lv[:, 2]) ** 2) * 1.0).mean().item()
    acc["ang_xy"] += (torch.exp(-2.0 * torch.sum((bav[:, :2] - ref_av[:, :2]) ** 2, dim=-1)) * 0.5).mean().item()
    acc["ang_z"] += (torch.exp(-2.0 * (bav[:, 2] - ref_av[:, 2]) ** 2) * 0.5).mean().item()
    acc["contact"] += (torch.sum((contacts == ref_c).float(), dim=-1) * 1.0).mean().item()
    acc["cmd_active"] += (torch.linalg.norm(u._command[:, :3], dim=-1) > 0.01).float().mean().item()
    n += 1

print("\n" + "=" * 66)
print(f"IMITATION INTERNALS (raw, pre-scale/pre-dt) over {n} steps")
print("=" * 66)
print(f"  joint_pos sum_err^2 (rad^2, 10 joints) : {acc['joint_pos_err2']/n:8.3f}")
print(f"  joint_pos_rew                          : {acc['joint_pos_rew']/n:+8.4f}   (max +1.0 if bounded)")
print(f"  joint_vel_rew                          : {acc['joint_vel_rew']/n:+8.4f}   (unbounded negative)")
print(f"  lin_vel_xy_rew                         : {acc['lin_xy']/n:+8.4f}   (max +1.0)")
print(f"  lin_vel_z_rew                          : {acc['lin_z']/n:+8.4f}   (max +1.0)")
print(f"  ang_vel_xy_rew                         : {acc['ang_xy']/n:+8.4f}   (max +0.5)")
print(f"  ang_vel_z_rew                          : {acc['ang_z']/n:+8.4f}   (max +0.5)")
print(f"  contact_rew                            : {acc['contact']/n:+8.4f}   (max +2.0)")
print(f"  fraction of envs with active command   : {acc['cmd_active']/n:8.3f}")
print("=" * 66, flush=True)
env.close()
simulation_app.close()
