"""Prints the mean per-step contribution of EACH reward term separately,
for a trained checkpoint rolled out with its real actions. Exists because
rsl_rl only logs the aggregate `Train/mean_reward` (full-episode cumulative
sum) to tensorboard -- no per-term breakdown -- so there was no way to see
which term was actually dominating/cancelling without this.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-OpenDuckMini-Joystick-A20J5-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=100)
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
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cfg_module  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity.rewards import (  # noqa: E402
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    reward_alive,
    cost_stand_still,
    reward_imitation,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

# Resolve the cfg class from --task by directly instantiating the matching
# JoystickEnvCfg_* class (matches contact_diagnostic.py's proven-stable
# pattern) instead of isaaclab_tasks.utils.parse_env_cfg -- parse_env_cfg
# was tried here first (to fix a real bug: hardcoding JoystickEnvCfg_A20J5
# silently used the WRONG alive_scale/imitation_w_joint_pos for any other
# checkpoint) but every run using it crashed silently with no traceback,
# while contact_diagnostic.py's direct-instantiation pattern never has --
# suspect parse_env_cfg's extra registry/hydra-config machinery introduces
# instability. This keeps the correctness fix (right weights per --task)
# without parse_env_cfg.
_TASK_TO_CFG_CLASS = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Alive5-v0": "JoystickEnvCfg_Alive5",
    "Isaac-OpenDuckMini-Joystick-Alive10-v0": "JoystickEnvCfg_Alive10",
    "Isaac-OpenDuckMini-Joystick-A10J10-v0": "JoystickEnvCfg_A10J10",
    "Isaac-OpenDuckMini-Joystick-A5J15-v0": "JoystickEnvCfg_A5J15",
    "Isaac-OpenDuckMini-Joystick-A20J5-v0": "JoystickEnvCfg_A20J5",
    "Isaac-OpenDuckMini-Joystick-A5J5-v0": "JoystickEnvCfg_A5J5",
    "Isaac-OpenDuckMini-Joystick-A30J25-v0": "JoystickEnvCfg_A30J25",
    "Isaac-OpenDuckMini-Joystick-A20J5NoRSI-v0": "JoystickEnvCfg_A20J5_NoRSI",
    "Isaac-OpenDuckMini-Joystick-A30J25Im2N2-v0": "JoystickEnvCfg_A30J25Im2",
}
_cfg_cls_name = _TASK_TO_CFG_CLASS[args_cli.task]
env_cfg = getattr(_cfg_module, _cfg_cls_name)()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = JoystickPPORunnerCfg()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"[info] task={args_cli.task} checkpoint={args_cli.checkpoint}", flush=True)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

unwrapped = env.unwrapped
cfg = unwrapped.cfg
print(f"[info] alive_scale={cfg.alive_scale} imitation_w_joint_pos={cfg.imitation_w_joint_pos} imitation_scale={cfg.imitation_scale}", flush=True)

obs = env.get_observations()

sums = {}
n_alive = torch.zeros(args_cli.num_envs, device=unwrapped.device)  # count of non-terminal steps
n_clamped = 0
clamp_loss_sum = 0.0

for step in range(args_cli.num_steps):
    with torch.inference_mode():
        actions = policy(obs)
    obs, rew, dones, infos = env.step(actions)

    joint_pos = unwrapped._robot.data.joint_pos[:, unwrapped._joint_ids]
    joint_vel = unwrapped._robot.data.joint_vel[:, unwrapped._joint_ids]
    default_joint_pos = unwrapped._robot.data.default_joint_pos[:, unwrapped._joint_ids]
    contact = unwrapped._get_foot_contact()

    terms = {
        "tracking_lin_vel": reward_tracking_lin_vel(unwrapped._command, unwrapped._robot.data.root_lin_vel_b, cfg.tracking_sigma) * cfg.tracking_lin_vel_scale,
        "tracking_ang_vel": reward_tracking_ang_vel(unwrapped._command, unwrapped._imu.data.ang_vel_b, cfg.tracking_sigma) * cfg.tracking_ang_vel_scale,
        "torques": cost_torques(unwrapped._robot.data.applied_torque[:, unwrapped._joint_ids]) * cfg.torques_scale,
        "action_rate": cost_action_rate(unwrapped._actions, unwrapped._last_act) * cfg.action_rate_scale,
        "alive": reward_alive(unwrapped.num_envs, unwrapped.device) * cfg.alive_scale,
        "stand_still": cost_stand_still(unwrapped._command, joint_pos, joint_vel, default_joint_pos) * cfg.stand_still_scale,
    }
    if cfg.use_imitation:
        terms["imitation"] = reward_imitation(
            unwrapped._robot.data.root_lin_vel_w,
            unwrapped._robot.data.root_ang_vel_w,
            joint_pos, joint_vel, contact,
            unwrapped._current_reference_motion,
            unwrapped._command,
            w_joint_pos=cfg.imitation_w_joint_pos,
        ) * cfg.imitation_scale

    raw_sum_step = torch.sum(torch.stack(list(terms.values())), dim=0)  # pre-*dt, pre-clamp
    scaled_step = raw_sum_step * unwrapped.step_dt  # matches env's own `* dt`
    clamped_step = torch.clamp(scaled_step, 0.0, 10000.0)  # matches env's own clamp

    n_clamped += (scaled_step < 0.0).sum().item()
    clamp_loss_sum += torch.clamp(-scaled_step, min=0.0).sum().item()  # magnitude eaten by the floor

    for k, v in terms.items():
        sums.setdefault(k, torch.zeros(args_cli.num_envs, device=unwrapped.device))
        sums[k] += v * unwrapped.step_dt  # each term's own *dt contribution (NOT clamped individually)
    sums.setdefault("__PRECLAMP_TOTAL__", torch.zeros(args_cli.num_envs, device=unwrapped.device))
    sums["__PRECLAMP_TOTAL__"] += scaled_step
    sums.setdefault("__POSTCLAMP_TOTAL__", torch.zeros(args_cli.num_envs, device=unwrapped.device))
    sums["__POSTCLAMP_TOTAL__"] += clamped_step

print("\n" + "=" * 70)
print(f"PER-TERM MEAN CONTRIBUTION over {args_cli.num_steps} steps ({args_cli.num_steps * unwrapped.step_dt:.1f}s), averaged over {args_cli.num_envs} envs")
print("=" * 70)
n_total_steps = args_cli.num_steps * args_cli.num_envs
for k, v in sums.items():
    if k.startswith("__"):
        continue
    print(f"  {k:18s} sum={v.mean().item():+9.4f}   per-step-avg={v.mean().item()/args_cli.num_steps:+.5f}")
preclamp = sums["__PRECLAMP_TOTAL__"]
postclamp = sums["__POSTCLAMP_TOTAL__"]
print(f"  {'SUM (pre-clamp)':18s} sum={preclamp.mean().item():+9.4f}   per-step-avg={preclamp.mean().item()/args_cli.num_steps:+.5f}")
print(f"  {'SUM (post-clamp, = env reward)':30s} sum={postclamp.mean().item():+9.4f}   per-step-avg={postclamp.mean().item()/args_cli.num_steps:+.5f}")
print("-" * 70)
print(f"  steps where scaled reward < 0 (clamped to 0): {n_clamped}/{n_total_steps} ({100*n_clamped/n_total_steps:.1f}%)")
print(f"  total reward magnitude eaten by the floor clamp: {clamp_loss_sum:.4f} (avg/step={clamp_loss_sum/n_total_steps:.5f})")
print("=" * 70)

env.close()
simulation_app.close()
