"""Trained-policy stability check — NOT a substitute for watching it, but a
numeric proxy that doesn't require a GUI/WebRTC session.

Loads a checkpoint and rolls it out with its REAL actions (unlike
check_joint_stability.py, which uses zero action) for a fixed window,
reporting base height / upright over time. Exists because reward and
episode-length alone are not trustworthy success signals for this task —
see docs/decisions.md's termination-fix note: a 3000-iter run once reported
reward=354/episode_length=902 while the robot was actually collapsed in a
heap the whole time, because the old termination check never caught that
pose. This script checks the thing that actually matters (does base height
stay near HOME_BASE_HEIGHT and stay upright) instead of trusting the
training log's summary stats.

Run via scripts/eval_policy_stability.sh:
  ISAACLAB_PATH=/path/to/IsaacLab ./scripts/eval_policy_stability.sh \
      --checkpoint /path/to/model_XXXX.pt [--headless] [--num_envs 8] [--num_steps 500]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trained-policy rollout stability check.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=500, help="500 steps * 0.02s ctrl_dt = 10s sim time")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401 - side effect: gym.register()
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = JoystickEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs

env = gym.make("Isaac-OpenDuckMini-Joystick-v0", cfg=env_cfg)
agent_cfg = JoystickPPORunnerCfg()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"[info] loading checkpoint: {args_cli.checkpoint}", flush=True)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

robot = env.unwrapped._robot
joint_ids = env.unwrapped._joint_ids
default_pos = robot.data.default_joint_pos[:, joint_ids].clone()

print(f"[info] HOME_BASE_HEIGHT (spawn z) = {env_cfg.robot.init_state.pos[2]:.4f} m", flush=True)

obs, _ = env.get_observations()

log_base_h, log_up, log_lin_vel_err = [], [], []
nan_step = None

for step in range(args_cli.num_steps):
    if step % 50 == 0:
        print(f"[progress] step {step}/{args_cli.num_steps}", flush=True)
    with torch.inference_mode():
        actions = policy(obs)
    obs, rew, dones, infos = env.step(actions)

    base_h = robot.data.root_pos_w[:, 2]
    up = robot.data.projected_gravity_b[:, 2]

    if torch.isnan(base_h).any() or torch.isnan(up).any():
        nan_step = step
        break

    log_base_h.append(base_h.min().item())  # worst-case env
    log_up.append(up.max().item())  # worst-case env (least upright)

print("\n" + "=" * 70)
print("TRAINED-POLICY STABILITY CHECK (real actions, not zero action)")
print("=" * 70)
if nan_step is not None:
    print(f"FAIL: NaN detected at step {nan_step}")
else:
    n = len(log_base_h)
    tail = log_base_h[-min(200, n):]
    tail_up = log_up[-min(200, n):]
    print(f"steps run: {n} ({n * env.unwrapped.step_dt:.1f}s sim time)")
    print(f"base height over full run   : [{min(log_base_h):.4f}, {max(log_base_h):.4f}] m")
    print(f"base height, last {len(tail)} steps  : [{min(tail):.4f}, {max(tail):.4f}] m  (HOME=0.150m)")
    print(f"worst-case upright, last {len(tail_up)} steps: {max(tail_up):.4f}  (-1=perfect, 0=sideways, >0=flipped)")
    standing = min(tail) > 0.10 and max(tail_up) < -0.5
    print("\nRESULT:", "LIKELY STANDING (not collapsed)" if standing else "LIKELY STILL COLLAPSED/UNSTABLE")
print("=" * 70)
sys.stdout.flush()

env.close()
simulation_app.close()
