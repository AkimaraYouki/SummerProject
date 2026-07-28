"""Drops the robot in the new home (crouch) pose and reports where it settles.

HOME_BASE_HEIGHT feeds both the spawn height and the `collapsed` termination
threshold (HOME_BASE_HEIGHT * min_base_height_ratio), and its old value (0.193)
was measured with straight legs. With home now the reference gait's crouch the
robot stands lower, so spawning at 0.193 would drop it and the old threshold
would be wrong -- this measures the real number instead of guessing it.
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg_A20J5_Bounded  # noqa: E402
from open_duck_mini_isaaclab.joint_order import HOME_BASE_HEIGHT  # noqa: E402

cfg = JoystickEnvCfg_A20J5_Bounded()
cfg.scene.num_envs = args_cli.num_envs
cfg.min_base_height_ratio = 0.0   # don't terminate while we measure the settle
env = gym.make("Isaac-OpenDuckMini-Joystick-A20J5Bounded-v0", cfg=cfg)
u = env.unwrapped
u.reset()
zero = torch.zeros(args_cli.num_envs, len(u._joint_ids), device=u.device)
hs = []
for i in range(args_cli.num_steps):
    u.step(zero)                      # action=0 => hold exactly the home pose
    hs.append(u._robot.data.root_pos_w[:, 2].clone())
h = torch.stack(hs)
last = h[-100:]
print("=" * 62)
print(f"SETTLE TEST — action=0 (hold home pose), {args_cli.num_steps} steps")
print("=" * 62)
print(f"  spawn height (config HOME_BASE_HEIGHT) : {HOME_BASE_HEIGHT:.4f} m")
print(f"  settled height, last 100 steps         : [{last.min():.4f}, {last.max():.4f}] m")
print(f"  settled mean                           : {last.mean():.4f} m")
print(f"  drop from spawn                        : {HOME_BASE_HEIGHT - last.mean():.4f} m")
print("=" * 62)
print(f"  -> suggested HOME_BASE_HEIGHT = {last.mean():.3f}")
print(f"  -> collapse threshold @0.75 would be  = {last.mean()*0.75:.4f} m")
print("=" * 62, flush=True)
env.close()
simulation_app.close()
