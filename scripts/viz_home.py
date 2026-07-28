"""Holds the robot at its home pose (action=0) and streams it, so the new
crouch home pose can be eyeballed before committing a training run to it."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg_A20J5_Bounded  # noqa: E402
from open_duck_mini_isaaclab.joint_order import READY_BASE_HEIGHT  # noqa: E402

cfg = JoystickEnvCfg_A20J5_Bounded()
cfg.scene.num_envs = 1
cfg.min_base_height_ratio = 0.0   # never terminate; this is a static viewer
cfg.events.push_robot = None      # disable the 5-10s random push (up to 1 m/s):
                                  # with action=0 there is no controller to
                                  # recover from it, so it just topples the
                                  # robot and tells us nothing about whether
                                  # the READY pose itself is stable
cfg.episode_length_s = 10000.0
env = gym.make("Isaac-OpenDuckMini-Joystick-A20J5Bounded-v0", cfg=cfg)
u = env.unwrapped
u.reset()
zero = torch.zeros(1, len(u._joint_ids), device=u.device)
print(f"[info] holding READY pose, push disabled, spawn READY_BASE_HEIGHT={READY_BASE_HEIGHT}", flush=True)
i = 0
while simulation_app.is_running():
    u.step(zero)
    i += 1
    if i % 250 == 0:
        print(f"[step {i}] base_z={u._robot.data.root_pos_w[0,2].item():.4f} "
              f"upright={u._robot.data.projected_gravity_b[0,2].item():+.4f}", flush=True)
env.close()
simulation_app.close()
