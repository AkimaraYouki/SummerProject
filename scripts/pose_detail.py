"""Reports trunk roll/pitch/yaw in degrees plus base drift for the home pose.

The earlier viz only logged projected_gravity_b[2] (= total tilt magnitude),
which reads -0.9987 whether the robot leans forward or backward -- it cannot
distinguish pitch direction, so it can't confirm or refute "it's falling
forward". This prints signed euler angles and x/y drift instead.
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, math, gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg_A20J5_Bounded  # noqa: E402

cfg = JoystickEnvCfg_A20J5_Bounded()
cfg.scene.num_envs = 4
cfg.min_base_height_ratio = 0.0
cfg.episode_length_s = 10000.0
env = gym.make("Isaac-OpenDuckMini-Joystick-A20J5Bounded-v0", cfg=cfg)
u = env.unwrapped
u.reset()
zero = torch.zeros(4, len(u._joint_ids), device=u.device)
x0 = u._robot.data.root_pos_w[:, 0].clone()
y0 = u._robot.data.root_pos_w[:, 1].clone()
print(f"{'step':>5}{'base_z':>9}{'roll°':>8}{'pitch°':>9}{'dx(cm)':>9}{'dy(cm)':>9}", flush=True)
for i in range(args_cli.num_steps):
    u.step(zero)
    if i % 50 == 0 or i == args_cli.num_steps - 1:
        q = u._robot.data.root_quat_w
        r, p, _ = mu.euler_xyz_from_quat(q)
        r = ((r + math.pi) % (2*math.pi) - math.pi) * 57.2958
        p = ((p + math.pi) % (2*math.pi) - math.pi) * 57.2958
        dx = (u._robot.data.root_pos_w[:, 0] - x0) * 100
        dy = (u._robot.data.root_pos_w[:, 1] - y0) * 100
        print(f"{i:>5}{u._robot.data.root_pos_w[:,2].mean():>9.4f}{r.mean():>8.2f}"
              f"{p.mean():>9.2f}{dx.mean():>9.2f}{dy.mean():>9.2f}", flush=True)
print("\n(pitch° > 0 = leaning forward/nose-down depending on convention;"
      " dx > 0 = drifting +x)", flush=True)
env.close(); simulation_app.close()
