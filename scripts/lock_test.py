"""Verifies lock_head_joints actually holds the head against large actions.

Feeds max-magnitude actions on every DOF: with the lock working, the 10 leg
joints must move and the 4 head joints must not.
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg_Walk  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES as AN  # noqa: E402

cfg = JoystickEnvCfg_Walk(); cfg.scene.num_envs = 4
cfg.min_base_height_ratio = 0.0; cfg.episode_length_s = 10000.0
env = gym.make("Isaac-OpenDuckMini-Joystick-Walk-v0", cfg=cfg)
u = env.unwrapped; u.reset()
start = u._robot.data.joint_pos[:, u._joint_ids].clone()
big = torch.full((4, len(u._joint_ids)), 3.0, device=u.device)   # far beyond normal policy output
for i in range(150):
    u.step(big if i % 2 == 0 else -big)                          # slam back and forth
moved = (u._robot.data.joint_pos[:, u._joint_ids] - start).abs().mean(0)
print("\n" + "=" * 64)
print("머리 잠금 검증 — 액션 ±3.0 강제 인가 후 관절 이동량")
print("=" * 64)
for i, n in enumerate(AN):
    head = n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    tag = "[머리]" if head else "[다리]"
    v = moved[i].item()
    verdict = ("  <== 잠김 OK" if v < 0.02 else "  <== !! 움직임") if head else ""
    print(f"  {tag} {n:<16}{v:>8.4f} rad{verdict}")
print("=" * 64)
h = moved[[AN.index(x) for x in ("neck_pitch","head_pitch","head_yaw","head_roll")]].max().item()
l = moved[[i for i,n in enumerate(AN) if n not in ("neck_pitch","head_pitch","head_yaw","head_roll")]].mean().item()
print(f"  머리 최대 이동 {h:.4f} rad | 다리 평균 이동 {l:.4f} rad")
print(f"  => {'잠금 정상 (머리 고정, 다리만 학습됨)' if h < 0.02 and l > 0.05 else '확인 필요'}")
print("=" * 64, flush=True)
env.close(); simulation_app.close()
