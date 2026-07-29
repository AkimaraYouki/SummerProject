"""Finds the (neck_pitch, head_pitch) pair that leaves the head horizontal.

The user wants a BD-X-style "Z" neck: the neck leans over while the head
counter-rotates back to level. Which sign of head_pitch cancels a given
neck_pitch isn't obvious from the URDF (both joints list axis 0 0 1 in their
own frames), so rather than assume, this drives the joints directly and reads
the head body's actual world-frame pitch back out.
"""
import argparse, math
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg_Walk  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402

COMBOS = [(0.0, 0.0), (0.7, 0.0), (0.7, -0.7), (0.7, 0.7), (1.0, -0.785), (0.5, -0.5), (0.9, -0.785)]
cfg = JoystickEnvCfg_Walk()
cfg.scene.num_envs = len(COMBOS)
cfg.min_base_height_ratio = 0.0
cfg.episode_length_s = 10000.0
env = gym.make("Isaac-OpenDuckMini-Joystick-Walk-v0", cfg=cfg)
u = env.unwrapped
u.reset()
hid, hname = u._robot.find_bodies(["head_pitch_assembly"], preserve_order=True)
tid, _ = u._robot.find_bodies(["trunk_assembly"], preserve_order=True)
ni = ACTUATOR_JOINT_NAMES.index("neck_pitch"); hi = ACTUATOR_JOINT_NAMES.index("head_pitch")

# The PD controller drives toward `default_joint_pos + action*scale`, so
# writing the joint STATE alone gets undone within a few steps -- the default
# (i.e. the action=0 target) has to move too.
u._robot.data.default_joint_pos[:, u._joint_ids[ni]] = torch.tensor(
    [c[0] for c in COMBOS], device=u.device)
u._robot.data.default_joint_pos[:, u._joint_ids[hi]] = torch.tensor(
    [c[1] for c in COMBOS], device=u.device)
jp = u._robot.data.default_joint_pos.clone()
u._robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
u._motor_targets[:] = jp[:, u._joint_ids]
zero = torch.zeros(len(COMBOS), len(u._joint_ids), device=u.device)
for _ in range(120):
    u.step(zero)

def elev_deg(q, axis):
    # elevation of a body-local axis above the horizon, in world frame:
    # 0 deg = the axis lies flat, +90 = points straight up.
    v = mu.quat_apply(q, torch.tensor(axis, device=q.device).expand(q.shape[0], 3))
    return torch.asin(v[:, 2].clamp(-1, 1)) * 57.2958

# Which body actually responds to neck_pitch? Compare every body's +X
# elevation between env0 (neck=0, head=0) and env1 (neck=0.7, head=0):
# only bodies downstream of the neck joint should differ.
ni_n, hi_n = u._joint_ids[ni], u._joint_ids[hi]
print("\n" + "=" * 74)
print("목표 vs 실제 관절값 (neck_pitch / head_pitch)")
print("=" * 74)
print(f"{'env':>4}{'목표neck':>10}{'실제neck':>10}{'목표head':>10}{'실제head':>10}   비고")
for e, (nk, hd) in enumerate(COMBOS):
    an = u._robot.data.joint_pos[e, ni_n].item()
    ah = u._robot.data.joint_pos[e, hi_n].item()
    bad = "  <== 목표 미달" if abs(an - nk) > 0.1 else ""
    print(f"{e:>4}{nk:>10.3f}{an:>10.3f}{hd:>10.3f}{ah:>10.3f}{bad}")
print("=" * 74)
print(f"default_joint_pos[0, neck] = {u._robot.data.default_joint_pos[0, ni_n].item():.3f}")
print(f"default_joint_pos[1, neck] = {u._robot.data.default_joint_pos[1, ni_n].item():.3f}")
print(f"_motor_targets[1, neck_actuator_idx] = {u._motor_targets[1, ni].item():.3f}")
print("=" * 74, flush=True)
env.close(); simulation_app.close()
