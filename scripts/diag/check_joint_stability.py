"""Standalone actuator-gain sanity check — NOT an RL training/eval script.

Bypasses rsl_rl entirely: drives the registered gym env directly with
zero action (== target stays at HOME_JOINT_POS, see joystick_env.py's
`target = default_pos + action * action_scale`) for a fixed window and
reports whether joint positions/velocities/torques and base height/tilt
stay bounded. Exists to catch PD-gain instability (oscillation, joint-limit
violation, NaN, base falling/launching) from a stiffness/damping change
without needing a trained policy or a GUI — see robot_cfg.py's actuator
docstring for the current gain derivation.

Run via scripts/check_joint_stability.sh (delegates through
_isaaclab_launch.py so this repo's task registration is importable, same as
train.sh/play.sh).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action joint stability check.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=300, help="300 steps * 0.02s ctrl_dt = 6s sim time")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401 - side effect: gym.register()
from open_duck_mini_isaaclab.tasks.velocity.joystick_env_cfg import JoystickEnvCfg  # noqa: E402

# gym.make's entry_point needs an explicit cfg instance (train.py/play.py get
# theirs via IsaacLab's @hydra_task_config decorator; this script skips that
# machinery and just builds JoystickEnvCfg directly).
env_cfg = JoystickEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = args_cli.device if getattr(args_cli, "device", None) else env_cfg.sim.device

env = gym.make("Isaac-OpenDuckMini-Joystick-v0", cfg=env_cfg).unwrapped
env.reset()

robot = env._robot
joint_ids = env._joint_ids
default_pos = robot.data.default_joint_pos[:, joint_ids].clone()
soft_limits = robot.data.soft_joint_pos_limits[:, joint_ids, :].clone()

zero_action = torch.zeros(args_cli.num_envs, env.cfg.action_space, device=env.device)

print(f"[info] HOME_BASE_HEIGHT (spawn z) = {env_cfg.robot.init_state.pos[2]:.4f} m", flush=True)

log_delta, log_vel, log_torque, log_base_h, log_up, log_limit_viol = [], [], [], [], [], []
nan_step = None
any_reset_steps = 0

for step in range(args_cli.num_steps):
    if step % 50 == 0:
        print(f"[progress] step {step}/{args_cli.num_steps}", flush=True)
    obs, rew, terminated, truncated, info = env.step(zero_action)

    jp = robot.data.joint_pos[:, joint_ids]
    jv = robot.data.joint_vel[:, joint_ids]
    tq = robot.data.applied_torque[:, joint_ids]
    base_h = robot.data.root_pos_w[:, 2]
    up = robot.data.projected_gravity_b[:, 2]

    if torch.isnan(jp).any() or torch.isnan(jv).any() or torch.isnan(tq).any():
        nan_step = step
        break

    log_delta.append((jp - default_pos).abs().max().item())
    log_vel.append(jv.abs().max().item())
    log_torque.append(tq.abs().max().item())
    log_base_h.append(base_h.min().item())
    log_up.append(up.max().item())  # worst-case env (max == least upright)
    log_limit_viol.append(bool((jp < soft_limits[..., 0]).any() or (jp > soft_limits[..., 1]).any()))
    if terminated.any() or truncated.any():
        any_reset_steps += 1


def _window_report(name: str, lo: int, hi: int):
    n = hi - lo
    print(f"--- {name} (steps {lo}-{hi}, n={n}) ---")
    print(f"  max |joint_pos - default| : {max(log_delta[lo:hi]):.4f} rad")
    print(f"  max |joint_vel|           : {max(log_vel[lo:hi]):.4f} rad/s")
    print(f"  max |applied_torque|      : {max(log_torque[lo:hi]):.4f} Nm  (effort_limit_sim=4.1)")
    print(f"  base height range         : [{min(log_base_h[lo:hi]):.4f}, {max(log_base_h[lo:hi]):.4f}] m")
    print(f"  worst-case upright (-1=perfect, 0=sideways): {max(log_up[lo:hi]):.4f}")
    print(f"  soft-limit violation steps: {sum(log_limit_viol[lo:hi])} / {n}")


print("\n" + "=" * 70)
print("JOINT STABILITY CHECK (zero action, holds HOME_JOINT_POS)")
print("=" * 70)
if nan_step is not None:
    print(f"FAIL: NaN detected at step {nan_step}")
else:
    n = len(log_delta)
    settle_end = min(50, n)
    print(f"steps run: {n} ({n * env.step_dt:.1f}s sim time)")
    print(f"envs terminated/truncated at some point: {any_reset_steps} step(s) had >=1 env reset (out of {n})")
    print()
    _window_report("SETTLING (initial drop/impact transient)", 0, settle_end)
    print()
    if n > settle_end:
        _window_report("STEADY STATE (after settling)", settle_end, n)
        steady_ok = (
            max(log_delta[settle_end:n]) < 0.15
            and sum(log_limit_viol[settle_end:n]) == 0
            and max(log_up[settle_end:n]) < -0.85
        )
        print("\nSTEADY-STATE RESULT:", "PASS (stable stand)" if steady_ok else "FAIL (still unstable after settling)")
print("=" * 70)
sys.stdout.flush()

env.close()
simulation_app.close()
