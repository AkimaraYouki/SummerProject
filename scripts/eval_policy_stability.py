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

Standing-upright alone is ALSO not sufficient proof of a working policy —
a policy can just as easily reward-hack by standing perfectly still (or
twitching in place) instead of tracking the commanded velocity, which
would still pass a height/upright-only check. So this also checks, over
the same rollout (commands come from the env's own resampling, same as
training):
  - lin-vel tracking error (achieved root_lin_vel_b.xy vs the commanded
    lin_vel_x/y), restricted to steps where the command isn't ~zero
  - leg joint range of motion (peak-to-peak per leg joint) — near-zero
    range across the board means the legs aren't moving, i.e. standing
    still or jittering in place rather than stepping
  - foot contact alternation (toggle count on the existing per-foot
    contact sensor, same FOOT_CONTACT_FORCE_THRESHOLD used by the env's
    own "contact" observation) — a real gait lifts each foot repeatedly;
    a foot that's always in contact (0 toggles) or never in contact means
    no stepping is happening regardless of what the joints are doing

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
# 2026-07-28: this script used to hardcode the BASE JoystickEnvCfg and the base
# gym task regardless of what the checkpoint was actually trained on. For v3-v9
# that was harmless — those variants differ only in reward weights, and this
# script measures rollout BEHAVIOR (toggle / leg ROM / vel error), never reward,
# so the historical toggle numbers remain comparable. It stopped being harmless
# at JoystickEnvCfg_Walk: that variant trains with `lock_head_joints=True` and
# all four head command ranges pinned to (0,0), so evaluating such a policy
# under the base cfg both (a) feeds it random head commands it never saw during
# training — an observation-distribution shift — and (b) lets its untrained head
# action outputs actually drive the head. Default stays the base task so old
# invocations reproduce exactly.
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-OpenDuckMini-Joystick-v0",
    help="gym task id the checkpoint was TRAINED with (must match, or eval is measuring a different env)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401 - side effect: gym.register()
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    JoystickPPORunnerCfg,
    JoystickPPORunnerCfg_Upstream,
    JoystickPPORunnerCfg_BigNet,
)

# The runner cfg must match the one the checkpoint was TRAINED with, not just
# the env cfg: Walk9 trains with the upstream network (512,256,128) while every
# other variant uses (256,128,64), and loading across them fails with a bare
# size-mismatch on actor.0.weight.
_TASK_TO_RUNNER = {
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": JoystickPPORunnerCfg_Upstream,
    "Isaac-OpenDuckMini-Joystick-Walk9Big-v0": JoystickPPORunnerCfg_BigNet,
}

from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity.joystick_env import FOOT_CONTACT_FORCE_THRESHOLD  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cfg_module  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

# Explicit task -> cfg-class table rather than isaaclab's parse_env_cfg: an
# earlier attempt to use parse_env_cfg here made this script die silently on
# every run (no traceback, no OOM, no signal), six times in a row, and the
# explicit-dict version worked first try. Keep it explicit.
_TASK_TO_CFG_CLASS = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9Big-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
if args_cli.task not in _TASK_TO_CFG_CLASS:
    raise SystemExit(
        f"unknown --task {args_cli.task!r}; add it to _TASK_TO_CFG_CLASS. known: {sorted(_TASK_TO_CFG_CLASS)}"
    )
env_cfg = getattr(_cfg_module, _TASK_TO_CFG_CLASS[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
print(f"[info] task={args_cli.task} cfg={_TASK_TO_CFG_CLASS[args_cli.task]}", flush=True)

env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = _TASK_TO_RUNNER.get(args_cli.task, JoystickPPORunnerCfg)()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"[info] loading checkpoint: {args_cli.checkpoint}", flush=True)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

unwrapped = env.unwrapped
robot = unwrapped._robot
joint_ids = unwrapped._joint_ids
contact_sensor = unwrapped._contact_sensor
feet_ids = unwrapped._feet_ids
leg_joint_ids = [joint_ids[i] for i in ACT_LEG_JOINT_IDX]

print(f"[info] HOME_BASE_HEIGHT (spawn z) = {env_cfg.robot.init_state.pos[2]:.4f} m", flush=True)

obs, _ = env.get_observations()

log_base_h, log_up = [], []
log_leg_pos = []  # per-step, mean-across-envs leg joint angles (10,)
vel_err_sum, vel_err_count = 0.0, 0
prev_contact = None
toggle_count = torch.zeros(args_cli.num_envs, 2, device=unwrapped.device)
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
    log_leg_pos.append(robot.data.joint_pos[:, leg_joint_ids].mean(dim=0).tolist())

    commanded = unwrapped._command[:, :2]
    achieved = robot.data.root_lin_vel_b[:, :2]
    moving_mask = commanded.norm(dim=-1) > 0.02
    if moving_mask.any():
        err = (achieved[moving_mask] - commanded[moving_mask]).norm(dim=-1)
        vel_err_sum += err.sum().item()
        vel_err_count += int(moving_mask.sum().item())

    forces = contact_sensor.data.net_forces_w_history[:, 0, feet_ids, :]
    contact = torch.norm(forces, dim=-1) > FOOT_CONTACT_FORCE_THRESHOLD  # (num_envs, 2)
    if prev_contact is not None:
        toggle_count += (contact != prev_contact).float()
    prev_contact = contact

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

    import statistics

    leg_pos_by_joint = list(zip(*log_leg_pos))  # 10 series of length n
    leg_rom = [max(series) - min(series) for series in leg_pos_by_joint]
    mean_leg_rom = statistics.mean(leg_rom)

    mean_vel_err = vel_err_sum / vel_err_count if vel_err_count > 0 else float("nan")
    mean_toggles_per_foot = toggle_count.mean().item()

    print(f"\nmean leg-joint range of motion (10 joints): {mean_leg_rom:.4f} rad  (min {min(leg_rom):.4f}, max {max(leg_rom):.4f})")
    print(f"lin-vel tracking error (moving steps only) : {mean_vel_err:.4f} m/s  (n={vel_err_count} env-steps with |cmd|>0.02)")
    print(f"mean foot-contact toggles per foot per env  : {mean_toggles_per_foot:.1f}  (over {n} steps / {n * env.unwrapped.step_dt:.1f}s)")

    walking = mean_leg_rom > 0.15 and mean_vel_err < 0.15 and mean_toggles_per_foot > 4

    print("\nSTANDING RESULT:", "LIKELY STANDING (not collapsed)" if standing else "LIKELY STILL COLLAPSED/UNSTABLE")
    print("WALKING RESULT: ", "LIKELY ACTUALLY WALKING" if walking else "LIKELY REWARD-HACKING (standing/twitching, not stepping)")
print("=" * 70)
sys.stdout.flush()

env.close()
simulation_app.close()
