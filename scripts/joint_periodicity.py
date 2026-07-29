"""Logs leg-joint trajectories from a trained policy and dumps them for plotting.

Answers a question the reward numbers cannot: is the policy producing PERIODIC
joint motion (walking) or high-frequency jitter around a fixed pose (the
trembling failure mode seen in v10/v11)? Saves raw arrays to .npz; plotting
happens off-box so this stays dependency-light and rerunnable.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--out", type=str, default="/home/do/joint_traj.npz")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    JoystickPPORunnerCfg,
    JoystickPPORunnerCfg_Upstream,
    JoystickPPORunnerCfg_BigNet,
    JoystickPPORunnerCfg_BigNetLowEnt,
    JoystickPPORunnerCfg_BigNetMB16,
    JoystickPPORunnerCfg_Gamma097,
)

# The runner cfg must match the one the checkpoint was TRAINED with, not just
# the env cfg: Walk9 trains with the upstream network (512,256,128) while every
# other variant uses (256,128,64), and loading across them fails with a bare
# size-mismatch on actor.0.weight.
_TASK_TO_RUNNER = {
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": JoystickPPORunnerCfg_Upstream,
    "Isaac-OpenDuckMini-Joystick-Walk9Big-v0": JoystickPPORunnerCfg_BigNet,
    "Isaac-OpenDuckMini-Joystick-Walk9BigLE-v0": JoystickPPORunnerCfg_BigNetLowEnt,
    "Isaac-OpenDuckMini-Joystick-Walk9MB16-v0": JoystickPPORunnerCfg_BigNetMB16,
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": JoystickPPORunnerCfg_Gamma097,
}

from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX, ACTUATOR_JOINT_NAMES,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

_MAP = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9Big-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9BigLE-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9MB16-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = _TASK_TO_RUNNER.get(args_cli.task, JoystickPPORunnerCfg)()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

# Pin every env to a real forward-walk command so the reference actually asks
# for stepping — with random commands, near-zero ones make standing correct and
# the periodicity question meaningless.
FWD = 0.15
leg_names = [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX]
print(f"[info] gait_period_steps={u._gait_period_steps} legs={leg_names}", flush=True)

qpos, qref, feet, phase, vel = [], [], [], [], []
obs, _ = env.get_observations()
for step in range(args_cli.num_steps):
    u._command[:, 0] = FWD
    u._command[:, 1] = 0.0
    u._command[:, 2] = 0.0
    with torch.inference_mode():
        actions = policy(obs)
    obs, rew, dones, infos = env.step(actions)
    qpos.append(u._robot.data.joint_pos[:, u._joint_ids][:, ACT_LEG_JOINT_IDX].cpu().numpy())
    rf = u._current_reference_motion
    qref.append(rf[:, 0:14][:, REF_LEG_JOINT_IDX].cpu().numpy())
    feet.append(u._get_foot_contact().float().cpu().numpy())
    phase.append(u._imitation_i.float().cpu().numpy())
    vel.append(u._robot.data.root_lin_vel_w[:, :2].cpu().numpy())

np.savez_compressed(
    args_cli.out,
    qpos=np.asarray(qpos), qref=np.asarray(qref), feet=np.asarray(feet),
    phase=np.asarray(phase), vel=np.asarray(vel),
    leg_names=np.array(leg_names), gait_period_steps=u._gait_period_steps,
    ctrl_dt=u.step_dt, cmd_fwd=FWD, checkpoint=args_cli.checkpoint,
)
print(f"[ok] wrote {args_cli.out}  qpos shape={np.asarray(qpos).shape}", flush=True)
env.close()
simulation_app.close()
