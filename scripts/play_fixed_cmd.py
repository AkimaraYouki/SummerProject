"""play.py with the velocity command PINNED instead of randomly resampled.

rsl_rl's play.py drives the env's own command sampler, so the robot is
constantly switching between forward/backward/lateral/turn targets and you
cannot tell from the video which one it is currently reacting to. For judging
"does it walk forward", pin the command.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cmd_x", type=float, default=0.15)
parser.add_argument("--cmd_y", type=float, default=0.0)
parser.add_argument("--cmd_yaw", type=float, default=0.0)
parser.add_argument("--seconds", type=float, default=1e9)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg  # noqa: E402
from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

_MAP = {
    "Isaac-OpenDuckMini-Joystick-Walk-v0": "JoystickEnvCfg_Walk",
    "Isaac-OpenDuckMini-Joystick-Walk2-v0": "JoystickEnvCfg_Walk2",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk4-v0": "JoystickEnvCfg_Walk4",
    "Isaac-OpenDuckMini-Joystick-Walk5-v0": "JoystickEnvCfg_Walk5",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk7-v0": "JoystickEnvCfg_Walk7",
    "Isaac-OpenDuckMini-Joystick-Walk8-v0": "JoystickEnvCfg_Walk8",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = JoystickPPORunnerCfg()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
dt = u.step_dt
print(f"[play] cmd pinned to ({args_cli.cmd_x:+.2f}, {args_cli.cmd_y:+.2f}, {args_cli.cmd_yaw:+.2f})", flush=True)

obs, _ = env.get_observations()
t_end = time.time() + args_cli.seconds
step = 0
while simulation_app.is_running() and time.time() < t_end:
    t0 = time.time()
    u._command[:, 0] = args_cli.cmd_x
    u._command[:, 1] = args_cli.cmd_y
    u._command[:, 2] = args_cli.cmd_yaw
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    step += 1
    if step % 250 == 0:
        v = u._robot.data.root_lin_vel_b[0, :2]
        print(f"[play] step {step}  vx={v[0]:+.3f} vy={v[1]:+.3f} (cmd {args_cli.cmd_x:+.2f})", flush=True)
    sleep = dt - (time.time() - t0)
    if sleep > 0:
        time.sleep(sleep)
env.close()
simulation_app.close()
