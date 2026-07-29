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
parser.add_argument("--cycle", action="store_true",
                    help="정지/전진/후진/좌/우/회전을 --hold 초씩 순환한다")
parser.add_argument("--hold", type=float, default=8.0, help="--cycle에서 명령 하나를 유지할 초")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402
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
    "Isaac-OpenDuckMini-Joystick-Path-v0": JoystickPPORunnerCfg_Gamma097,
}

from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm  # noqa: E402
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
    "Isaac-OpenDuckMini-Joystick-Path-v0": "JoystickEnvCfg_Path",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}
env_cfg = getattr(_cm, _MAP[args_cli.task])()
env_cfg.scene.num_envs = args_cli.num_envs
# 재생에는 외란을 항상 끈다. push_robot은 5~10초마다 ±1 m/s로 몸통을 밀어서
# 학습 때는 강건성을 주지만, 눈으로 보행을 판단할 때는 정책의 문제인지 외력
# 때문인지 구분할 수 없게 만든다.
env_cfg.events.push_robot = None
print("[play] 외란(push_robot) 비활성화", flush=True)
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = _TASK_TO_RUNNER.get(args_cli.task, JoystickPPORunnerCfg)()
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
dt = u.step_dt
print(f"[play] cmd pinned to ({args_cli.cmd_x:+.2f}, {args_cli.cmd_y:+.2f}, {args_cli.cmd_yaw:+.2f})", flush=True)

# 순환 모드용 명령 목록. gait_compare.py의 조건과 같은 값이라 영상과 측정치를
# 직접 대응시켜 볼 수 있다.
CYCLE = [
    ("정지",  0.00,  0.00, 0.0),
    ("전진",  0.15,  0.00, 0.0),
    ("후진", -0.15,  0.00, 0.0),
    ("좌",    0.00,  0.20, 0.0),
    ("우",    0.00, -0.20, 0.0),
    ("회전",  0.00,  0.00, 1.0),
]

obs, _ = env.get_observations()
t_end = time.time() + args_cli.seconds
step = 0
hold_steps = max(1, int(args_cli.hold / dt))
cur_idx = -1
while simulation_app.is_running() and time.time() < t_end:
    t0 = time.time()
    if args_cli.cycle:
        idx = (step // hold_steps) % len(CYCLE)
        if idx != cur_idx:
            cur_idx = idx
            nm, cx, cy, cw = CYCLE[idx]
            print(f"\n[play] ▶ {nm}  cmd=({cx:+.2f}, {cy:+.2f}, {cw:+.2f})", flush=True)
        _, cx, cy, cw = CYCLE[idx]
    else:
        cx, cy, cw = args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw
    u._command[:, 0] = cx
    u._command[:, 1] = cy
    u._command[:, 2] = cw
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    step += 1
    if step % 100 == 0:
        v = u._robot.data.root_lin_vel_b[0, :2]
        w = u._robot.data.root_ang_vel_b[0, 2]
        tag = CYCLE[cur_idx][0] if args_cli.cycle else "고정"
        print(f"[play] {tag:4} vx={v[0]:+.3f} vy={v[1]:+.3f} yaw={w:+.3f}  (cmd {cx:+.2f},{cy:+.2f},{cw:+.2f})", flush=True)
    sleep = dt - (time.time() - t0)
    if sleep > 0:
        time.sleep(sleep)
env.close()
simulation_app.close()
