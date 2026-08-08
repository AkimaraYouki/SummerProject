#!/usr/bin/env python3
"""**관절별 실제 토크**를 잰다 — 정지와 보행 각각, 스톨(4.1 N·m) 대비로.

왜 필요한가. 리워드에 `torques` 항이 있지만 `Episode_Reward/torques` 는 정규화된
값이라 절대 크기를 못 읽는다. 정적 계산(접지 반력 기준)으로는 hip_roll 이
0.79 N·m 로 지배적이라고 나왔는데 로그를 역산한 값과 한 자릿수가 안 맞았다.
모터 부하를 목표로 삼으려면 **어느 관절이 실제로 몇 N·m 를 쓰는지**부터 정확히
알아야 한다. 그게 없으면 리워드 계수를 어디로 움직일지 정할 수 없다.

`applied_torque` 는 액추에이터가 실제로 낸 토크다(ImplicitActuator 의 PD 결과,
effort_limit_sim 4.1 로 잘린 뒤). 스톨 대비 비율이 실기 여유를 그대로 말해준다.

    odm stop
    $ISAACLAB/isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/torque_load.py \
        --task Isaac-OpenDuckMini-Joystick-V37-v0 --checkpoint <...>/model_2999.pt \
        --out /tmp/tq.txt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warm", type=int, default=150)
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from open_duck_mini_isaaclab.agents.rsl_rl_compat import build_runner, load_checkpoint  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for, runner_cfg_for  # noqa: E402

STALL = 4.1  # XM430-W350 @12V

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 1.0e9
env_cfg.events.push_robot = None       # 외란 토크와 자세 유지 토크를 섞지 않는다
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
NAMES = list(ACTUATOR_JOINT_NAMES)

obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)
res = []
for label, vx, vy, wz in (("정지", 0.0, 0.0, 0.0), ("전진", 0.15, 0.0, 0.0)):
    T = []
    for step in range(args_cli.steps):
        u._command[:, :] = 0.0
        u._command[:, 0], u._command[:, 1], u._command[:, 2] = vx, vy, wz
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if step < args_cli.warm:
            continue
        ok = u._robot.data.projected_gravity_b[:, 2] <= -0.5
        t = u._robot.data.applied_torque[:, u._joint_ids]
        T.append(torch.where(ok.unsqueeze(-1), t, torch.full_like(t, float("nan"))))
    res.append((label, torch.cat(T)))

_lines = []
_p = print
def print(*a, **k):          # noqa: A001
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})

print("=" * 86)
print(f"관절 토크 — {args_cli.task}")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ·  {args_cli.num_envs} env × "
      f"{args_cli.steps - args_cli.warm} 스텝  ·  스톨 {STALL} N·m")
for label, T in res:
    rms = torch.sqrt(torch.nanmean(T * T, dim=0))
    mean = torch.nanmean(T, dim=0)
    peak = torch.nan_to_num(T.abs(), nan=0.0).amax(dim=0)
    q99 = torch.nanquantile(T.abs().flatten(0, 0), 0.99, dim=0)
    print("-" * 86)
    print(f"[{label}]  {'관절':16}{'평균':>9}{'RMS':>9}{'99%':>9}{'최대':>9}{'스톨대비':>10}")
    order = torch.argsort(rms, descending=True)
    for i in order.tolist():
        f = float(peak[i]) / STALL * 100.0
        flag = "  <<<" if f > 50 else ""
        print(f"  {'':10}{NAMES[i]:16}{float(mean[i]):+8.3f} {float(rms[i]):8.3f} "
              f"{float(q99[i]):8.3f} {float(peak[i]):8.3f} {f:9.1f}%{flag}")
    tot = float(torch.nansum(rms * rms))
    print(f"  {'':10}{'— 14축 제곱합':16}{tot:8.3f}   (리워드 cost_torques 와 같은 양)")
print("=" * 86)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
