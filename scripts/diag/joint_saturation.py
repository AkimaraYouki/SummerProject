#!/usr/bin/env python3
"""정책이 **URDF 관절 한계를 넘는 목표각을 명령하는지** 잰다.

잿슨측 분석(`docs/reports/joint_saturation_2026-08-09.md`)에서 나온 물음에
답하기 위한 것이다. 실기에서 `left_hip_pitch` 목표가 469 스텝 중 432 스텝(92%)
동안 URDF 한계 70 도를 넘었고(최대 81.2 도), 실기 코드가 그걸 70 도로 잘라내고
있었다. 즉 우리가 "모터 처짐" 으로 본 -7.3 도는 모터가 못 간 거리가 아니라
**우리가 잘라낸 양**이었다.

그래서 시뮬에서도 같은 포화가 일어나는지가 갈림길이다:

* 시뮬도 포화한다면 → sim2real 격차가 아니라 **정책이 도달 불가능한 자세를
  학습**한 것이다. 고칠 곳은 실기가 아니라 리워드/레퍼런스다.
* 시뮬은 여유가 있다면 → 그 차이가 격차의 후보다.

한계 **처리 방식**이 양쪽에서 다르다는 점도 유의한다. 시뮬은 PD 목표가 77 도인
채 PhysX 가 70 도에서 물리적으로 막아 **한계를 계속 밀어붙이는** 상태가 되고,
실기는 goal position 자체를 잘라 보내 **미는 힘 없이 그 자리에 서는** 상태가
된다. 같은 각도라도 접촉력과 전류가 다르다.

    odm stop
    $ISAACLAB/isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/joint_saturation.py \
        --task Isaac-OpenDuckMini-Joystick-V35-v0 --checkpoint <...>/model_2999.pt \
        --out /tmp/sat.txt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warm", type=int, default=150, help="램프인/과도구간 제외")
parser.add_argument("--vx", type=float, default=0.0, help="전진 명령 (기본 0 = 정지)")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from open_duck_mini_isaaclab.agents.rsl_rl_compat import build_runner, load_checkpoint  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for, runner_cfg_for  # noqa: E402

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 1.0e9
env_cfg.events.push_robot = None       # 외란으로 밀려서 넘는 것과 구분한다
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

# USD 가 들고 있는 관절 한계. 실기는 URDF 를 읽지만 시뮬은 USD 에서 읽으므로
# **여기서 USD 값을 그대로 가져와야** 실기와 같은 자를 쓴다.
LIM = u._robot.data.joint_pos_limits[0, u._joint_ids]     # [14,2] (lower, upper)
NAMES = list(ACTUATOR_JOINT_NAMES)

obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)
over_hi, over_lo, tmax, tmin, pmax, n = [], [], None, None, None, 0
for step in range(args_cli.steps):
    # 정지만 보면 보행 중 초과를 놓친다 — 2026-08-12 실기에서 left_hip_pitch 가
    # 보행 중 URDF 상한을 8.7도 넘겨 클램프에 잘리고 있었는데, 이 진단은 정지만
    # 봐서 "초과 0.0 %" 로 통과시켰다.
    u._command[:, :] = 0.0
    u._command[:, 0] = args_cli.vx
    u._command[:, 1] = args_cli.vy
    u._command[:, 2] = args_cli.wz
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    if step < args_cli.warm:
        continue
    # 속도 제한과 고관절 클램프까지 다 지난 **실제로 보낸 목표각**이다.
    t = u._motor_targets
    p = u._robot.data.joint_pos[:, u._joint_ids]
    over_hi.append((t > LIM[:, 1]).float())
    over_lo.append((t < LIM[:, 0]).float())
    tmax = t.amax(0) if tmax is None else torch.maximum(tmax, t.amax(0))
    tmin = t.amin(0) if tmin is None else torch.minimum(tmin, t.amin(0))
    pmax = p.amax(0) if pmax is None else torch.maximum(pmax, p.amax(0))
    n += 1

OH = torch.stack(over_hi).mean(dim=(0, 1))
OL = torch.stack(over_lo).mean(dim=(0, 1))

_lines = []
_p = print
def print(*a, **k):          # noqa: A001
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})

d = math.degrees
print("=" * 92)
print(f"관절 한계 포화 — {args_cli.task}")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ·  {args_cli.num_envs} env × {n} 스텝  ·  정지 명령")
print("  목표각 = _motor_targets (속도제한·고관절클램프까지 적용된, 실제로 보낸 값)")
print("-" * 92)
print(f"  {'관절':16} {'한계 하':>8} {'한계 상':>8} | {'목표 최소':>9} {'목표 최대':>9} "
      f"| {'상한초과':>8} {'하한초과':>8} | {'여유':>7}")
for i, nm in enumerate(NAMES):
    lo, hi = d(float(LIM[i, 0])), d(float(LIM[i, 1]))
    tl, th = d(float(tmin[i])), d(float(tmax[i]))
    margin = min(hi - th, tl - lo)
    flag = "  <<< 포화" if OH[i] + OL[i] > 0.01 else ("  <- 여유 2도 미만" if margin < 2.0 else "")
    print(f"  {nm:16} {lo:+8.1f} {hi:+8.1f} | {tl:+9.1f} {th:+9.1f} "
          f"| {float(OH[i])*100:7.1f}% {float(OL[i])*100:7.1f}% | {margin:+7.1f}{flag}")
print("-" * 92)
print("  '여유' = 목표각이 한계에 가장 가까웠을 때 남은 각도. 음수면 한계를 넘은 것.")
print("  실기 비교값 (v35, 잿슨 로그): left_hip_pitch 상한초과 92.1%, 목표 최대 +81.2도")
print("=" * 92)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
