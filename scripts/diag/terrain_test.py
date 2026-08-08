#!/usr/bin/env python3
"""학습된 정책을 **평지가 아닌 지형**에 올려놓고 버티는지 잰다 (제로샷).

배경: 이 프로젝트의 모든 학습은 `terrain_type="plane"`, 즉 완벽한 평면에서만
돌았다. 게다가 관측에 지형을 보는 수단이 하나도 없다 — height scan 도 ray caster
도 없다. 즉 **완전한 blind 정책**이다. 지형을 못 보니 발밑이 달라져도 정책은
자기가 평지에 있다고 믿고, 오직 IMU 와 관절 되먹임으로만 반응한다.

그래서 "울퉁불퉁한 데서 되나" 는 추측이 아니라 재봐야 아는 질문이다. 마찰
(0.5~1.0) · 질량 · 액추에이터 게인 · 외력 push 랜덤화는 이미 켜져 있어서 수 mm
급 요철은 외란처럼 흡수할 여지가 있다.

**스케일 주의.** IsaacLab 의 `ROUGH_TERRAINS_CFG` 기본값은 계단 5~23 cm, 요철
2~10 cm 다. 그건 ANYmal(선 키 ~55 cm) 기준이고, 우리 로봇은 서 있는 높이가
125~140 mm, 발 들어올림이 4 cm 다. **23 cm 계단은 로봇 키보다 높다.** 그래서
같은 지형을 IsaacLab 기본값과 로봇 스케일(대략 1/4) 두 벌로 준비했다 —
기본값에서 넘어지는 건 정책 탓이 아니라 지형이 로봇에 안 맞는 것이다.

    python3 scripts/diag/terrain_test.py --task <task> --checkpoint <pt> \
        --terrain rough_s --out /tmp/t.txt --headless

지형 하나당 프로세스 하나로 돌린다 (한 프로세스에서 Isaac 환경을 여러 번
만들었다 부수면 불안정하다). 목록은 `--list` 로 볼 수 있다.
"""

import argparse

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "source"))
from open_duck_mini_isaaclab.terrains import TERRAIN_CHOICES  # noqa: E402

TERRAINS = list(TERRAIN_CHOICES.items())

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--list", action="store_true")
if _ap.parse_known_args()[0].list:
    print(f"{'이름':14} 설명")
    for n, d in TERRAINS:
        print(f"  {n:14} {d}")
    raise SystemExit(0)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--terrain", type=str, required=True, choices=[n for n, _ in TERRAINS])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps_per_phase", type=int, default=300)
parser.add_argument("--warm", type=int, default=100, help="각 구간에서 버릴 과도구간")
parser.add_argument("--list", action="store_true")
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
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for, runner_cfg_for  # noqa: E402
from open_duck_mini_isaaclab.terrains import apply_terrain  # noqa: E402

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 1.0e9        # 시간초과 리셋은 끄고, 넘어짐만 센다

apply_terrain(env_cfg, args_cli.terrain)

env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

PHASES = [("정지", 0.0, 0.0, 0.0), ("전진", 0.15, 0.0, 0.0), ("회전", 0.0, 0.0, 1.0)]
rows = []
obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)

for label, vx, vy, wz in PHASES:
    fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    sp, pi, hs = [], [], []
    for step in range(args_cli.steps_per_phase):
        u._command[:, :] = 0.0
        u._command[:, 0], u._command[:, 1], u._command[:, 2] = vx, vy, wz
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        # 넘어짐은 "한 번이라도" 로 센다. 환경이 리셋해 버리므로 순간값만 보면
        # 놓친다 — 이게 지형 평가에서 제일 중요한 수치다.
        fell |= dones.bool()
        if step < args_cli.warm:
            continue
        g3 = u._robot.data.projected_gravity_b
        ok = g3[:, 2] <= -0.5
        v = u._robot.data.root_lin_vel_b
        sp.append(torch.where(ok, torch.linalg.norm(v[:, :2] - torch.tensor([vx, vy], device=u.device), dim=-1),
                              torch.full_like(v[:, 0], float("nan"))))
        pi.append(torch.where(ok, torch.atan2(g3[:, 0], -g3[:, 2]), torch.full_like(g3[:, 0], float("nan"))))
        hs.append(u._robot.data.root_pos_w[:, 2])
    S = torch.stack(sp); P = torch.stack(pi)
    nan = lambda t: float(torch.nanmean(t))
    rows.append((label, float(fell.float().mean()) * 100.0, nan(S), math.degrees(nan(P))))

_lines = []
_p = print
def print(*a, **k):          # noqa: A001
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})

desc = dict(TERRAINS)[args_cli.terrain]
print("=" * 70)
print(f"지형 제로샷 — {args_cli.terrain}  ({desc})")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ·  {args_cli.num_envs} env × "
      f"{args_cli.steps_per_phase} 스텝/구간 (과도 {args_cli.warm} 제외)")
print("-" * 70)
print(f"  {'구간':6} {'넘어짐':>8} {'속도오차':>10} {'몸통피치':>10}")
for label, f, s, p in rows:
    print(f"  {label:6} {f:7.1f}% {s:10.4f} {p:+9.2f}°")
print("-" * 70)
print("  넘어짐 = 그 구간에서 한 번이라도 종료된 env 비율. 피치 + 는 앞으로 숙임.")
print("=" * 70)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
