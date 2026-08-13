#!/usr/bin/env python3
"""정지 중인 로봇을 **밀어보고 얼마나 잘 버티는지** 잰다.

학습에는 `push_robot` 외란이 5~10 초마다 ±1.0 m/s 로 들어가 있지만, **그 결과를
재는 지표가 없었다.** "외란을 잘 견딘다" 를 숫자로 만들지 않으면 개선했는지
나빠졌는지 알 수 없어서, 학습 방향 세 가지 중 하나(외란 극복)를 판정할 수 없다.

방식은 학습과 같은 메커니즘이다 — 몸통의 선속도를 순간적으로 바꾼다
(`push_by_setting_velocity` 와 동일). 세기를 여러 단계로 훑어 **어디서 무너지는지**
를 본다. 방향은 앞/뒤/좌/우 네 가지를 균등하게 섞는다.

재는 것:
  넘어짐      그 세기에서 종료된 env 비율 — 가장 중요한 수치
  최대 기울기 밀린 직후 몸통이 가장 많이 기운 각도 (projected_gravity 기준)
  회복 시간   기울기가 다시 3 도 아래로 돌아오기까지 (못 돌아오면 미회복)
  잔류 이동   회복 후 원래 자리에서 얼마나 밀려나 있는지

    odm stop
    $ISAACLAB/isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/disturbance.py \
        --task Isaac-OpenDuckMini-Joystick-V36-v0 --checkpoint <...>/model_2999.pt \
        --out /tmp/dist.txt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--pushes", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5],
                    help="가할 속도 크기 [m/s]. 학습 외란은 ±1.0")
parser.add_argument("--settle", type=int, default=150, help="밀기 전 정착 스텝")
parser.add_argument("--watch", type=int, default=150, help="밀고 나서 관찰할 스텝 (3 초)")
parser.add_argument("--recover_deg", type=float, default=3.0, help="이 각도 아래로 오면 회복")
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

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 1.0e9
# 학습용 무작위 외란은 끈다 — 우리가 가하는 것만 보이게 해야 세기별로 갈린다.
env_cfg.events.push_robot = None
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
dev = u.device
N = u.num_envs

# env 를 4등분해 앞/뒤/좌/우로 민다. 한 방향만 보면 좌우 비대칭에 속는다.
DIRS = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], device=dev)
dir_of = DIRS[torch.arange(N, device=dev) % 4]           # [N,2]


def tilt_deg():
    g = u._robot.data.projected_gravity_b
    return torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1.0, 1.0)))


def run_steps(n, obs):
    for _ in range(n):
        u._command[:, :] = 0.0
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
    return obs


obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)
rows = []
for mag in args_cli.pushes:
    obs = run_steps(args_cli.settle, obs)
    p0 = u._robot.data.root_pos_w[:, :2].clone()

    # 밀기 — 학습의 push_by_setting_velocity 와 같은 방식으로 선속도를 덮어쓴다.
    v = u._robot.data.root_lin_vel_w.clone()
    v[:, :2] = dir_of * mag
    u._robot.write_root_velocity_to_sim(torch.cat([v, u._robot.data.root_ang_vel_w], dim=-1))

    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    peak = torch.zeros(N, device=dev)
    rec_at = torch.full((N,), -1.0, device=dev)
    for k in range(args_cli.watch):
        u._command[:, :] = 0.0
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        fell |= dones.bool()
        t = tilt_deg()
        peak = torch.maximum(peak, torch.where(fell, torch.zeros_like(t), t))
        # 한 번 밀린 뒤(20 스텝 지나) 처음으로 기준 아래로 내려온 시점을 회복으로 본다.
        newly = (rec_at < 0) & (t < args_cli.recover_deg) & (~fell) & (k > 20)
        rec_at = torch.where(newly, torch.full_like(rec_at, float(k)), rec_at)
    ok = ~fell
    drift = torch.linalg.norm(u._robot.data.root_pos_w[:, :2] - p0, dim=-1)
    got = ok & (rec_at >= 0)
    rows.append((
        mag,
        float(fell.float().mean()) * 100.0,
        float(peak[ok].mean()) if ok.any() else float("nan"),
        float(rec_at[got].mean()) * (u.cfg.sim.dt * u.cfg.decimation) * 1000.0 if got.any() else float("nan"),
        float(got.float().sum()) / max(float(ok.float().sum()), 1.0) * 100.0,
        float(drift[ok].mean()) * 1000.0 if ok.any() else float("nan"),
    ))

_lines = []
_p = print
def print(*a, **k):          # noqa: A001
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})

print("=" * 84)
print(f"외란 극복 — {args_cli.task}")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ·  {N} env (앞/뒤/좌/우 균등)  ·  "
      f"밀고 {args_cli.watch} 스텝({args_cli.watch*0.02:.1f} s) 관찰")
print(f"  학습 외란은 5~10 초마다 ±1.0 m/s 다 — 그 값 근처가 기준선.")
print("-" * 84)
print(f"  {'세기 m/s':>9} | {'넘어짐':>7} {'최대기울기':>10} {'회복시간':>9} {'회복률':>7} {'잔류이동':>9}")
for mag, fl, pk, rt, rr, dr in rows:
    print(f"  {mag:9.2f} | {fl:6.1f}% {pk:9.2f}° "
          f"{('  --' if rt != rt else f'{rt:8.0f}ms')} {rr:6.1f}% {dr:8.1f}mm")
print("-" * 84)
print(f"  회복 = 기울기가 {args_cli.recover_deg:.0f}도 아래로 복귀. 회복률은 안 넘어진 env 중 비율.")
print("  잔류이동 = 밀린 뒤 3 초 시점에 원래 자리에서 떨어진 거리.")
print("=" * 84)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
