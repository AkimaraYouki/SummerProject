#!/usr/bin/env python3
"""액션 저역통과 계수(alpha)를 훑어 **떨림이 최소가 되는 값**을 찾는다.

떨림만 재면 답은 항상 alpha->1 (얼어붙음)이다. 그래서 떨림과 **대가**를 같이
잰다:

  떨림  · trunk_gyro_rms  몸통 각속도 RMS  <- 사용자가 말하는 "몸 떨림" 그 자체
        · joint_vel_rms   관절 속도 RMS
        · act_rate        |a[t]-a[t-1]| 평균 (정책 출력의 요동)
  대가  · 정지 속력       정지 명령에서 몸이 실제로 움직인 속도
        · 추종 오차       전진 명령에서 명령 대비 오차
        · 넘어짐          그 구간에서 한 번이라도 종료된 env 비율

alpha 는 매 스텝 cfg 에서 읽으므로 **환경을 다시 만들 필요가 없다** — 한
프로세스에서 전부 훑는다 (Isaac 을 여러 번 띄우면 느리고 불안정하다).

alpha 를 바꾼 직후에는 필터 상태가 이전 값에 물들어 있으므로 구간마다
과도구간을 버린다.

    odm stop
    $ISAACLAB/isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/smooth_sweep.py \
        --task Isaac-OpenDuckMini-Joystick-V35-v0 --checkpoint <...>/model_2999.pt \
        --out /tmp/sweep.txt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=350)
parser.add_argument("--warm", type=int, default=150)
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
env_cfg.events.push_robot = None      # 외란을 켜두면 떨림인지 외력인지 못 가린다
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
FS = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)


def cutoff(a: float) -> float:
    """1차 EMA 의 -3 dB 차단주파수 [Hz]. a=0 이면 필터 없음(= 나이퀴스트)."""
    if a <= 0.0:
        return FS / 2.0
    c = 1.0 - (1.0 - a) ** 2 / (2.0 * a)
    return FS / (2 * math.pi) * math.acos(max(-1.0, min(1.0, c)))


PHASES = [("정지", 0.0, 0.0, 0.0), ("전진", 0.15, 0.0, 0.0)]
rows = []
obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)

for alpha in args_cli.alphas:
    u.cfg.action_lowpass_alpha = float(alpha)
    for label, vx, vy, wz in PHASES:
        fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        gyro, jvel, arate, spd = [], [], [], []
        prev_act = None
        for step in range(args_cli.steps):
            u._command[:, :] = 0.0
            u._command[:, 0], u._command[:, 1], u._command[:, 2] = vx, vy, wz
            with torch.inference_mode():
                act = policy(obs)
                obs, _, dones, _ = env.step(act)
            fell |= dones.bool()
            if prev_act is not None and step >= args_cli.warm:
                arate.append((act - prev_act).abs().mean(dim=-1))
            prev_act = act.clone()
            if step < args_cli.warm:
                continue
            ok = u._robot.data.projected_gravity_b[:, 2] <= -0.5
            nan = lambda t: torch.where(ok, t, torch.full_like(t, float("nan")))
            gyro.append(nan(torch.linalg.norm(u._robot.data.root_ang_vel_b, dim=-1)))
            jvel.append(nan(torch.linalg.norm(u._robot.data.joint_vel[:, u._joint_ids], dim=-1)))
            v = u._robot.data.root_lin_vel_b
            tgt = torch.tensor([vx, vy], device=u.device)
            spd.append(nan(torch.linalg.norm(v[:, :2] - tgt, dim=-1)))
        nm = lambda L: float(torch.nanmean(torch.stack(L)))
        rows.append((alpha, label, nm(gyro), nm(jvel), nm(arate), nm(spd),
                     float(fell.float().mean()) * 100.0))

_lines = []
_p = print
def print(*a, **k):          # noqa: A001
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})

print("=" * 88)
print(f"액션 저역통과 스윕 — {args_cli.task}")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ·  {args_cli.num_envs} env × "
      f"{args_cli.steps - args_cli.warm} 스텝/조건  ·  제어 {FS:.0f} Hz")
print("  EMA 는 brick-wall 이 아니라 -20 dB/decade 완만한 감쇠 + 위상 지연이다.")
print("  군지연 = a/(1-a) 샘플. 보행 주기는 0.54 s (1.85 Hz).")
for ph in ("정지", "전진"):
    print("-" * 88)
    print(f"[{ph}]")
    print(f"  {'alpha':>6} {'차단Hz':>7} {'지연ms':>7} | {'몸통각속도':>10} {'관절속도':>9} "
          f"{'액션요동':>9} | {'속도오차':>9} {'넘어짐':>7}")
    base = None
    for a, label, g, jv, ar, sp, fl in rows:
        if label != ph:
            continue
        if base is None:
            base = (g, jv, ar)
        d = a / (1.0 - a) / FS * 1000.0 if a < 1.0 else float("inf")
        print(f"  {a:6.2f} {cutoff(a):7.1f} {d:7.1f} | {g:10.4f} {jv:9.3f} {ar:9.4f} "
              f"| {sp:9.4f} {fl:6.1f}%")
    print(f"  (alpha 0 대비 몸통각속도 감소율)")
    line = "   "
    for a, label, g, jv, ar, sp, fl in rows:
        if label != ph:
            continue
        line += f"  a{a:.2f}:{(1 - g / base[0]) * 100:+5.1f}%"
    print(line)
print("=" * 88)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
