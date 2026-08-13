"""새 리워드 항의 **원시 크기**를 재고, 목표 비중을 내는 계수를 역산한다.

    ./scripts/odm probe v53          # 래퍼가 있으면
    $IL -p scripts/_isaaclab_launch.py scripts/diag/probe_terms.py \
        --task Isaac-OpenDuckMini-Joystick-V53-v0 --checkpoint <ck>

## 왜 이게 필요한가

2026-08-14, `cost_foot_lift` 를 계수 -300 으로 넣었다가 v53 이 무너졌다
(리워드 318.8 -> 100.4). 원인은 두 가지였는데 둘 다 **재지 않고 추정**해서
생겼다:

  1. 발 링크 원점의 좌우 z 차이(39 mm)를 몰라 정지 상태에서도 -0.2415 가
     상시로 깎였다 — 리워드 예산의 45 %.
  2. 계수를 몸통 기준 위치의 미분으로 어림잡았는데 실제와 37 배 어긋났다.

그래서 규칙을 세웠다: **리워드 항을 새로 넣으면 학습 전에 원시값을 재고,
특히 정지 명령에서 0 인지 본다.** 이 스크립트가 그 절차다.

## 무엇을 출력하는가

명령 6종 각각에 대해 항의 원시값(계수 1.0, dt 곱하기 전)을 재고,

    계수 = 목표비중 x 스텝당_총리워드 / (원시값 x dt)

로 역산해 준다. 정지 열이 0 이 아니면 그 항은 **상시 세금**이므로 경고한다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps_per_cmd", type=int, default=200)
#: 각 항이 스텝당 총 리워드에서 차지했으면 하는 비중 (%).
parser.add_argument("--target-share", type=float, default=3.0)
#: Isaac Kit 이 부팅 뒤 stdout 을 가로채므로 결과는 파일로도 쓴다.
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_compat import build_runner, load_checkpoint  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import (  # noqa: E402
    env_cfg_for,
    runner_cfg_for,
)
from open_duck_mini_isaaclab.tasks.velocity.rewards import (  # noqa: E402
    cost_foot_clearance,
    cost_foot_lateral,
    cost_foot_slip,
    cost_joint_accel,
    cost_torso_ang_vel,
)

COMMANDS = [
    ("stop", 0.00, 0.00, 0.0),
    ("fwd", 0.15, 0.00, 0.0),
    ("back", -0.15, 0.00, 0.0),
    ("left", 0.00, 0.20, 0.0),
    ("right", 0.00, -0.20, 0.0),
    ("turn", 0.00, 0.00, 1.0),
]

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.events.push_robot = None

env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
cfg = u.cfg


def raw_terms():
    """계수와 dt 를 빼고 순수 비용값만."""
    fpos = u._robot.data.body_pos_w[:, u._feet_ids]
    fvel = u._feet_vel_b()
    off = u._foot_z_offset()
    con = u._get_foot_contact()
    return {
        "foot_clearance": cost_foot_clearance(fpos, fvel, off, cfg.foot_clearance_target),
        "foot_lateral": cost_foot_lateral(fvel),
        "foot_slip": cost_foot_slip(fvel, con),
        "torso_ang_vel": cost_torso_ang_vel(u._robot.data.root_ang_vel_b),
        "joint_accel": cost_joint_accel(u._robot.data.joint_acc[:, u._joint_ids]),
    }


per_cmd, reward_per_cmd = {}, {}
obs = env.get_observations()
for name, cx, cy, cw in COMMANDS:
    acc, rew = {}, []
    for step in range(args_cli.steps_per_cmd):
        u._command[:, 0], u._command[:, 1], u._command[:, 2] = cx, cy, cw
        with torch.inference_mode():
            obs, r, _, _ = env.step(policy(obs))
        if step < args_cli.steps_per_cmd // 2:
            continue
        for k, v in raw_terms().items():
            acc.setdefault(k, []).append(float(v.mean()))
        rew.append(float(r.mean()))
    per_cmd[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    reward_per_cmd[name] = float(np.mean(rew))

_lines = []


def emit(line=""):
    """Kit 이 stdout 을 가로채므로 모아 뒀다가 파일로도 쓴다."""
    _lines.append(line)
    print(line, flush=True)


dt = u.step_dt
names = list(per_cmd["fwd"].keys())
tot = float(np.mean(list(reward_per_cmd.values())))

emit("")
emit("=" * 86)
emit(f"새 리워드 항 원시값 (계수 1.0, dt {dt:.4f} 곱하기 전) — {args_cli.task}")
emit("=" * 86)
hdr = f"{'항목':<18}" + "".join(f"{c:>10}" for c, *_ in COMMANDS) + f"{'평균':>11}"
emit(hdr)
emit("-" * len(hdr))
for n in names:
    vals = [per_cmd[c][n] for c, *_ in COMMANDS]
    emit(f"{n:<18}" + "".join(f"{v:>10.4f}" for v in vals) + f"{np.mean(vals):>11.4f}")

emit("-" * len(hdr))
emit(f"스텝당 총 리워드 {tot:.4f}  (정지 {reward_per_cmd['stop']:.4f} · "
     f"전진 {reward_per_cmd['fwd']:.4f})")

emit("")
emit("=" * 86)
emit(f"목표 비중 {args_cli.target_share:.1f} % 를 내는 계수")
emit("=" * 86)
emit(f"  {'항목':<18}{'권장계수':>14}{'정지 기여':>14}   비고")
emit("  " + "-" * 70)
for n in names:
    mean = float(np.mean([per_cmd[c][n] for c, *_ in COMMANDS]))
    if mean <= 1e-12:
        emit(f"  {n:<18}{'—':>14}{'—':>14}   원시값이 0 — 항이 작동하지 않는다")
        continue
    scale = args_cli.target_share / 100.0 * tot / (mean * dt)
    stop_c = per_cmd["stop"][n] * dt * scale
    note = ""
    if abs(stop_c) > 0.02 * tot:
        note = f"⚠ 정지에서 총리워드의 {100*abs(stop_c)/tot:.1f} % 를 상시로 깎는다"
    elif abs(stop_c) < 1e-4:
        note = "정지 0 — 좋다"
    emit(f"  {n:<18}{-scale:>14.4g}{-stop_c:>14.4f}   {note}")
emit("")
emit("  계수는 음수(벌점)로 쓴다. 위 값은 부호까지 붙여 놓았다.")
emit("=" * 86)
if args_cli.out:
    with open(args_cli.out, "w") as fh:
        fh.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
