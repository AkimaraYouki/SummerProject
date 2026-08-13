#!/usr/bin/env python3
"""정책이 **실제로 내는** 좌우 다리 대칭을 잰다 — 정지와 보행 각각.

기본 자세(`READY_JOINT_POS_*`)가 비대칭이라는 건 파일만 봐도 안다. 하지만
"정지/보행 중에 오른다리가 더 벌어진다" 는 정책이 실행 중에 만드는 결과이고,
그건 돌려봐야 안다. 여기서 재는 것은 두 가지다:

재는 것은 **관절각뿐**이다. 발 위치도 재봤다가 뺐다 — `foot_assembly` /
`foot_assembly_2` 링크 원점이 좌우 메시 안에서 서로 다른 자리에 박혀 있어서,
관절각을 완벽히 대칭으로 넣어도 y 가 38.6 mm 어긋난다 (진짜 발 프레임인
`left_foot`/`right_foot` 로 재면 0.5 mm). 링크 원점으로 "발이 더 벌어졌다" 를
판정하면 없는 비대칭을 보게 된다. 발 위치가 정말 필요하면 `left_foot`/
`right_foot` 프레임을 쓸 것.

거울 규칙은 추측하지 않는다. URDF 에서 FK 로 전수 탐색해 얻은 것이고
(`right = (-hip_yaw, +hip_roll, +hip_pitch, -knee, -ankle) * left`), 그 규칙에서
URDF 자체는 대칭이다 (발 위치 거울오차 0.5 mm, 좌우 링크 질량 동일). 따라서
여기서 나오는 비대칭은 전부 **레퍼런스 또는 정책** 탓이다.

    odm stop
    $ISAACLAB/isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/leg_symmetry.py \
        --task Isaac-OpenDuckMini-Joystick-V35-v0 --checkpoint <...>/model_2999.pt \
        --out /tmp/sym.txt --headless
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

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from open_duck_mini_isaaclab.agents.rsl_rl_compat import build_runner, load_checkpoint  # noqa: E402
from open_duck_mini_isaaclab.joint_order import ACTUATOR_JOINT_NAMES  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for, runner_cfg_for  # noqa: E402

# FK 전수 탐색으로 확정한 거울 부호 (scripts/diag 주석 및 커밋 2c0b12a 참고)
MIRROR_SIGN = {"hip_yaw": -1.0, "hip_roll": +1.0, "hip_pitch": +1.0, "knee": -1.0, "ankle": -1.0}

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 1.0e9
env_cfg.events.push_robot = None          # 외란은 끈다 — 좌우차가 외력 때문이면 못 읽는다
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

NAMES = list(ACTUATOR_JOINT_NAMES)
LI = {j: NAMES.index("left_" + j) for j in MIRROR_SIGN}
RI = {j: NAMES.index("right_" + j) for j in MIRROR_SIGN}


PHASES = [("정지", 0.0, 0.0, 0.0), ("전진", 0.15, 0.0, 0.0)]
results = []
obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)

for label, vx, vy, wz in PHASES:
    qs = []
    for step in range(args_cli.steps):
        u._command[:, :] = 0.0
        u._command[:, 0], u._command[:, 1], u._command[:, 2] = vx, vy, wz
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if step < args_cli.warm:
            continue
        ok = u._robot.data.projected_gravity_b[:, 2] <= -0.5      # 넘어진 env 제외
        q = u._robot.data.joint_pos[:, u._joint_ids]
        qs.append(torch.where(ok.unsqueeze(-1), q, torch.full_like(q, float("nan"))))
    results.append((label, torch.cat(qs)))

_lines = []
_p = print
def print(*a, **k):          # noqa: A001
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})

nm = lambda t: float(torch.nanmean(t))
print("=" * 74)
print(f"좌우 다리 대칭 — {args_cli.task}")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ·  {args_cli.num_envs} env × "
      f"{args_cli.steps - args_cli.warm} 스텝 (과도 {args_cli.warm} 제외)")
for label, Q in results:
    print("-" * 74)
    print(f"[{label}]  관절각 — 거울 규칙 적용 후 좌우차")
    print(f"  {'관절':10} {'좌':>9} {'우':>9} {'거울기대':>9} {'어긋남':>9}")
    for j, sg in MIRROR_SIGN.items():
        L, R_ = nm(Q[:, LI[j]]), nm(Q[:, RI[j]])
        exp = sg * L
        d = math.degrees(R_ - exp)
        print(f"  {j:10} {math.degrees(L):+8.2f}° {math.degrees(R_):+8.2f}° "
              f"{math.degrees(exp):+8.2f}° {d:+8.2f}°{'  <<<' if abs(d) > 1.0 else ''}")
    # env 별 시간평균을 낸 뒤 그 분포를 본다. 표준편차가 크면 "평균이 12도 어긋남"
    # 이 아니라 "환경마다 다른 자세로 수렴" 이라는 뜻이라 해석이 완전히 달라진다.
    for j, sg in MIRROR_SIGN.items():
        per_env = Q[:, RI[j]] - sg * Q[:, LI[j]]
        pe = torch.nanmean(per_env.reshape(-1, u.num_envs), dim=0)
        pe = pe[~torch.isnan(pe)]
        print(f"    {j:10} env별 어긋남  평균 {math.degrees(float(pe.mean())):+7.2f}°  "
              f"표준편차 {math.degrees(float(pe.std())):6.2f}°  "
              f"범위 {math.degrees(float(pe.min())):+7.2f} ~ {math.degrees(float(pe.max())):+7.2f}")
print("=" * 74)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
