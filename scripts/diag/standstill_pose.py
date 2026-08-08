#!/usr/bin/env python3
"""정지 명령에서 로봇이 실제로 어떤 자세로 서는지 잰다 — 특히 **몸통 피치**.

    odm stop
    ISAACLAB=~/Desktop/IsaacLab
    $ISAACLAB/isaaclab.sh -p scripts/_isaaclab_launch.py scripts/diag/standstill_pose.py \
        --task Isaac-OpenDuckMini-Joystick-V34U-v0 \
        --checkpoint <...>/model_1500.pt --headless

`gait_compare.py` 는 속도만 남기고 자세는 안 남긴다. 그런데 "정지하면 가만히
있는가" 와 "정지했을 때 몸통이 곧게 서는가" 는 다른 질문이다 — 속도가 0 이어도
4 도 기운 채로 멈춰 있을 수 있다 (실제로 v34c10 이 그랬다).

몸통 피치의 출처: placo 프리셋의 walk_trunk_pitch = -4 도가 레퍼런스 관절각에
배어 있고, 그 평균 자세가 곧 `cost_stand_still` 의 목표였다. 발바닥이 지면에
눕도록 물리가 몸통을 돌리므로 서 있을 때 그만큼 기운다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=400)
parser.add_argument("--warm", type=int, default=150, help="과도구간 제외 스텝")
# Isaac Sim 아래서는 stdout 이 버퍼에 남아 그냥 사라진다 (여러 번 겪었다).
# 결과는 파일로도 쓴다 — 이게 실제로 읽히는 경로다.
parser.add_argument("--out", type=str, default="", help="결과를 이 파일에도 쓴다")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

from open_duck_mini_isaaclab.agents.rsl_rl_compat import build_runner, load_checkpoint  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for, runner_cfg_for  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 1.0e9      # 정지만 볼 것이므로 시간초과 리셋을 끈다
env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
# rsl-rl 러너는 VecEnv 규약(get_observations 등)을 요구한다. 맨 gym.make 결과를
# 넘기면 OrderEnforcing 래퍼에 막힌다 — gait_compare.py 와 같은 순서로 감싼다.
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

obs, _ = env.get_observations() if isinstance(env.get_observations(), tuple) else (env.get_observations(), None)
pitch_hist, roll_hist, h_hist, fallen_hist = [], [], [], []
for step in range(args_cli.num_steps):
    u._command[:, :] = 0.0            # 정지 명령 (머리 명령 포함 전부 0)
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _ = env.step(act)
    if step < args_cli.warm:
        continue
    # 오일러 분해 대신 projected_gravity 를 쓴다. 직립이면 정확히 (0,0,-1) 이고,
    # euler_xyz_from_quat 처럼 요각과 얽혀 튀지 않는다 (처음에 그걸 썼다가
    # 평균 +19 도에 범위 -73~+87 이라는 못 믿을 값이 나왔다).
    #   projected_gravity_b = (sin(pitch), -sin(roll)cos(pitch), -cos(pitch)cos(roll))
    g3 = u._robot.data.projected_gravity_b
    pitch_hist.append(torch.atan2(g3[:, 0], -g3[:, 2]))
    roll_hist.append(torch.atan2(-g3[:, 1], -g3[:, 2]))
    # 넘어진 환경(중력 z 성분이 위를 향함)은 자세 통계에서 뺀다.
    fallen_hist.append(g3[:, 2] > -0.5)
    h_hist.append(u._robot.data.root_pos_w[:, 2])

P = torch.stack(pitch_hist); R = torch.stack(roll_hist); H = torch.stack(h_hist)
F = torch.stack(fallen_hist)
keep = ~F
if keep.sum() == 0:
    raise SystemExit("전 환경이 넘어졌다 — 자세를 잴 것이 없다")
P, R, H = P[keep], R[keep], H[keep]
d = lambda t: math.degrees(float(t))
_lines = []
_p = print
def print(*a, **k):          # noqa: A001 — 화면과 파일에 동시에
    _lines.append(" ".join(str(x) for x in a))
    _p(*a, **{**k, "flush": True})
print("=" * 62)
print(f"정지 자세 — {args_cli.task}")
print(f"  {args_cli.checkpoint.split('/')[-1]}  ({args_cli.num_envs} env × "
      f"{args_cli.num_steps - args_cli.warm} 스텝, 과도구간 {args_cli.warm} 제외)")
print("-" * 62)
print(f"  몸통 피치   평균 {d(P.mean()):+7.2f}°   표준편차 {d(P.std()):5.2f}°   "
      f"범위 {d(P.min()):+.2f} ~ {d(P.max()):+.2f}")
print(f"  몸통 롤     평균 {d(R.mean()):+7.2f}°   표준편차 {d(R.std()):5.2f}°")
print(f"  몸통 높이   평균 {float(H.mean())*1000:7.1f} mm  표준편차 {float(H.std())*1000:.1f} mm")
print(f"  넘어진 비율 {float(F.float().mean())*100:5.1f} %  (자세 통계에서 제외)")
print("-" * 62)
# 부호: projected_gravity_b 의 x 성분이 sin(pitch) 다. 몸통이 앞으로 숙으면
# 몸통의 +x(전방)축이 아래를 향하게 되어 중력의 x 성분이 **양수**가 된다.
# (한동안 이 줄이 반대로 적혀 있었다 — v34c10 의 +8.19 도를 "뒤로 젖힘" 으로
#  읽으면 정반대의 결론이 나온다.)
print("  피치 0 = 수직.  양수 = 앞으로 숙임, 음수 = 뒤로 젖힘")
print("=" * 62)

if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write("\n".join(_lines) + "\n")

env.close()
simulation_app.close()
