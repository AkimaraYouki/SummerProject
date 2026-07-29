"""고관절 roll/yaw 이탈이 몸통을 붙잡는 보상 동작인가, 독립적인 결함인가.

왜 이걸 재는가. v25 측정에서 다리-몸통 접촉이 레퍼런스 0% 대 정책 56.7%였고,
원인은 고관절 roll/yaw가 레퍼런스에서 8~9° 벗어나 다리를 안쪽으로 도는 것으로
특정됐다(무릎·발목은 거의 무관). 그런데 고관절 roll/yaw는 이족보행에서 몸통을
지지발 위에 세워두는 바로 그 관절이다. 그래서 두 가지 해석이 가능하다:

  (A) 보상 동작   몸통이 흔들려서 고관절이 그걸 붙잡고 있다.
                  -> 몸통을 먼저 안정화해야 한다. 고관절 범위를 먼저 조이면
                     균형 권한을 뺏는 셈이라 더 휘청이거나 넘어진다.
  (B) 독립 결함   몸통과 무관하게 그냥 레퍼런스를 못 따라간다.
                  -> 고관절 roll/yaw 액션 범위를 레퍼런스 범위로 제한하면
                     기구적으로 접촉이 불가능해진다. 싸고 안전하다.

둘은 상관계수로 갈린다. 고관절 roll/yaw 이탈이 몸통 roll/pitch(또는 그 각속도)와
같이 움직이면 (A), 따로 놀면 (B)다.

**해석 주의**: 상관은 인과가 아니다. 여기서 높은 상관이 나와도 "몸통이 원인"이
확정되는 것은 아니고, 몸통과 고관절이 함께 흔들리는 하나의 진동 모드일 수도
있다. 다만 상관이 **낮으면** (A)는 확실히 배제된다 — 그쪽이 이 측정의 진짜 힘이다.

접촉률 자체는 여기서 재지 않는다. 그건 pinocchio 정확 메시가 필요한 별도 작업이고
(링크 원점 거리는 못 쓴다 — 고정 형상이 지배해 실제 간격과 상관 -0.333),
인과 판별에는 필요 없다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps_per_cmd", type=int, default=250)
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.agents.rsl_rl_compat import build_runner, load_checkpoint  # noqa: E402
from open_duck_mini_isaaclab.tasks.task_registry import (  # noqa: E402
    env_cfg_for,
    runner_cfg_for,
)
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX  # noqa: E402


# 다리 10관절 부분집합 안에서의 자리 (joint_order.py 의 ACT_LEG_JOINT_IDX 순서):
#   0 L_hip_yaw  1 L_hip_roll  2 L_hip_pitch  3 L_knee  4 L_ankle
#   5 R_hip_yaw  6 R_hip_roll  7 R_hip_pitch  8 R_knee  9 R_ankle
L_YAW, L_ROLL, R_YAW, R_ROLL = 0, 1, 5, 6
KNEE_ANKLE = [3, 4, 8, 9]

# gait_compare.py 와 같은 조건이라 수치를 나란히 놓고 볼 수 있다.
COMMANDS = [
    ("stop", 0.00, 0.00, 0.0),
    ("forward", 0.15, 0.00, 0.0),
    ("backward", -0.15, 0.00, 0.0),
    ("left", 0.00, 0.20, 0.0),
    ("right", 0.00, -0.20, 0.0),
    ("turn", 0.00, 0.00, 1.0),
]

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
# 외란이 있으면 "몸통이 흔들려서 고관절이 반응했다"가 정책 탓인지 외력 탓인지
# 구분되지 않는다. 여기서는 정책만 보고 싶으므로 끈다.
env_cfg.events.push_robot = None

env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

rec = {k: [] for k in ("cmd", "hip_roll_dev", "hip_yaw_dev", "knee_ankle_dev",
                       "torso_roll", "torso_pitch", "torso_roll_rate", "torso_pitch_rate")}

obs = env.get_observations()
for ci, (name, cx, cy, cw) in enumerate(COMMANDS):
    for step in range(args_cli.steps_per_cmd):
        u._command[:, 0] = cx
        u._command[:, 1] = cy
        u._command[:, 2] = cw
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))

        # 앞쪽 절반은 명령이 바뀐 직후의 과도구간이라 버린다.
        if step < args_cli.steps_per_cmd // 2:
            continue

        jp = u._robot.data.joint_pos[:, u._joint_ids][:, ACT_LEG_JOINT_IDX]
        ref = u._current_reference_motion[:, 0:14][:, REF_LEG_JOINT_IDX]
        dev = jp - ref  # [N, 10] 레퍼런스 대비 이탈 (rad)

        roll, pitch, _ = math_utils.euler_xyz_from_quat(u._robot.data.root_quat_w)
        # euler_xyz_from_quat 은 [0, 2pi) 로 감아서 돌려준다. 그대로 쓰면
        # -1° 가 359° 가 되어 상관이 완전히 망가진다. [-pi, pi) 로 편다.
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        ang = u._robot.data.root_ang_vel_b

        rec["cmd"].append(np.full(u.num_envs, ci))
        # 좌우를 부호까지 합쳐 하나로 본다. 양다리가 같은 방향으로 벌어지는 것과
        # 서로 반대로 도는 것은 몸통에 주는 영향이 다르다 -- 평균이 그걸 담는다.
        rec["hip_roll_dev"].append(((dev[:, L_ROLL] + dev[:, R_ROLL]) * 0.5).cpu().numpy())
        rec["hip_yaw_dev"].append(((dev[:, L_YAW] + dev[:, R_YAW]) * 0.5).cpu().numpy())
        rec["knee_ankle_dev"].append(dev[:, KNEE_ANKLE].abs().mean(dim=-1).cpu().numpy())
        rec["torso_roll"].append(roll.cpu().numpy())
        rec["torso_pitch"].append(pitch.cpu().numpy())
        rec["torso_roll_rate"].append(ang[:, 0].cpu().numpy())
        rec["torso_pitch_rate"].append(ang[:, 1].cpu().numpy())

data = {k: np.concatenate(v) for k, v in rec.items()}
DEG = 180.0 / np.pi


def corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


print("\n" + "=" * 68)
print("고관절 roll/yaw 이탈  vs  몸통 자세")
print("=" * 68)
print(f"표본 {len(data['cmd'])} (env-step), 명령 {len(COMMANDS)}종, 과도구간 제외")

print("\n[크기]")
print(f"  고관절 roll 이탈 |평균| : {np.abs(data['hip_roll_dev']).mean()*DEG:6.2f} deg")
print(f"  고관절 yaw  이탈 |평균| : {np.abs(data['hip_yaw_dev']).mean()*DEG:6.2f} deg")
print(f"  무릎/발목 이탈  |평균| : {data['knee_ankle_dev'].mean()*DEG:6.2f} deg  (비교용)")
print(f"  몸통 roll  RMS         : {np.sqrt((data['torso_roll']**2).mean())*DEG:6.2f} deg")
print(f"  몸통 pitch RMS         : {np.sqrt((data['torso_pitch']**2).mean())*DEG:6.2f} deg")

print("\n[상관] 고관절 이탈 vs 몸통")
pairs = [
    ("hip_roll_dev", "torso_roll", "고관절roll 이탈 ~ 몸통 roll"),
    ("hip_roll_dev", "torso_roll_rate", "고관절roll 이탈 ~ 몸통 roll 각속도"),
    ("hip_yaw_dev", "torso_roll", "고관절yaw  이탈 ~ 몸통 roll"),
    ("hip_yaw_dev", "torso_pitch", "고관절yaw  이탈 ~ 몸통 pitch"),
    ("hip_roll_dev", "torso_pitch", "고관절roll 이탈 ~ 몸통 pitch"),
]
best = 0.0
for a, b, label in pairs:
    c = corr(data[a], data[b])
    best = max(best, abs(c) if c == c else 0.0)
    print(f"  {label:34s} r = {c:+.3f}")

print("\n[명령별 고관절 roll 이탈 ~ 몸통 roll]")
for ci, (name, *_rest) in enumerate(COMMANDS):
    m = data["cmd"] == ci
    print(f"  {name:9s} r = {corr(data['hip_roll_dev'][m], data['torso_roll'][m]):+.3f}"
          f"   이탈 {np.abs(data['hip_roll_dev'][m]).mean()*DEG:5.2f} deg"
          f"   몸통roll RMS {np.sqrt((data['torso_roll'][m]**2).mean())*DEG:5.2f} deg")

print("\n[판정]")
if best >= 0.5:
    print(f"  최대 |r| = {best:.3f} -> 몸통과 함께 움직인다. 고관절 범위를 먼저 조이면")
    print("  균형 권한을 뺏을 위험이 크다. **토르소 안정화를 먼저** 하는 쪽이 안전하다.")
elif best >= 0.3:
    print(f"  최대 |r| = {best:.3f} -> 애매하다. 부분적으로만 얽혀 있다. 고관절 범위를")
    print("  조이되 넉넉한 여유를 두고, 넘어짐 빈도를 같이 감시할 것.")
else:
    print(f"  최대 |r| = {best:.3f} -> 몸통과 따로 논다. 보상 동작이 아니라 독립적인")
    print("  추종 실패다. **고관절 roll/yaw 액션 범위 제한**이 안전하고 빠르다.")
print("  (상관은 인과가 아니다. 낮은 값이 '몸통 원인'을 배제하는 근거이지,")
print("   높은 값이 '몸통이 원인'을 증명하지는 않는다.)")

if args_cli.out:
    np.savez_compressed(args_cli.out, **data)
    print(f"\n[ok] wrote {args_cli.out}")

env.close()
simulation_app.close()
