"""주어진 태스크의 READY 자세로 로봇을 놓고 **실제로 안착하는 높이**를 잰다.

`settle_pose.py` 는 전역 설정만 재서 새 태스크(예: 키를 높인 Tall)에 못 쓴다.
여기서는 태스크를 인자로 받아 그 설정의 자세로 잰다.

왜 실측하는가. 이 값 두 개가 추측이면 곧바로 깨진다:

  ready_base_height   붕괴 종료 판정의 기준. 너무 높게 잡으면 멀쩡히 걷는데도
                      "쓰러졌다"고 에피소드를 끊고, 낮게 잡으면 주저앉아도 안 끊긴다.
  스폰 높이           너무 낮으면 발이 지면을 파고들어 PhysX 가 튕겨내고
                      (사용자가 봤던 "처음에 뿅 하고 튀어오름"), 높으면 낙하한다.

액션 0 으로 두고 물리에 맡겨 안착시킨 뒤, 마지막 구간의 루트 z 를 본다.
레퍼런스 위상마다 다리 높이가 다르므로 스폰 높이는 **최댓값 + 여유**로 잡아야
어떤 위상에서도 관통이 없다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--spawn", type=float, default=0.0,
                    help="스폰 높이를 강제한다. 0 이면 태스크 설정값 그대로")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import open_duck_mini_isaaclab.tasks  # noqa: E402, F401
from open_duck_mini_isaaclab.tasks.task_registry import env_cfg_for  # noqa: E402

env_cfg = env_cfg_for(args_cli.task)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.events.push_robot = None
if args_cli.spawn > 0:
    env_cfg.robot.init_state.pos = (0.0, 0.0, args_cli.spawn)

env = gym.make(args_cli.task, cfg=env_cfg)
u = env.unwrapped
u.reset()

spawn_z = env_cfg.robot.init_state.pos[2]
zs = []
zero = torch.zeros(u.num_envs, u.action_space.shape[-1] if hasattr(u.action_space, "shape") else 14,
                   device=u.device)
for step in range(args_cli.num_steps):
    with torch.inference_mode():
        u.step(zero)
    if step >= args_cli.num_steps // 2:  # 안착 과도구간은 버린다
        zs.append(u._robot.data.root_pos_w[:, 2].cpu().numpy())

z = np.concatenate(zs)
settled = float(np.mean(z))
lines = [
    "=" * 62,
    f"태스크: {args_cli.task}",
    f"스폰 높이      : {spawn_z*1000:7.1f} mm",
    f"안착 높이 평균 : {settled*1000:7.1f} mm   (표준편차 {np.std(z)*1000:.1f})",
    f"안착 최소/최대 : {np.min(z)*1000:7.1f} / {np.max(z)*1000:.1f} mm",
    "-" * 62,
    f"  ready_base_height 로 쓸 값 : {settled:.4f}",
    f"  스폰 높이로 쓸 값          : {float(np.max(z)) + 0.004:.4f}   (최대 안착 + 4 mm 여유)",
    "=" * 62,
]
print("\n" + "\n".join(lines), flush=True)
with open("/home/parksuho/odm_out/settle_height.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

env.close()
simulation_app.close()
