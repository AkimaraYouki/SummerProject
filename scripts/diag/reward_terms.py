"""학습된 정책이 실제로 어느 리워드 항에서 점수를 벌고 있는지.

`reward_breakdown_v2.py`를 대체한다. 그쪽은 리워드 수식을 스크립트 안에 **복제**해
놓아서 환경이 바뀌면 조용히 어긋났고, 실제로 Path 태스크를 모른 채 남아 있었다.
여기서는 환경이 `_get_rewards`에서 직접 채우는 `extras["log"]`를 그대로 읽는다.
학습이 텐서보드에 올리는 값과 정의상 같은 숫자다.

각 항은 이미 dt가 곱해져 있어서 **항목들의 합 = 스텝당 총 리워드**다. 그래서
"이 항이 전체의 몇 할인가"를 그대로 읽을 수 있다.

명령 6종을 gait_compare.py와 같은 값으로 돌린다. 명령마다 균형이 다를 수 있어서
(예: 정지에서는 stand_still, 전진에서는 tracking) 나눠서 본다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps_per_cmd", type=int, default=200)
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
env_cfg.events.push_robot = None  # 외란이 섞이면 항목 균형이 흐려진다

env = gym.make(args_cli.task, cfg=env_cfg)
agent_cfg = runner_cfg_for(args_cli.task)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = build_runner(env, agent_cfg)
load_checkpoint(runner, args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

# extras["log"] 는 _get_rewards 가 채우므로 리셋 직후에는 아직 없다.
# 첫 스텝을 밟은 뒤에 확인한다.
_checked = False

per_cmd = {}  # 명령 -> {항목: [값...]}
obs = env.get_observations()
for name, cx, cy, cw in COMMANDS:
    acc = {}
    for step in range(args_cli.steps_per_cmd):
        u._command[:, 0] = cx
        u._command[:, 1] = cy
        u._command[:, 2] = cw
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if not _checked:
            if "log" not in u.extras:
                raise SystemExit(
                    "환경이 extras['log'] 를 채우지 않습니다. joystick_env.py 의 "
                    "_get_rewards 에 항목별 로깅이 들어갔는지 확인하세요 (2026-07-30 추가)."
                )
            _checked = True
        if step < args_cli.steps_per_cmd // 2:
            continue  # 명령 전환 직후 과도구간은 버린다
        for k, v in u.extras["log"].items():
            acc.setdefault(k.replace("Episode_Reward/", ""), []).append(float(v))
    per_cmd[name] = {k: float(np.mean(v)) for k, v in acc.items()}

terms = sorted(per_cmd["forward"].keys())
overall = {t: float(np.mean([per_cmd[c][t] for c in per_cmd])) for t in terms}
total = sum(overall.values())

lines = []
lines.append("=" * 74)
lines.append("리워드 항목별 기여 (스텝당, 이미 dt 곱해짐 — 합 = 스텝당 총 리워드)")
lines.append("=" * 74)
header = f"{'항목':<20}{'전체':>9}{'비중':>8}   " + "".join(f"{c[:5]:>8}" for c, *_ in COMMANDS)
lines.append(header)
lines.append("-" * len(header))
for t in sorted(terms, key=lambda k: -abs(overall[k])):
    share = 100.0 * overall[t] / total if total else float("nan")
    row = f"{t:<20}{overall[t]:>9.4f}{share:>7.1f}%   " + "".join(f"{per_cmd[c][t]:>8.4f}" for c, *_ in COMMANDS)
    lines.append(row)
lines.append("-" * len(header))
lines.append(f"{'합계':<20}{total:>9.4f}")

pos = {t: v for t, v in overall.items() if v > 0}
if pos:
    top = max(pos, key=pos.get)
    lines.append("")
    lines.append(f"[해석] 양의 리워드 중 최대는 '{top}' ({pos[top]:.4f}, 양의 합의 "
                 f"{100*pos[top]/sum(pos.values()):.0f}%).")
    track = sum(v for k, v in overall.items() if k.startswith("tracking_"))
    lines.append(f"        명령 추종(tracking_*) 합계는 {track:.4f}.")
    if track < 0.5 * pos[top]:
        lines.append("        추종이 최대 항의 절반에도 못 미친다 — 정책이 명령을 따르는 것보다")
        lines.append("        다른 것으로 점수를 버는 구조다. 추종 성능이 안 오르면 여기를 의심할 것.")

text = "\n".join(lines)
print("\n" + text, flush=True)

# Isaac Sim 이 stdout 을 삼키는 경우가 있어 파일로도 남긴다.
if args_cli.out:
    with open(args_cli.out, "w") as f:
        f.write(text + "\n")
    np.savez_compressed(args_cli.out.replace(".txt", ".npz"),
                        **{f"{c}|{t}": np.array(per_cmd[c][t]) for c in per_cmd for t in terms})

env.close()
simulation_app.close()
