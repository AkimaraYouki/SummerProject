"""학습과 재생이 **같은 신경망 크기**를 쓰는지 정적으로 확인한다.

## 왜 이 테스트가 있나

2026-09-17, v85 를 학습하고 측정·녹화하려는데 둘 다 죽었다:

    size mismatch for mlp.0.weight: checkpoint [512, 107], current model [256, 107]

학습(`odm train` -> IsaacLab train.py)은 `__init__.py` 의 gym 등록
`rsl_rl_cfg_entry_point` 를 따르고, 측정·재생(`gait_compare`,
`play_fixed_cmd`)은 `task_registry.runner_cfg_for` 를 따른다. 두 경로가
**같은 러너를 가리켜야 하는데** 따로 적혀 있다.

v83~v87 을 추가하는 스크립트가 `_BIG_NET_TASKS` 에 넣는 문자열 치환에서
일치하는 곳을 못 찾고 **아무 오류 없이 넘어갔다.** 그래서 학습은 큰 망
(Gamma097, 512-256-128)으로, 재생은 기본 망(256-128-64)으로 만들어졌다.
기존 `test_task_registry_consistency` 는 태스크 **이름**만 대조하고 러너는
안 봐서 못 잡았다.

Isaac 없이 돈다 (두 파일을 텍스트로 파싱).
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "source" / "open_duck_mini_isaaclab" / "__init__.py"
REG = ROOT / "source" / "open_duck_mini_isaaclab" / "tasks" / "task_registry.py"


def _gym_runners():
    """gym 등록: 태스크 -> rsl_rl_cfg_entry_point 의 러너 클래스 이름."""
    s = INIT.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(
        r'id="(Isaac-OpenDuckMini-Joystick-[^"]+)".*?rsl_rl_ppo_cfg:(\w+)"', s, re.S
    ):
        out[m.group(1)] = m.group(2)
    return out


def _registry_runner(task: str, s: str) -> str:
    """task_registry.runner_cfg_for 의 분기를 그대로 흉내 낸다."""
    exp = re.search(r"if task in \((.*?)\):\s*\n\s*from .*?import (JoystickPPORunnerCfg_Explore13)", s, re.S)
    if exp and f'"{task}"' in exp.group(1):
        return "JoystickPPORunnerCfg_Explore13"
    if 'task == "Isaac-OpenDuckMini-Joystick-Sym-v0"' in s and task.endswith("Sym-v0"):
        return "JoystickPPORunnerCfg_Symmetry"
    big = re.search(r"_BIG_NET_TASKS\s*=\s*\{(.*?)\n\}", s, re.S)
    if big and f'"{task}"' in big.group(1):
        return "JoystickPPORunnerCfg_Gamma097"
    return "JoystickPPORunnerCfg"


def test_train_and_play_use_same_runner():
    s = REG.read_text(encoding="utf-8")
    gym = _gym_runners()
    assert gym, "__init__.py 에서 등록을 하나도 못 읽었다 — 정규식을 확인할 것"
    # v42 이후만 본다. 그 전 판들은 러너 구조가 달라 이 규칙의 대상이 아니다.
    bad = []
    for task, trained in sorted(gym.items()):
        m = re.search(r"-V(\d+)", task)
        if not m or int(m.group(1)) < 42:
            continue
        played = _registry_runner(task, s)
        if trained != played:
            bad.append(f"{task}: 학습 {trained}  vs  재생 {played}")
    assert not bad, "학습과 재생의 러너가 다르다 (신경망 크기 불일치):\n  " + "\n  ".join(bad)
