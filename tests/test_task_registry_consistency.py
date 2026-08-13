"""태스크 등록이 **세 곳에 흩어져 있다** — 그 셋이 어긋나지 않는지 본다.

새 버전을 추가하려면 지금 이 셋을 다 고쳐야 한다:

    1. `joystick_env_cfg.py`            설정 클래스 `JoystickEnvCfg_Vxx`
    2. `__init__.py`                    `gym.register(id=...)`
    3. `tasks/task_registry.py`         `ENV_CFG_CLASS` + `GAMMA097_TASKS`

`task_registry.py` 의 맨 위 주석이 "같은 매핑이 여섯 군데에 복제돼 있었다" 고
경고하면서 그걸 한 곳으로 모았는데, **gym 등록은 아직 __init__.py 에 따로
남아 있다.** 2026-08-13 에 v47 을 추가하면서 1·3 만 고치고 2 를 빠뜨려
학습이 `NameNotFound: Environment 'Isaac-OpenDuckMini-Joystick-V47' doesn't
exist` 로 죽었다. Isaac Sim 부팅에 1 분이 걸리므로 이런 실수는 늘 1 분 뒤에
발견된다. 이 테스트는 1 초 만에 잡는다.

일부러 **정적 파싱**으로 짰다 (ast). isaaclab 없이 아무 데서나 돌아야
커밋 전에 실제로 돌려 보게 된다.
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "source"
sys.path.insert(0, str(SRC))

from open_duck_mini_isaaclab.tasks.task_registry import (  # noqa: E402
    ENV_CFG_CLASS,
)


def _gym_registrations() -> dict[str, str]:
    """`__init__.py` 의 gym.register 에서 {태스크 id: 설정 클래스 이름}."""
    tree = ast.parse((SRC / "open_duck_mini_isaaclab" / "__init__.py").read_text())
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register"):
            continue
        tid = cls = None
        for kw in node.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                tid = kw.value.value
            elif kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if (isinstance(k, ast.Constant) and k.value == "env_cfg_entry_point"
                            and isinstance(v, ast.JoinedStr)):
                        # f"{__name__}.….joystick_env_cfg:JoystickEnvCfg_V47"
                        tail = "".join(p.value for p in v.values
                                       if isinstance(p, ast.Constant))
                        if ":" in tail:
                            cls = tail.rsplit(":", 1)[1]
        # 루프로 여러 변종을 등록하는 블록은 id 가 상수가 아니라 건너뛴다.
        if tid and cls:
            out[tid] = cls
    return out


def _cfg_classes() -> set[str]:
    tree = ast.parse((SRC / "open_duck_mini_isaaclab" / "tasks" / "velocity"
                      / "joystick_env_cfg.py").read_text())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def test_registry_tasks_are_gym_registered():
    """ENV_CFG_CLASS 에 있는 태스크는 gym 에도 등록돼 있어야 한다."""
    gym_ids = _gym_registrations()
    # 루프로 등록되는 변종들(id 가 상수가 아님)은 정적으로 못 읽으므로,
    # 이 테스트가 실제로 지키는 것은 **명시적으로 등록된 버전들**이다.
    # 그게 곧 새 버전을 추가할 때 손대는 자리다.
    missing = [t for t in ENV_CFG_CLASS
               if t.startswith("Isaac-OpenDuckMini-Joystick-V") and t not in gym_ids]
    assert not missing, (
        "ENV_CFG_CLASS 에는 있는데 __init__.py 의 gym.register 에 없다:\n  "
        + "\n  ".join(missing)
        + "\n\n__init__.py 에 gym.register 블록을 추가할 것. 안 그러면 학습이 "
          "NameNotFound 로 죽는다 (Isaac Sim 부팅 1분 뒤에).")


def test_gym_and_registry_agree_on_cfg_class():
    """같은 태스크 id 는 양쪽에서 같은 설정 클래스를 가리켜야 한다."""
    gym_ids = _gym_registrations()
    bad = [(t, ENV_CFG_CLASS[t], c) for t, c in gym_ids.items()
           if t in ENV_CFG_CLASS and ENV_CFG_CLASS[t] != c]
    assert not bad, (
        "gym.register 와 ENV_CFG_CLASS 가 다른 클래스를 가리킨다:\n  "
        + "\n  ".join(f"{t}: registry={a}  gym={b}" for t, a, b in bad))


def test_referenced_cfg_classes_exist():
    """양쪽이 가리키는 설정 클래스가 joystick_env_cfg.py 에 실제로 있어야 한다."""
    known = _cfg_classes()
    gym_ids = _gym_registrations()
    missing = sorted({c for c in ENV_CFG_CLASS.values() if c not in known}
                     | {c for c in gym_ids.values() if c not in known})
    assert not missing, (
        "가리키는 설정 클래스가 joystick_env_cfg.py 에 없다:\n  "
        + "\n  ".join(missing))
