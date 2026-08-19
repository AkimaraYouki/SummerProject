"""소스 전체에 **정의되지 않은 이름**이 없는지 정적으로 본다.

## 왜 이 테스트가 있나

2026-08-20, 죽은 cfg 클래스 4 개를 지우는 정리 커밋(59dab20)이 그 사이에 끼어
있던 `READY_JOINT_POS_H175_ZNECK` 표까지 같이 지웠다. 그런데 그 표를 쓰는
`JoystickEnvCfg_ZNeck` 은 남아 있었다.

결과는 `NameError` 한 줄이 아니라 **`joystick_env_cfg` 모듈 import 자체가
깨진 것**이었다. 그 모듈은 태스크 등록의 뿌리라서 학습·재생·측정이 전부 죽는다.
증상은 스트림 테스트를 하다 우연히 드러났고, 그때까지 반나절을 모르고 있었다.

파이썬은 모듈을 끝까지 실행해 봐야 이걸 알고, Isaac 부팅에 20 초가 걸린다.
pyflakes 의 F821 은 같은 것을 **Isaac 없이 1 초에** 잡는다.

Isaac 없이 돈다.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = ("source", "scripts")


def test_no_undefined_names():
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pyflakes", *TARGETS],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:                                    # pragma: no cover
        pytest.skip("pyflakes 없음")
    if r.returncode == 2 and "No module named" in r.stderr:
        pytest.skip("pyflakes 없음")
    bad = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
    assert not bad, (
        "정의되지 않은 이름이 있다 — 모듈 import 가 깨진다:\n  "
        + "\n  ".join(bad)
    )
