"""`rl_walk --lean-back` 의 **부호**를 고정한다.

이 회전이 반대로 들어가면 정책이 "뒤로 젖혔다" 고 판단해 앞으로 숙이고,
이미 앞으로 넘어지는 로봇을 더 빨리 넘어뜨린다. 오류 없이 조용히 틀리는
종류라 정적으로 막는다.

기준은 IsaacLab 의 projected_gravity 다. 몸통 +x 앞, +z 위에서 기수를 deg 만큼
숙이면 projected_gravity = (sin deg, 0, -cos deg). 심 반복측정에서 전진 보행의
proj_grav_x 가 +0.062, 후진이 -0.113 으로 나와 "+x = 앞으로 숙임" 을 확인했다.

rl_walk.py 는 하드웨어 라이브러리를 import 하므로 함수만 잘라 실행한다.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hw" / "rl_walk.py"


def _load():
    src = SRC.read_text(encoding="utf-8")
    m = re.search(r"def lean_back_rotation.*?return np\.array\(.*?\)\n", src, re.S)
    assert m, "rl_walk.py 에 lean_back_rotation 이 없다"
    ns = {"np": np}
    exec(m.group(0), ns)
    return ns["lean_back_rotation"]


@pytest.mark.parametrize("deg", [0.0, 2.0, 4.0, 8.0])
def test_upright_reads_as_nose_down(deg):
    rot = _load()
    grav_up = np.array([0.0, 0.0, 9.81])        # rl_walk: 직립 시 grav_src=(0,0,+g)
    pg = -(rot(deg) @ grav_up) / 9.81            # rl_walk 의 projected_gravity 식
    t = np.radians(deg)
    assert np.allclose(pg, [np.sin(t), 0.0, -np.cos(t)], atol=1e-9)


def test_positive_deg_makes_policy_lean_back():
    """양수 deg 에서 proj_grav_x > 0 (숙인 것처럼 보임) 이어야 정책이 뒤로 젖힌다."""
    pg = -(_load()(4.0) @ np.array([0.0, 0.0, 9.81])) / 9.81
    assert pg[0] > 0


def test_rotation_is_proper():
    r = _load()(5.0)
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1.0)
