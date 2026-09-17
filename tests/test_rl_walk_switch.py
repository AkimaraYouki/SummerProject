"""`rl_walk` 의 후진 정책 전환(`--onnx-back`)과 방위 오차 끈(`--path-yaw-clip`).

둘 다 실기에서만 도는 코드라 오류 없이 조용히 틀리면 로봇이 넘어진다.
rl_walk.py 는 하드웨어 라이브러리를 import 하므로 함수만 잘라 실행한다.
"""
from __future__ import annotations

import math
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hw" / "rl_walk.py"


def _load():
    src = SRC.read_text(encoding="utf-8")
    ns = {"math": math}
    for pat, fl in ((r"^BACK_ENTER_VX = [^\n]*$", re.M), (r"^BACK_LEAVE_VX = [^\n]*$", re.M),
                    (r"^def _wrap\(.*?(?=^\S)", re.S | re.M),
                    (r"^def choose_back_policy\(.*?(?=^\S)", re.S | re.M),
                    (r"^def leash_path_yaw\(.*?(?=^\S)", re.S | re.M)):
        m = re.search(pat, src, fl)
        assert m, f"rl_walk.py 에서 못 찾았다: {pat}"
        exec(m.group(0), ns)
    return ns


NS = _load()
choose = NS["choose_back_policy"]
leash = NS["leash_path_yaw"]


def test_forward_and_zero_use_forward_policy():
    for vx in (0.15, 0.05, 0.0, -0.01):
        assert choose(False, vx) is False


def test_enters_back_below_threshold():
    assert choose(False, -0.03) is True
    assert choose(False, -0.15) is True


def test_hysteresis_holds_back_until_leave():
    # 들어간 뒤에는 -0.01 에서도 후진 정책을 유지한다
    assert choose(True, -0.01) is True
    assert choose(True, -0.004) is False
    assert choose(True, 0.0) is False
    assert choose(True, 0.1) is False


def test_no_chatter_on_noise_near_zero():
    using, flips = False, 0
    for vx in (-0.012, -0.008, -0.015, -0.011, -0.009) * 20:
        nb = choose(using, vx)
        flips += nb != using
        using = nb
    assert flips == 0


def test_leash_off_is_identity():
    assert leash(0.0, 3.0, 0.0) == 3.0


def test_leash_inside_is_identity():
    assert leash(0.2, 0.0, 0.5) == 0.0


# 뒤의 둘은 ±π 경계를 넘는 경우: wrap(3-(-2)) = -1.28, wrap(-3-2) = +1.28
@pytest.mark.parametrize("robot,path", [(0.0, 2.0), (0.0, -2.0), (3.0, -2.0), (-3.0, 2.0)])
def test_leash_caps_error_with_right_sign(robot, path):
    wrap = NS["_wrap"]
    e_before = wrap(robot - path)
    new = leash(robot, path, 0.5)
    e_after = wrap(robot - new)
    assert abs(e_after) == pytest.approx(0.5, abs=1e-9)
    assert math.copysign(1, e_after) == math.copysign(1, e_before)


def test_leash_prevents_sign_flip_when_lagging():
    """명령 1 rad/s, 실제 0.4 rad/s 로 10 초. 끈이 없으면 오차 부호가 뒤집힌다."""
    wrap = NS["_wrap"]
    dt = 0.02
    for clip, expect_flip in ((0.0, True), (0.5, False)):
        robot = path = 0.0
        signs = set()
        for _ in range(500):
            robot = wrap(robot + 0.4 * dt)
            path = wrap(path + 1.0 * dt)
            path = leash(robot, path, clip)
            e = wrap(robot - path)
            if abs(e) > 1e-6:
                signs.add(e > 0)
        assert (len(signs) == 2) is expect_flip
