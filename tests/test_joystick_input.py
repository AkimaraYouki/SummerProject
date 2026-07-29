"""joydev 리더 검증 — 컨트롤러도 Isaac Sim도 없이 돈다.

joydev는 8바이트 고정 구조를 그냥 흘려보내는 프로토콜이라, 진짜 장치 대신
파이프에 같은 바이트를 써 넣으면 커널이 보내는 것과 구분되지 않는다.
그래서 컨트롤러가 안 꽂혀 있어도 파싱·데드존·스케일을 전부 검증할 수 있다.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.joystick_input import (  # noqa: E402
    AXIS_LEFT_X,
    AXIS_LEFT_Y,
    AXIS_RIGHT_X_CLASSIC,
    AXIS_RIGHT_X_MODERN,
    BUTTON_A,
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_INIT,
    Gamepad,
    GamepadUnavailable,
    command_from_gamepad,
)

FMT = "IhBB"
X_RANGE = (-0.15, 0.15)
Y_RANGE = (-0.2, 0.2)
YAW_RANGE = (-1.0, 1.0)


def _event(ev_type, number, value):
    return struct.pack(FMT, 0, value, ev_type, number)


def _pad_fed_with(*events):
    """이벤트를 파이프에 미리 써 두고, 그 읽기쪽을 장치처럼 물린 Gamepad를 만든다."""
    r, w = os.pipe()
    os.set_blocking(r, False)
    for e in events:
        os.write(w, e)
    os.close(w)

    pad = Gamepad.__new__(Gamepad)  # __init__ 의 os.open 을 우회
    pad.path = "<pipe>"
    pad.deadzone = 0.12
    pad._axes, pad._buttons = {}, {}
    pad._fd = r
    pad.connected = True
    pad.axis_right_x = AXIS_RIGHT_X_CLASSIC
    pad.layout = "미판별"
    pad._layout_done = False
    return pad


def _resting(overrides=None):
    """8축 전부에 대해 '쉬고 있는' 초기 상태 이벤트를 만든다.
    커널이 장치를 열 때 보내주는 것과 같은 모양이다."""
    values = {n: 0 for n in range(8)}
    values.update(overrides or {})
    return [_event(JS_EVENT_AXIS | JS_EVENT_INIT, n, v) for n, v in values.items()]


def test_missing_device_is_a_clear_error():
    try:
        Gamepad("/dev/input/js-does-not-exist")
    except GamepadUnavailable as exc:
        assert "js" in str(exc)
    else:
        raise AssertionError("장치가 없는데 예외가 안 났습니다")


def test_axis_and_button_parsing():
    pad = _pad_fed_with(
        _event(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767),
        _event(JS_EVENT_BUTTON, BUTTON_A, 1),
    )
    pad.poll()
    assert pad.axis(AXIS_LEFT_Y) < -0.9
    assert pad.button(BUTTON_A) is True


def test_init_events_are_not_discarded():
    """장치를 열면 커널이 현재 상태를 INIT 플래그로 한 번 보낸다. 이걸 버리면
    사용자가 스틱을 건드리기 전까지 상태를 모른다."""
    pad = _pad_fed_with(_event(JS_EVENT_AXIS | JS_EVENT_INIT, AXIS_RIGHT_X_MODERN, 32767))
    pad.poll()
    assert pad.axis(AXIS_RIGHT_X_MODERN) > 0.9


# ── 축 배치 자동 판별 ────────────────────────────────────────────────────
# 하드코딩했다가 실제 Xbox Wireless 패드에서 틀렸다. 오른쪽 스틱 X라고 믿은
# 3번이 그 패드에서는 오른쪽 스틱 Y였고, 스틱을 안 건드렸는데 yaw 명령이
# -1.0 rad/s 로 꽉 차 있었다. 두 배치를 모두 고정해 둔다.

def test_detects_modern_layout_from_trigger_rest_positions():
    """Xbox Wireless: 4=LT, 5=RT 가 -1 에서 쉰다 -> 오른쪽 스틱 X 는 2번."""
    pad = _pad_fed_with(*_resting({4: -32767, 5: -32767}))
    pad.poll()
    assert pad.axis_right_x == AXIS_RIGHT_X_MODERN


def test_detects_classic_layout_from_trigger_rest_positions():
    """유선 xpad: 2=LT, 5=RT 가 -1 에서 쉰다 -> 오른쪽 스틱 X 는 3번."""
    pad = _pad_fed_with(*_resting({2: -32767, 5: -32767}))
    pad.poll()
    assert pad.axis_right_x == AXIS_RIGHT_X_CLASSIC


def test_idle_pad_commands_nothing_on_either_layout():
    """스틱에서 손을 뗐는데 명령이 나가면 안 된다. 실제로 겪은 버그다 —
    yaw 가 -1.0 rad/s 로 고정돼 로봇이 계속 돌았다."""
    for triggers in ({4: -32767, 5: -32767}, {2: -32767, 5: -32767}):
        pad = _pad_fed_with(*_resting(triggers))
        pad.poll()
        assert command_from_gamepad(pad, X_RANGE, Y_RANGE, YAW_RANGE) == (0.0, 0.0, 0.0)


def test_right_stick_strafes_on_the_detected_axis():
    """오른쪽 스틱 가로 = 게걸음(vy). 판별된 축을 써야 한다."""
    pad = _pad_fed_with(
        *_resting({4: -32767, 5: -32767}),
        _event(JS_EVENT_AXIS, AXIS_RIGHT_X_MODERN, 32767),
    )
    pad.poll()
    vx, vy, yaw = command_from_gamepad(pad, X_RANGE, Y_RANGE, YAW_RANGE)
    assert vy < -0.19, "오른쪽으로 밀면 -y 로 게걸음"
    assert yaw == 0.0, "오른쪽 스틱은 더 이상 회전이 아니다"


def test_deadzone_zeroes_small_drift_and_rescales():
    pad = _pad_fed_with(_event(JS_EVENT_AXIS, AXIS_LEFT_X, int(0.05 * 32767)))
    pad.poll()
    assert pad.axis(AXIS_LEFT_X) == 0.0, "드리프트가 명령에 실리면 정지가 안 된다"

    # 데드존 바로 바깥은 0에서 다시 시작해야 한다 (안 그러면 명령이 뚝 튄다)
    pad = _pad_fed_with(_event(JS_EVENT_AXIS, AXIS_LEFT_X, int(0.13 * 32767)))
    pad.poll()
    assert 0.0 < pad.axis(AXIS_LEFT_X) < 0.05


def test_full_deflection_reaches_exactly_the_trained_range():
    """끝까지 밀면 학습 때 쓴 상한에 닿아야 하고, 넘으면 안 된다 —
    정책은 학습 중 본 명령 분포 밖에서는 믿을 수 없다."""
    pad = _pad_fed_with(_event(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767))
    pad.poll()
    vx, _, _ = command_from_gamepad(pad, X_RANGE, Y_RANGE, YAW_RANGE)
    assert abs(vx - 0.15) < 1e-6


def test_stick_direction_matches_robot_frame():
    """스틱을 미는 방향으로 로봇이 가야 한다.
    로봇 기준 +x 앞, +y 왼쪽, yaw 반시계 +. 스틱은 위/왼쪽이 음수다.

    배치(2026-07-30 사용자 요청으로 두 스틱의 가로축을 맞바꿈):
      왼쪽 세로=전후, 왼쪽 가로=회전, 오른쪽 가로=게걸음
    """
    up = _pad_fed_with(*_resting({1: -32767, 4: -32767, 5: -32767}))
    up.poll()
    assert command_from_gamepad(up, X_RANGE, Y_RANGE, YAW_RANGE)[0] > 0, "위로 밀면 전진"

    left = _pad_fed_with(*_resting({0: -32767, 4: -32767, 5: -32767}))
    left.poll()
    assert command_from_gamepad(left, X_RANGE, Y_RANGE, YAW_RANGE)[2] > 0, "왼쪽 스틱 좌 = 반시계 회전"

    # 오른쪽 스틱을 왼쪽으로 밀면 2번이 -1이 되어 판별이 모호해진다. 그래서
    # 여기서는 오른쪽으로 밀어 -y 를 확인한다 (모호한 경우 자체는 아래 테스트에서).
    right_stick = _pad_fed_with(*_resting({2: 32767, 4: -32767, 5: -32767}))
    right_stick.poll()
    assert command_from_gamepad(right_stick, X_RANGE, Y_RANGE, YAW_RANGE)[1] < 0, "오른쪽 스틱 우 = -y 게걸음"


def test_ambiguous_layout_falls_back_to_modern_and_says_so():
    """연결 순간 오른쪽 스틱을 왼쪽으로 밀고 있으면 2번과 4번이 동시에 -1이라
    두 배치를 구분할 수 없다. 조용히 틀린 쪽을 고르지 말고 알려야 한다."""
    pad = _pad_fed_with(*_resting({2: -32767, 4: -32767, 5: -32767}))
    pad.poll()
    assert pad.axis_right_x == AXIS_RIGHT_X_MODERN
    assert "모호" in pad.layout


def test_left_and_right_horizontal_axes_are_swapped():
    """사용자 요청으로 맞바꾼 것 — 되돌아가면 여기서 잡힌다.
    왼쪽 가로는 회전만, 오른쪽 가로는 게걸음만 건드려야 한다."""
    left_only = _pad_fed_with(*_resting({0: 32767, 4: -32767, 5: -32767}))
    left_only.poll()
    vx, vy, yaw = command_from_gamepad(left_only, X_RANGE, Y_RANGE, YAW_RANGE)
    assert yaw != 0.0 and vy == 0.0, "왼쪽 스틱 가로는 회전만"

    right_only = _pad_fed_with(*_resting({2: 32767, 4: -32767, 5: -32767}))
    right_only.poll()
    vx, vy, yaw = command_from_gamepad(right_only, X_RANGE, Y_RANGE, YAW_RANGE)
    assert vy != 0.0 and yaw == 0.0, "오른쪽 스틱 가로는 게걸음만"


def test_button_a_is_an_emergency_stop():
    pad = _pad_fed_with(
        _event(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767),
        _event(JS_EVENT_BUTTON, BUTTON_A, 1),
    )
    pad.poll()
    assert command_from_gamepad(pad, X_RANGE, Y_RANGE, YAW_RANGE) == (0.0, 0.0, 0.0)


def test_zero_width_range_stays_zero():
    """머리축처럼 태스크가 잠가둔 축은 스틱을 밀어도 0이어야 한다."""
    pad = _pad_fed_with(_event(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767))
    pad.poll()
    vx, _, _ = command_from_gamepad(pad, (0.0, 0.0), Y_RANGE, YAW_RANGE)
    assert vx == 0.0


def test_poll_returns_immediately_when_idle():
    """시뮬 루프가 조이스틱 때문에 멈추면 안 된다."""
    pad = _pad_fed_with()
    pad.poll()
    pad.poll()
    assert pad.axis(AXIS_LEFT_X) == 0.0


def test_unplug_mid_run_goes_neutral_instead_of_crashing():
    pad = _pad_fed_with(_event(JS_EVENT_AXIS, AXIS_LEFT_Y, -32767))
    pad.poll()
    os.close(pad._fd)  # 케이블이 빠진 상황
    pad.poll()
    assert pad.connected is False
    assert command_from_gamepad(pad, X_RANGE, Y_RANGE, YAW_RANGE) == (0.0, 0.0, 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
