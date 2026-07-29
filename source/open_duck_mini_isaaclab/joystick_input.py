"""Xbox 컨트롤러로 시뮬레이션 명령을 실시간 조종한다.

리눅스 joydev 인터페이스(`/dev/input/js0`)를 직접 읽는다. pygame이나 evdev를
쓰지 않는 이유는 두 가지다: Isaac 파이썬에 그 패키지들이 있으리라는 보장이 없고,
joydev 프로토콜은 8바이트 고정 구조라 표준 라이브러리 `struct`만으로 충분하다.
의존성이 없으니 Isaac Sim 없이도 이 파일 하나만 따로 테스트할 수 있다
(`tests/test_joystick_input.py`).

joydev 이벤트 (리눅스 커널 `linux/joystick.h`):

    uint32 time      밀리초 타임스탬프
    int16  value     축이면 -32767..32767, 버튼이면 0/1
    uint8  type      0x01 버튼, 0x02 축, 0x80 은 초기 상태 통보 플래그
    uint8  number    축/버튼 번호

장치를 열면 커널이 현재 상태를 JS_EVENT_INIT 플래그가 붙은 이벤트로 한 번씩
보내준다. 이걸 버리면 안 된다 — 버리면 사용자가 스틱을 건드리기 전까지 축이
0인지 아닌지 알 수 없다.

읽기는 논블로킹이다. 시뮬 루프는 60Hz로 돌아야 하는데, 조이스틱 입력이 없다고
루프가 멈추면 안 된다. 매 스텝 `poll()`로 밀린 이벤트만 훑고 즉시 반환한다.

xpad 드라이버가 붙인 Xbox 컨트롤러의 축 번호는 다음과 같다:

    0 왼쪽 스틱 X (오른쪽 +)      3 오른쪽 스틱 X (오른쪽 +)
    1 왼쪽 스틱 Y (아래쪽 +)      4 오른쪽 스틱 Y (아래쪽 +)
    2 LT                          5 RT
    6 D-pad X                     7 D-pad Y
"""

from __future__ import annotations

import os
import struct

_EVENT_FORMAT = "IhBB"
_EVENT_SIZE = struct.calcsize(_EVENT_FORMAT)

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

#: 스틱을 놓아도 정확히 0으로 돌아오지 않는다. 이 값 이하는 0으로 본다.
#: 로봇 명령에 그대로 실리면 "정지" 명령이 영영 안 나온다.
DEFAULT_DEADZONE = 0.12

AXIS_LEFT_X, AXIS_LEFT_Y = 0, 1
AXIS_RIGHT_X = 3

BUTTON_A = 0


class GamepadUnavailable(RuntimeError):
    """장치가 없거나 열 수 없을 때."""


class Gamepad:
    """joydev 장치 하나를 논블로킹으로 읽는다."""

    def __init__(self, path: str = "/dev/input/js0", deadzone: float = DEFAULT_DEADZONE):
        self.path = path
        self.deadzone = deadzone
        self._axes: dict[int, float] = {}
        self._buttons: dict[int, bool] = {}
        self.connected = False
        try:
            self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            raise GamepadUnavailable(
                f"{path} 를 열 수 없습니다 ({exc}). 컨트롤러가 연결되어 있는지 "
                "확인하세요 — 유선 Xbox 패드는 꽂으면 커널 xpad 드라이버가 바로 "
                "잡습니다. `ls /dev/input/js*` 로 확인할 수 있습니다."
            ) from exc
        self.connected = True

    def poll(self) -> None:
        """밀린 이벤트를 모두 소화한다. 없으면 바로 반환한다."""
        if not self.connected:
            return
        while True:
            try:
                data = os.read(self._fd, _EVENT_SIZE)
            except BlockingIOError:
                return  # 더 읽을 것이 없다 — 정상
            except OSError:
                # 실행 중 케이블이 빠지면 여기로 온다. 죽지 말고 중립으로 둔다.
                self.connected = False
                self._axes.clear()
                self._buttons.clear()
                return
            if not data or len(data) < _EVENT_SIZE:
                return
            self._apply(data)

    def _apply(self, data: bytes) -> None:
        _, value, ev_type, number = struct.unpack(_EVENT_FORMAT, data)
        # 초기 상태 통보도 실제 상태이므로 플래그만 떼고 똑같이 반영한다.
        ev_type &= ~JS_EVENT_INIT
        if ev_type == JS_EVENT_AXIS:
            self._axes[number] = max(-1.0, value / 32767.0)
        elif ev_type == JS_EVENT_BUTTON:
            self._buttons[number] = bool(value)

    def axis(self, number: int) -> float:
        """-1..1. 데드존 안이면 0. 데드존 바깥은 0에서 다시 시작하도록 재조정한다
        (안 하면 스틱을 살짝 밀자마자 명령이 뚝 튄다)."""
        raw = self._axes.get(number, 0.0)
        if abs(raw) <= self.deadzone:
            return 0.0
        scaled = (abs(raw) - self.deadzone) / (1.0 - self.deadzone)
        return scaled if raw > 0 else -scaled

    def button(self, number: int) -> bool:
        return self._buttons.get(number, False)

    def close(self) -> None:
        if self.connected:
            os.close(self._fd)
            self.connected = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def command_from_gamepad(
    pad: Gamepad,
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float],
) -> tuple[float, float, float]:
    """스틱 -> (vx, vy, yaw_rate). 학습 때 쓴 명령 범위로 스케일한다.

    범위를 넘겨받는 이유: 정책은 학습 중 본 명령 분포 안에서만 믿을 수 있다.
    스틱을 끝까지 밀었을 때 그 상한에 정확히 닿아야 하고, 넘어가면 안 된다.

    방향 규약 — 로봇 기준 +x 앞, +y 왼쪽, yaw는 반시계 방향이 +.
    스틱은 위/왼쪽이 음수라서 부호를 뒤집는다. 그래야 "스틱을 미는 방향으로
    로봇이 간다"가 된다.
    """
    if pad.button(BUTTON_A):  # 비상 정지 — 손을 떼는 것보다 확실하다
        return 0.0, 0.0, 0.0

    vx = -pad.axis(AXIS_LEFT_Y)
    vy = -pad.axis(AXIS_LEFT_X)
    yaw = -pad.axis(AXIS_RIGHT_X)

    return (
        _scale(vx, lin_vel_x_range),
        _scale(vy, lin_vel_y_range),
        _scale(yaw, ang_vel_yaw_range),
    )


def _scale(value: float, rng: tuple[float, float]) -> float:
    """-1..1 을 범위로 편다. 범위가 0폭이면(예: 머리축이 잠긴 태스크) 0."""
    lo, hi = rng
    if hi == lo:
        return lo
    return value * (hi if value >= 0 else -lo)
