#!/usr/bin/env python3
"""젯슨에 직접 꽂은 조이스틱으로 속도 명령을 만든다. `cmd_udp.CommandReceiver` 와
같은 인터페이스라 `rl_walk.py` 에서 바꿔 끼우기만 하면 된다.

    python3 ~/joy_local.py            # 명령이 어떻게 나오는지만 확인
    python3 ~/rl_walk.py --onnx ~/policy_v46/policy.onnx --joy

## 왜 UDP 중계 말고 직접인가

`cmd_udp.py` 는 **심과 실기를 같은 입력으로 나란히 보려고** 만든 것이다.
조이스틱이 데스크톱에 꽂혀 있고 심이 읽어서 중계한다. 비교에는 맞지만
실기만 걷게 할 때는 대가가 크다:

  * WiFi 지터가 그대로 들어온다. 워치독이 300 ms 라 순간 끊김에 로봇이 선다.
  * 데스크톱에 Isaac Sim 이 떠 있어야 한다 (GPU 를 물고, 두 개 동시 실행 금지
    규칙에 걸린다).
  * 로봇이 책상에서 못 벗어난다.

직접 꽂으면 셋 다 없어지고, **정지 버튼이 손 안에 있다.** SSH 로 Ctrl+C 를
치는 것보다 항상 빠르다.

## 의존성이 없다

젯슨에는 pygame 도 evdev 도 없다. 리눅스 joystick API 는 8바이트 고정
구조(`__u32 time; __s16 value; __u8 type; __u8 number;`)라 struct 하나로
읽으면 된다 — `joy_monitor.py` 가 이미 그렇게 하고 있고 같은 방식이다.

## 안전

  * **데드맨.** RT 를 당기고 있을 때만 명령이 나간다. 놓으면 즉시 (0,0,0).
    손을 떼는 것이 가장 자연스러운 정지 동작이라 이걸 기본으로 둔다.
  * **연결 감시.** 무선 패드가 꺼지거나 블루투스가 끊기면 `/dev/input/js0`
    이 사라진다. 그걸 감지해 명령을 0 으로 만들고 `stale` 을 세운다.
    **조이스틱은 가만히 있으면 이벤트를 안 보내므로 "이벤트가 없다" 를
    끊김으로 쓸 수 없다** — 장치 존재 여부로 판단해야 한다.
  * **비상정지 래치.** A 버튼. 한 번 걸리면 재실행 전엔 안 풀린다
    (`cmd_udp` 와 같은 규약).
  * **범위 클램프.** 학습 명령 범위 밖은 정책이 본 적이 없다.

## 축 배치 (Xbox)

    0,1 왼쪽 스틱 X,Y      2,3 오른쪽 스틱 X,Y
    4,5 LT,RT (-32767=뗌)  6,7 십자키

    왼쪽 스틱 위/아래 -> vx (앞/뒤)      스틱 Y 는 위가 음수라 부호를 뒤집는다
    왼쪽 스틱 좌/우   -> vy (좌/우)
    오른쪽 스틱 좌/우 -> wz (제자리 회전)

컨트롤러가 안 붙으면 `joy_monitor.py` 로 먼저 확인할 것. 페어링 함정
(ERTM, joydev 모듈, 재연결마다 바뀌는 MAC)은 그 파일 독스트링에 있다.
"""
from __future__ import annotations

import os
import select
import struct
import threading
import time

DEV = "/dev/input/js0"
FMT = "<IhBB"
SZ = struct.calcsize(FMT)
FULL = 32767.0

#: 학습 명령 범위. cmd_udp.py 와 같은 값이어야 한다.
VX_RANGE = (-0.15, 0.15)
VY_RANGE = (-0.20, 0.20)
WZ_RANGE = (-1.0, 1.0)

# 축 배정. 2026-08-18 에 사용자 요청으로 좌우를 맞바꿨다.
#
#   전:  왼쪽 스틱 = 전후(1) + 게걸음(0),  오른쪽 스틱 좌우 = 회전(2)
#   후:  왼쪽 스틱 = 전후(1) + **회전**(0), 오른쪽 스틱 좌우 = **게걸음**(2)
#
# 사용자: "오른쪽 조이스틱이 회전처럼 동작하고 왼쪽 조이스틱을 좌우로 했을 때
# 회전이 안 된다. 둘이 기능 바꿔봐." 즉 차/전차식(왼손으로 방향, 오른손으로
# 횡이동)을 원한 것이다.
#
# ODM_JOY_AXES 로 덮어쓸 수 있다: `ODM_JOY_AXES=1,0,2` 면 예전 배정.
AX_VX, AX_VY, AX_WZ = 1, 2, 0
_env = os.environ.get("ODM_JOY_AXES")
if _env:
    AX_VX, AX_VY, AX_WZ = (int(x) for x in _env.split(","))
AX_DEADMAN = 5          # RT

# ── 버튼 번호는 연결 방식마다 다르다 ──────────────────────────────────
#
# 같은 Xbox 패드인데 USB(xpad 드라이버)로 붙으면 버튼이 **11 개**, 블루투스
# HID 로 붙으면 **15 개**이고 번호 배치도 다르다. 2026-08-13 실측: 이 젯슨에
# 블루투스로 붙은 패드는 축 8 개 / 버튼 15 개였다.
#
#   11 개 (USB xpad)   0=A 1=B 2=X 3=Y 4=LB 5=RB 6=Back 7=Start 8=Guide
#   15 개 (BT HID)     0=A 1=B 3=X 4=Y 6=LB 7=RB 10=Back 11=Start 12=Guide
#
# 그래서 하드코딩 하나로는 안 되고, 열 때 버튼 개수를 물어서 고른다.
# 그래도 패드마다 예외가 있을 수 있으니 `joy_monitor.py` 로 확인하고
# `--joy-map` 으로 덮을 수 있게 열어 둔다. 시작할 때 고른 표를 찍는다.

#: 리눅스 joystick API ioctl. _IOR('j', 0x11/0x12, __u8).
JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12

BUTTONS_XPAD = {"estop": 0, "start": 7, "pause": 6, "ready": 8}
BUTTONS_HID = {"estop": 0, "start": 11, "pause": 10, "ready": 12}


def probe_counts(dev: str = DEV) -> tuple[int, int]:
    """(축 개수, 버튼 개수). 못 읽으면 (0, 0)."""
    import array
    import fcntl
    try:
        fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return (0, 0)
    try:
        buf = array.array("B", [0])
        fcntl.ioctl(fd, JSIOCGAXES, buf, True)
        n_ax = buf[0]
        fcntl.ioctl(fd, JSIOCGBUTTONS, buf, True)
        return (n_ax, buf[0])
    except OSError:
        return (0, 0)
    finally:
        os.close(fd)


def default_buttons(dev: str = DEV) -> dict[str, int]:
    """버튼 개수를 보고 배치를 고른다."""
    _ax, n = probe_counts(dev)
    return dict(BUTTONS_HID if n >= 13 else BUTTONS_XPAD)


#: 기본 표 (개수를 못 물었을 때의 폴백). 실제로는 default_buttons() 가 고른다.
BUTTONS = BUTTONS_HID

#: 스틱이 중립에서 이만큼 안쪽이면 0 으로 본다 (32767 기준 약 15 %).
#: 오래 쓴 패드는 중립이 꽤 흔들린다 — 너무 작게 잡으면 손을 안 대도 로봇이 긴다.
DEADZONE = 5000
#: RT 가 이보다 크면 '당겼다'. 뗀 상태는 -32767 이다.
DEADMAN_ON = 0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _scale(raw: int, lo: float, hi: float) -> float:
    """축 원값(-32767..32767) -> 명령. 데드존 밖을 0 부터 다시 편다."""
    if abs(raw) <= DEADZONE:
        return 0.0
    span = FULL - DEADZONE
    u = (abs(raw) - DEADZONE) / span
    u = min(u, 1.0) * (1.0 if raw > 0 else -1.0)
    return _clamp(u * (hi if u > 0 else -lo), lo, hi)


class JoystickCommand:
    """로컬 조이스틱에서 (vx, vy, yaw). `cmd_udp.CommandReceiver` 와 같은 인터페이스."""

    def __init__(self, dev: str = DEV, deadman: bool = True,
                 buttons: dict[str, int] | None = None) -> None:
        self.dev = dev
        self.deadman = deadman
        #: 이름 -> 버튼번호. 반대 방향 조회를 미리 만들어 둔다.
        self.n_axes, self.n_buttons = probe_counts(dev)
        self.buttons = dict(default_buttons(dev) if buttons is None else buttons)
        self._by_num = {v: k for k, v in self.buttons.items()}
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._axes: dict[int, int] = {}
        self._down: set[int] = set()
        self._pressed: list[str] = []     # 아직 안 가져간 '눌림' 이벤트
        self._connected = False
        self._events = 0
        self._reconnects = 0
        self._held = False
        self.estopped = False
        self._stop = False

        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        print(f"[joy] {dev} 에서 명령을 읽는다"
              + ("  ·  RT 를 당기고 있을 때만 움직인다 (놓으면 정지)"
                 if deadman else "  ·  데드맨 없음"), flush=True)
        kind = ("BT HID" if self.n_buttons >= 13 else "USB xpad") \
            if self.n_buttons else "개수 불명"
        print(f"[joy] 축 {self.n_axes}개 · 버튼 {self.n_buttons}개 -> {kind} 배치", flush=True)
        print("[joy] 버튼 표: "
              + "  ".join(f"{k}={v}" for k, v in self.buttons.items())
              + "   (다르면 joy_monitor.py 로 확인 후 --joy-map 으로 덮을 것)",
              flush=True)

    # ── 내부 ──────────────────────────────────────────────────────────
    def _open(self):
        try:
            return os.open(self.dev, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return None

    def _loop(self) -> None:
        fd = None
        while not self._stop:
            if fd is None:
                fd = self._open()
                if fd is None:
                    with self._lock:
                        if self._connected:
                            print("\n[joy] !! 조이스틱이 끊겼다 — 명령을 0 으로 둔다",
                                  flush=True)
                        self._connected = False
                        self._cmd = (0.0, 0.0, 0.0)
                        self._held = False
                    time.sleep(0.25)
                    continue
                with self._lock:
                    self._connected = True
                    self._reconnects += 1
                    self._axes.clear()
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
            except OSError:
                os.close(fd)
                fd = None
                continue
            if r:
                closed = False
                while True:
                    try:
                        data = os.read(fd, SZ)
                    except BlockingIOError:
                        break
                    except OSError:
                        closed = True
                        break
                    if not data or len(data) < SZ:
                        closed = True
                        break
                    _tm, val, typ, num = struct.unpack(FMT, data)
                    base = typ & 0x7F
                    init = bool(typ & 0x80)
                    with self._lock:
                        if base == 2:
                            self._axes[num] = val
                        elif base == 1:
                            was = num in self._down
                            if val:
                                self._down.add(num)
                            else:
                                self._down.discard(num)
                            # **누르는 순간(엣지)만** 이벤트로 남긴다. 연결 직후
                            # 커널이 보내는 초기상태(init)는 사람이 누른 게
                            # 아니므로 세면 안 된다 — 켜자마자 로봇이 움직인다.
                            if val and not was and not init:
                                name = self._by_num.get(num)
                                if name == "estop":
                                    if not self.estopped:
                                        print("\n[joy] !! 비상정지 래치 "
                                              "(재실행해야 풀린다)", flush=True)
                                    self.estopped = True
                                self._pressed.append(name or f"btn{num}")
                        if not init:
                            self._events += 1
                if closed:
                    os.close(fd)
                    fd = None
                    continue
            # 장치 노드가 사라졌는지 본다. 패드가 조용해도(이벤트 없음) 연결은
            # 살아 있을 수 있으므로 이벤트 유무로는 판단하면 안 된다.
            if not os.path.exists(self.dev):
                os.close(fd)
                fd = None
                continue
            self._recompute()
        if fd is not None:
            os.close(fd)

    def _recompute(self) -> None:
        with self._lock:
            a = self._axes
            held = (not self.deadman) or (a.get(AX_DEADMAN, -32767) > DEADMAN_ON)
            if held != self._held:
                self._held = held
                print(f"\n[joy] {'출발 (RT 당김)' if held else '정지 (RT 놓음)'}", flush=True)
            if not held or self.estopped:
                self._cmd = (0.0, 0.0, 0.0)
                return
            self._cmd = (
                _scale(-a.get(AX_VX, 0), *VX_RANGE),   # 스틱 Y 는 위가 음수
                _scale(-a.get(AX_VY, 0), *VY_RANGE),   # 왼쪽으로 밀면 +y (로봇 좌측)
                _scale(-a.get(AX_WZ, 0), *WZ_RANGE),   # 왼쪽으로 밀면 +yaw (반시계)
            )

    # ── CommandReceiver 와 같은 공개 인터페이스 ───────────────────────
    def get(self) -> tuple[float, float, float]:
        with self._lock:
            if self.estopped or not self._connected:
                return (0.0, 0.0, 0.0)
            return self._cmd

    def poll_pressed(self) -> list[str]:
        """마지막 호출 이후 **새로 눌린** 버튼 이름들. 가져가면 비워진다."""
        with self._lock:
            out, self._pressed = self._pressed, []
            return out

    def held(self, name: str) -> bool:
        """지금 눌려 있는가 (엣지가 아니라 상태)."""
        with self._lock:
            n = self.buttons.get(name)
            return n is not None and n in self._down

    @property
    def deadman_held(self) -> bool:
        with self._lock:
            return self._held

    @property
    def stale(self) -> bool:
        """조이스틱이 안 붙어 있으면 True. 가만히 있는 것은 stale 이 아니다."""
        with self._lock:
            return not self._connected

    def stats(self) -> str:
        with self._lock:
            return (f"이벤트 {self._events} · 재연결 {max(self._reconnects-1,0)} · "
                    f"{'RT held' if self._held else 'RT 뗌'}"
                    + (" · ESTOP" if self.estopped else ""))

    def close(self) -> None:
        self._stop = True


if __name__ == "__main__":
    import sys
    joy = JoystickCommand(deadman="--no-deadman" not in sys.argv)
    print("Ctrl+C 로 종료. 스틱을 움직이고 버튼을 눌러 보라.\n")
    try:
        while True:
            vx, vy, wz = joy.get()
            for name in joy.poll_pressed():
                print(f"\n  버튼 눌림: {name}")
            sys.stdout.write(f"\r  vx {vx:+.3f}  vy {vy:+.3f}  wz {wz:+.3f}   "
                             f"{'STALE' if joy.stale else 'LIVE '}  {joy.stats():50s}")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        joy.close()
        print("\n종료")
