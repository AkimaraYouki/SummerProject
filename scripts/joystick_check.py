"""조이스틱이 실제로 잡히는지, 어느 축이 어느 스틱인지 눈으로 확인한다.

Isaac Sim도 GPU도 안 쓴다 — 학습이 도는 중에도 그냥 돌려도 된다.
축 번호는 드라이버/패드 모델에 따라 다를 수 있으므로, 매핑이 이상하면 여기서
실제 번호를 확인한 뒤 `joystick_input.py`의 AXIS_* 상수를 고치면 된다.

    python3 scripts/joystick_check.py

Ctrl-C로 끝낸다.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.joystick_input import (  # noqa: E402
    Gamepad,
    GamepadUnavailable,
    command_from_gamepad,
)

# v25/v26(Path 태스크)이 학습에 쓴 명령 범위. 스틱을 끝까지 밀었을 때 이 값에
# 닿는지 보려고 그대로 적어 둔다 (joystick_env_cfg.py::JoystickEnvCfg).
X_RANGE = (-0.15, 0.15)
Y_RANGE = (-0.2, 0.2)
YAW_RANGE = (-1.0, 1.0)

BAR = 20


def _bar(v):
    """-1..1 을 막대로. 가운데가 0."""
    n = int(round(v * BAR))
    left = "#" * -n if n < 0 else " " * 0
    cells = [" "] * (2 * BAR + 1)
    cells[BAR] = "|"
    for i in range(min(abs(n), BAR)):
        cells[BAR + (i + 1 if n > 0 else -(i + 1))] = "#"
    return "".join(cells) + (f" {v:+.2f}" if abs(v) > 1e-9 else "  0.00") + left[:0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="/dev/input/js0")
    p.add_argument("--raw", action="store_true", help="모든 축/버튼 번호를 그대로 보여준다")
    args = p.parse_args()

    if not os.path.exists(args.device):
        print(f"!! {args.device} 가 없습니다.")
        print("   유선 Xbox 패드는 꽂으면 커널 xpad 드라이버가 바로 잡습니다.")
        print("   확인:  ls /dev/input/js*")
        print("   무선(블루투스)이면 먼저 페어링해야 합니다.")
        return 1

    try:
        pad = Gamepad(args.device)
    except GamepadUnavailable as exc:
        print(f"!! {exc}")
        return 1

    print(f"연결됨: {args.device}   (Ctrl-C로 종료)")
    print("왼쪽 스틱=전후·좌우, 오른쪽 스틱 X=회전, A=비상정지\n")

    try:
        while True:
            pad.poll()
            if not pad.connected:
                print("\n!! 장치가 끊겼습니다 (케이블/배터리 확인).")
                return 1

            vx, vy, yaw = command_from_gamepad(pad, X_RANGE, Y_RANGE, YAW_RANGE)
            stop = pad.button(0)

            if args.raw:
                axes = " ".join(f"{n}:{pad.axis(n):+.2f}" for n in sorted(pad._axes))
                btns = " ".join(str(n) for n, on in sorted(pad._buttons.items()) if on)
                out = f"축 [{axes}]  버튼 [{btns}]"
            else:
                out = (
                    f"vx {vx:+.3f} m/s  vy {vy:+.3f} m/s  yaw {yaw:+.3f} rad/s"
                    f"{'   [A: 비상정지]' if stop else ''}"
                )
            print(f"\r{out:<110}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        pad.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
