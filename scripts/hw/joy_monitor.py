#!/usr/bin/env python3
"""조이스틱 입력 실시간 모니터 (Jetson). 의존성 없음 — /dev/input/js0 직접 읽는다.

    python3 ~/joy_monitor.py

Jetson 에는 pygame/evdev 가 없고 apt 로 넣으려면 네트워크가 필요하다. 리눅스
joystick API 는 8바이트 고정 구조라(`__u32 time; __s16 value; __u8 type;
__u8 number;`) struct 하나로 읽으면 되므로 의존성을 안 만든다.

js0 은 world-readable(crw-rw-r--)이라 input 그룹이 아니어도 읽힌다.

2026-08-10 Xbox Wireless Controller 연결 시 걸렸던 것:
  * ERTM 을 꺼야 페어링이 붙는다 (/sys/module/bluetooth/parameters/disable_ertm).
    /etc/modprobe.d/99-xbox-ertm.conf 에 넣어 부팅 시 적용되게 해뒀다.
  * joydev 모듈이 있어야 /dev/input/js0 이 생긴다 (/etc/modules-load.d/joydev.conf).
  * 컨트롤러가 재연결마다 MAC 끝자리를 바꾼다 (...1A -> ...1B). 안 붙으면
    `bluetoothctl scan on` 으로 현재 주소를 다시 확인할 것.

축 배치 (Xbox):
    0,1 왼쪽 스틱 X,Y      2,3 오른쪽 스틱 X,Y
    4,5 LT,RT (-32767=뗌)  6,7 십자키
"""

import os
import select
import struct
import sys
import time

DEV = "/dev/input/js0"
FMT = "<IhBB"
SZ = struct.calcsize(FMT)
AXIS_NAME = {0: "L스틱X", 1: "L스틱Y", 2: "R스틱X", 3: "R스틱Y",
             4: "LT", 5: "RT", 6: "십자X", 7: "십자Y"}


def bar(v, width=21):
    """-32767..32767 를 가운데가 0 인 막대로."""
    mid = width // 2
    k = int(round(v / 32767.0 * mid))
    cells = ["·"] * width
    cells[mid] = "|"
    if k:
        for i in range(min(mid, mid + k), max(mid, mid + k) + 1):
            if 0 <= i < width:
                cells[i] = "="
    cells[max(0, min(width - 1, mid + k))] = "#"
    return "".join(cells)


def main():
    if not os.path.exists(DEV):
        sys.exit(f"{DEV} 이 없다. 컨트롤러가 연결됐는지 확인할 것:\n"
                 f"  bluetoothctl devices Connected")
    f = os.open(DEV, os.O_RDONLY | os.O_NONBLOCK)
    axes, btns = {}, {}
    n_events = 0
    last_draw = 0.0
    print("Ctrl+C 로 종료. 스틱/버튼을 움직여라.\n")
    try:
        while True:
            r, _, _ = select.select([f], [], [], 0.05)
            if r:
                while True:
                    try:
                        data = os.read(f, SZ)
                    except BlockingIOError:
                        break
                    if not data or len(data) < SZ:
                        break
                    _tm, val, typ, num = struct.unpack(FMT, data)
                    init = bool(typ & 0x80)
                    base = typ & 0x7F
                    if base == 2:
                        axes[num] = val
                    elif base == 1:
                        btns[num] = val
                    if not init:
                        n_events += 1
            now = time.time()
            if now - last_draw < 0.05:
                continue
            last_draw = now
            lines = [f"입력 이벤트 {n_events}개"]
            for k in sorted(axes):
                lines.append(f"  {AXIS_NAME.get(k, 'axis'+str(k)):>7} {axes[k]:+7d} {bar(axes[k])}")
            pressed = [str(k) for k in sorted(btns) if btns[k]]
            lines.append("  버튼: " + (" ".join(pressed) if pressed else "-"))
            out = "\n".join(lines)
            sys.stdout.write("\033[H\033[J" + out + "\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print(f"\n종료. 입력 이벤트 {n_events}개 받음.")
    finally:
        os.close(f)


if __name__ == "__main__":
    main()
