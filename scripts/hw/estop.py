#!/usr/bin/env python3
"""비상정지: 어느 콘솔에서 실행하든 14축 전부 토크를 즉시 끈다.

    python3 ~/estop.py

다른 스크립트(rl_walk.py 등)가 포트를 쓰고 있어도 별도 프로세스로 떠서
독립적으로 작동한다 (시리얼 포트는 GPIO 와 달리 여러 프로세스가 동시에 열 수
있다). "홀드"가 아니라 "토크 끔" 이다 — 로봇이 매달려 있을 때만 안전하다,
바닥에 서 있는 상태에서 쓰면 그대로 주저앉는다.
"""

import rustypot

PORT = "/dev/ttyUSB0"
BAUD = 1_000_000
IDS = [3, 8, 9, 10, 11, 2, 12, 13, 14, 1, 4, 5, 6, 7]


def main():
    c = rustypot.Xl430PyController(PORT, BAUD, 0.05)
    c.sync_write_torque_enable(IDS, [0] * 14)
    print("[estop] 전체 14축 토크 껐음")


if __name__ == "__main__":
    main()
