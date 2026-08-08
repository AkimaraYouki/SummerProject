#!/usr/bin/env python3
"""발바닥 접촉 스위치 (GPIO 31=left, 29=right). Jetson.GPIO 버전.

2026-08-09: 처음엔 29=left/31=right로 배선했다고 가정했는데 실기로 확인하니
반대였다(사용자 확인) — 물리적으로 GPIO31이 왼발, GPIO29가 오른발에 붙어있다.

원본 리포(Open_Duck_Mini_Runtime)의 feet_contacts.py 는 라즈베리파이용
`digitalio`/`board` (Blinka) 로 짜여 있어 Jetson 에서 못 쓴다. 여기선 이미
foot_contact_test.py 로 검증된 Jetson.GPIO 방식을 그대로 쓴다.

Active-low: 평소 HIGH(안 밟음), 밟으면 LOW. 외부 풀업 저항 필요
(Jetson.GPIO 는 setup() 의 pull_up_down 을 무시한다 — 소프트웨어로 켤 수 없음).

get() 은 제어 루프에서 매 틱(예: 50 Hz)마다 불러야 한다 — 내부 디바운스가
"직전 호출 이후 흐른 실제 시간"을 기준으로 판정하기 때문에, 호출 주기가
DEBOUNCE_SEC 보다 훨씬 길면(예: 1Hz) 접촉 판정이 한 틱 이상 늦게 반영될 수 있다.
"""

import time

import Jetson.GPIO as GPIO

LEFT = 31
RIGHT = 29

DEBOUNCE_SEC = 0.03  # 이 시간 이상 안정돼야 실제 변화로 인정 (foot_contact_test.py 로 검증됨)


class FeetContacts:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(LEFT, GPIO.IN)
        GPIO.setup(RIGHT, GPIO.IN)

        now = time.monotonic()
        self._l_stable = self._l_raw = GPIO.input(LEFT)
        self._r_stable = self._r_raw = GPIO.input(RIGHT)
        self._l_t = self._r_t = now

    def _debounced(self, pin, stable, raw, change_t, now):
        cur = GPIO.input(pin)
        if cur != raw:
            change_t = now
            raw = cur
        elif cur != stable and (now - change_t) >= DEBOUNCE_SEC:
            stable = cur
        return stable, raw, change_t

    def get(self):
        """[left_pressed, right_pressed] (bool). Active-low 이므로 LOW==눌림."""
        now = time.monotonic()
        self._l_stable, self._l_raw, self._l_t = self._debounced(
            LEFT, self._l_stable, self._l_raw, self._l_t, now)
        self._r_stable, self._r_raw, self._r_t = self._debounced(
            RIGHT, self._r_stable, self._r_raw, self._r_t, now)
        return [not self._l_stable, not self._r_stable]

    def stop(self):
        GPIO.cleanup()


if __name__ == "__main__":
    fc = FeetContacts()
    try:
        while True:
            print(fc.get())
            time.sleep(0.05)
    finally:
        fc.stop()
