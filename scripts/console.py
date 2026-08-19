#!/usr/bin/env python3
"""키보드로 조종하고 로봇 상태를 한 화면에서 보는 터미널 대시보드.

    # PC 에서 (WebRTC 로 화면을 내보내며 명령을 받는다)
    ODM_HOST_IP=100.108.1.35 ./scripts/odm play v70 --stream --key

    # 노트북에서 SSH 로 붙어
    python3 scripts/console.py

## 왜

WebRTC 는 화면만 보낸다 — 노트북에서 보면서 조향할 방법이 없었다. 조이스틱은
PC 에 물려 있어야 하고. 그래서 명령을 UDP 로 받고 상태를 UDP 로 되돌리는 길을
열고, 이 스크립트가 그 양쪽을 맡는다. 로컬호스트든 Tailscale 이든 같다.

의존성이 없다 — 표준 라이브러리 `curses` 와 `socket` 만 쓴다. 젯슨에서도
같은 스크립트로 실기를 몰 수 있게 하려는 것이다 (실기는 rl_walk 의
`--cmd-udp-port` 가 같은 형식을 받는다).

## 키

    W / S      전진 / 후진
    A / D      좌 / 우 (게걸음)
    Q / E      좌회전 / 우회전
    Z / X      입력 스로틀 -10 %% / +10 %%
    Space      즉시 정지 (명령 0)
    Tab        연속(홀드) / 순간(탭) 전환
    Ctrl-C     종료

기본은 **홀드**다. 누르고 있는 동안만 그 방향으로 간다 — 터미널은 키를 뗀
것을 알려 주지 않으므로 `HOLD_MS` 안에 같은 키가 다시 안 오면 0 으로 돌린다.
"""
from __future__ import annotations

import argparse
import curses
import json
import math
import socket
import time

#: 학습 명령 범위. joystick_env_cfg 와 같아야 한다.
VX, VY, WZ = 0.15, 0.20, 1.0
#: 키를 뗀 것으로 볼 시간. 터미널 키 반복 속도(보통 30 ms)보다 넉넉해야 한다.
HOLD_MS = 220
#: 명령 송신 주기.
SEND_HZ = 50


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def bar(v, lo, hi, w=18):
    """-1~+1 을 가운데가 0 인 막대로."""
    if hi == lo:
        return " " * w
    mid = w // 2
    n = int(round(clamp(v / max(abs(lo), abs(hi)), -1, 1) * mid))
    s = [" "] * w
    s[mid] = "|"
    for i in range(1, abs(n) + 1):
        j = mid + i if n > 0 else mid - i
        if 0 <= j < w:
            s[j] = "█"
    return "".join(s)


def main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        for i, c in enumerate((curses.COLOR_GREEN, curses.COLOR_YELLOW,
                               curses.COLOR_RED, curses.COLOR_CYAN), 1):
            curses.init_pair(i, c, -1)
    OK, WARN, BAD, INFO = (curses.color_pair(i) for i in (1, 2, 3, 4))

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setblocking(False)
    rx.bind(("0.0.0.0", args.telem_port))
    dst = (args.host, args.port)

    throttle = 1.0
    hold = True
    last = {}          # 키 -> 마지막 입력 시각
    telem, telem_at = {}, 0.0
    sent = 0
    t0 = time.time()

    while True:
        now = time.time()
        # ── 키 ────────────────────────────────────────────────────────
        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            k = chr(ch).lower() if 0 <= ch < 256 else ""
            if ch in (3, 27):                      # Ctrl-C / ESC
                return
            if k == " ":
                last.clear()
            elif k == "z":
                throttle = clamp(throttle - 0.1, 0.1, 1.0)
            elif k == "x":
                throttle = clamp(throttle + 0.1, 0.1, 1.0)
            elif ch == 9:                          # Tab
                hold = not hold
                last.clear()
            elif k in "wasdqe":
                last[k] = now

        def on(k):
            if k not in last:
                return False
            if hold and (now - last[k]) * 1000 > HOLD_MS:
                del last[k]
                return False
            if not hold and (now - last[k]) > 0.35:
                del last[k]
                return False
            return True

        cx = (VX if on("w") else 0.0) - (VX if on("s") else 0.0)
        cy = (VY if on("a") else 0.0) - (VY if on("d") else 0.0)
        cw = (WZ if on("q") else 0.0) - (WZ if on("e") else 0.0)
        cx, cy, cw = cx * throttle, cy * throttle, cw * throttle
        tx.sendto(f"{cx:.4f},{cy:.4f},{cw:.4f}".encode(), dst)
        sent += 1

        # ── 상태 ──────────────────────────────────────────────────────
        while True:
            try:
                data, _ = rx.recvfrom(2048)
            except (BlockingIOError, OSError):
                break
            try:
                telem = json.loads(data.decode())
                telem_at = now
            except (ValueError, UnicodeDecodeError):
                pass
        fresh = (now - telem_at) < 1.0

        # ── 그리기 ────────────────────────────────────────────────────
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        r = 0

        def line(s, attr=0):
            nonlocal r
            if r < h - 1:
                stdscr.addnstr(r, 0, s, w - 1, attr)
            r += 1

        task = telem.get("task", "?")
        line(f" ODM 콘솔   {task:<18} {dst[0]}:{dst[1]} <- 명령   :{args.telem_port} -> 상태", INFO)
        line(f" {'연결됨' if fresh else '상태 없음 (play 가 --telem 으로 떠 있나?)':<40}"
             f" 스로틀 {throttle*100:3.0f} %   모드 {'HOLD' if hold else 'TAP'}",
             OK if fresh else BAD)
        line("")
        line(" 명령", INFO)
        line(f"   전후 {cx:+6.3f}  {bar(cx, -VX, VX)}   W / S")
        line(f"   좌우 {cy:+6.3f}  {bar(cy, -VY, VY)}   A / D")
        line(f"   회전 {cw:+6.3f}  {bar(cw, -WZ, WZ)}   Q / E")
        line("")

        if telem:
            v = telem.get("vel", [0, 0, 0])
            line(" 실제", INFO)
            line(f"   전후 {v[0]:+6.3f}  {bar(v[0], -VX, VX)}   오차 {v[0]-cx:+.3f}")
            line(f"   좌우 {v[1]:+6.3f}  {bar(v[1], -VY, VY)}   오차 {v[1]-cy:+.3f}")
            line(f"   회전 {v[2]:+6.3f}  {bar(v[2], -WZ, WZ)}   오차 {v[2]-cw:+.3f}")
            line("")
            roll, pitch = telem.get("roll", 0), telem.get("pitch", 0)
            ra = BAD if abs(roll) > 25 else (WARN if abs(roll) > 12 else OK)
            pa = BAD if abs(pitch) > 25 else (WARN if abs(pitch) > 12 else OK)
            line(" 자세", INFO)
            line(f"   roll  {roll:+6.1f}도  {bar(roll, -30, 30)}", ra)
            line(f"   pitch {pitch:+6.1f}도  {bar(pitch, -30, 30)}", pa)
            c = telem.get("contact", [0, 0])
            line(f"   높이  {telem.get('h', 0)*1000:6.1f} mm    접지 "
                 f"{'L' if c[0] else '·'}{'R' if c[1] else '·'}    "
                 f"관절오차 최대 {telem.get('jerr', 0):.1f}도")
        else:
            line(" 상태 수신 대기...", WARN)

        line("")
        line(f" W/S 전후 · A/D 좌우 · Q/E 회전 · Z/X 스로틀 · Space 정지 · Tab 모드 · Ctrl-C 종료", INFO)
        line(f" 보낸 패킷 {sent}   {now-t0:.0f} 초", INFO)
        stdscr.refresh()
        time.sleep(1.0 / SEND_HZ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="명령을 보낼 주소 (play 가 도는 곳)")
    ap.add_argument("--port", type=int, default=9997, help="play 의 --cmd-listen 포트")
    ap.add_argument("--telem-port", type=int, default=9998, help="상태를 받을 포트")
    a = ap.parse_args()
    try:
        curses.wrapper(main, a)
    except KeyboardInterrupt:
        pass
    print("콘솔 종료")
