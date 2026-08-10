#!/usr/bin/env python3
"""심(play_fixed_cmd.py --cmd-udp)이 보내는 속도 명령을 받는다.

목적은 **같은 입력으로 심과 실기를 나란히 보는 것**이다. 조이스틱은 데스크톱에
꽂혀 있으므로 심이 읽어서 여기로 중계한다.

안전이 이 파일의 존재 이유다. 네트워크로 로봇을 움직이는 이상, 끊겼을 때
로봇이 마지막 명령을 붙들고 계속 걷는 일이 있으면 안 된다. 그래서:

  * **워치독** — `timeout` 동안 패킷이 없으면 명령이 (0,0,0) 으로 떨어진다.
    WiFi 가 끊기든 심이 죽든 조이스틱을 뽑든 전부 같은 결과다.
  * **비상정지 비트** — 패드 A 를 누르면 estop=1 이 와서 latch 된다. 한 번
    걸리면 풀리지 않는다(재실행해야 한다) — 통신이 복구됐다고 로봇이 저절로
    다시 걷기 시작하면 안 된다.
  * **범위 클램프** — 심이 이미 자르고 보내지만 여기서 한 번 더 자른다.
    학습 범위 밖 명령은 정책이 본 적이 없다.
  * UDP 는 순서를 보장하지 않으므로 **seq 가 역행하는 패킷은 버린다.**

사용 (rl_walk.py 안):

    from cmd_udp import CommandReceiver
    rx = CommandReceiver(port=9999) if args.cmd_udp_port else None
    ...
    if rx is not None:
        vx, vy, wz = rx.get()          # 워치독 걸리면 (0,0,0)
        if rx.estopped:
            break                      # 즉시 홀드로 빠진다
        command[0], command[1], command[2] = vx, vy, wz
"""
from __future__ import annotations

import socket
import struct
import threading
import time

#: play_fixed_cmd.py `_send_cmd` 와 **반드시 같아야 한다**.
#: seq(uint32) · 보낸시각(double) · vx · vy · yaw(float) · estop(uint8)
PACKET = "<IdfffB"
PACKET_SIZE = struct.calcsize(PACKET)   # 25

#: 학습 명령 범위 (joystick_env_cfg 의 lin_vel_x/y_range, ang_vel_yaw_range).
#: 여기 밖은 정책이 본 적 없는 입력이다.
VX_RANGE = (-0.15, 0.15)
VY_RANGE = (-0.20, 0.20)
WZ_RANGE = (-1.0, 1.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CommandReceiver:
    """UDP 로 (vx, vy, yaw) 를 받는다. 끊기면 정지."""

    def __init__(self, port: int = 9999, timeout: float = 0.3,
                 bind: str = "0.0.0.0") -> None:
        #: timeout 기본 0.3 s = 50 Hz 기준 15 패킷. WiFi 순간 끊김은 넘기고
        #: 진짜 단절은 잡는 값이다. 더 줄이면 무선에서 오작동으로 자꾸 선다.
        self.timeout = timeout
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, port))
        self._sock.settimeout(0.2)

        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._last_rx = 0.0          # 마지막으로 유효 패킷을 받은 시각
        self._seq = 0
        self._n = 0                  # 받은 패킷 수
        self._dropped = 0            # 순서 역행으로 버린 수
        self.estopped = False
        self._stop = False

        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        print(f"[cmd_udp] {bind}:{port} 에서 명령 수신 대기 "
              f"(워치독 {timeout*1000:.0f} ms)", flush=True)

    def _loop(self) -> None:
        while not self._stop:
            try:
                data, _ = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(data) != PACKET_SIZE:
                continue
            seq, _sent, vx, vy, wz, estop = struct.unpack(PACKET, data)
            # UDP 는 순서를 보장하지 않는다. 늦게 도착한 옛 패킷이 최신 명령을
            # 덮어쓰면 로봇이 한 박자 뒤로 튄다.
            #
            # 단, **끊겼다가 다시 들어오는 경우는 예외**다. 심을 재시작하면 seq 가
            # 1 부터 다시 시작하는데, 그걸 전부 역행으로 버리면 로봇이 영구히
            # 먹통이 된다(워치독으로 서 있기만 하고 재실행 전엔 안 풀린다).
            # 이미 stale 이면 워치독이 명령을 0 으로 만들어 둔 뒤이므로, 새 흐름을
            # 받아들이고 seq 를 다시 맞추는 게 안전하다.
            with self._lock:
                resync = (self._n == 0
                          or (time.time() - self._last_rx) > self.timeout)
            if not resync and (seq - self._seq) % (1 << 32) > (1 << 31):
                self._dropped += 1
                continue
            with self._lock:
                self._seq = seq
                self._n += 1
                self._last_rx = time.time()
                if estop:
                    self.estopped = True     # latch — 스스로 안 풀린다
                self._cmd = (_clamp(vx, *VX_RANGE),
                             _clamp(vy, *VY_RANGE),
                             _clamp(wz, *WZ_RANGE))

    def get(self) -> tuple[float, float, float]:
        """최신 명령. 워치독이 걸렸거나 estop 이면 (0,0,0)."""
        with self._lock:
            if self.estopped:
                return (0.0, 0.0, 0.0)
            if self._n == 0 or (time.time() - self._last_rx) > self.timeout:
                return (0.0, 0.0, 0.0)
            return self._cmd

    @property
    def stale(self) -> bool:
        with self._lock:
            return self._n == 0 or (time.time() - self._last_rx) > self.timeout

    def stats(self) -> str:
        with self._lock:
            age = (time.time() - self._last_rx) * 1000 if self._n else -1
            return (f"패킷 {self._n} · 역행버림 {self._dropped} · "
                    f"마지막 {age:.0f} ms 전" + (" · ESTOP" if self.estopped else ""))

    def close(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    # 단독 실행하면 수신만 해서 찍는다 — 로봇 없이 배선을 먼저 확인할 때 쓴다.
    #   잿슨:    python3 cmd_udp.py 9999
    #   데스크톱: odm play v41 --joystick --cmd-udp <잿슨IP>:9999
    import sys

    rx = CommandReceiver(port=int(sys.argv[1]) if len(sys.argv) > 1 else 9999)
    try:
        while True:
            time.sleep(0.25)
            vx, vy, wz = rx.get()
            print(f"  vx {vx:+.3f}  vy {vy:+.3f}  yaw {wz:+.3f}   "
                  f"{'STALE ' if rx.stale else ''}{rx.stats()}", flush=True)
    except KeyboardInterrupt:
        rx.close()
