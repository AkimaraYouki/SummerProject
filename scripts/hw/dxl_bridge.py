#!/usr/bin/env python3
"""관절 목표를 TCP 로 받아 실기에 쓰고 실측을 돌려주는 데몬 (Jetson).

    ssh -t parksuho@192.168.137.7 'python3 ~/dxl_bridge.py --arm'
    ssh    parksuho@192.168.137.7 'python3 ~/dxl_bridge.py --dry-run'   # 모터 안 건드림

프로토콜: 줄바꿈으로 끊은 JSON, 한 줄 요청 -> 한 줄 응답.
    보냄  {"q": [14개 rad, ACTUATOR_JOINT_NAMES 순서]}
    받음  {"q": [실측 14개 rad], "tick": [...], "clamped": [...], "ok": true}

`q` 를 빼고 보내면 쓰지 않고 읽기만 한다 (`{"read": true}`).

────────────────────────────────────────────────────────────────────────
⚠️  로봇을 **매달아 놓고** 쓸 것. 웹 슬라이더로 실물이 움직인다.
────────────────────────────────────────────────────────────────────────
안전 장치 네 겹:

  1. **URDF 한계 클램프** — 범위 밖 명령은 잘린다. 응답의 `clamped` 로 알려준다.
  2. **슬루 제한** (기본 60 °/s) — 슬라이더를 끝까지 확 당겨도 목표가 한 번에
     점프하지 않고 그 속도로 따라간다. 이게 이 도구의 핵심 안전장치다.
     원격에서 오는 값은 언제든 튈 수 있으므로 **여기서** 막아야 한다.
  3. **전류 제한** (기본 350 unit = 0.94 A, 스톨 2.3 A 의 41 %) — 관절이 걸리면
     밀어붙이지 못하고 진다.
  4. **데드맨** (기본 0.5 s) — 클라이언트가 끊기거나 조용하면 그 자리에서 홀드.

Ctrl+C / SIGTERM / SIGHUP 이면 현재 위치를 목표로 덮어쓰고 종료한다.
"""

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time

from dynamixel_sdk import (PortHandler, PacketHandler, GroupSyncWrite,
                           GroupSyncRead, COMM_SUCCESS)

BAUD = 1_000_000
TICK_RAD = 2.0 * math.pi / 4096.0
CENTER = 2048

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
MODE_CURRENT_POSITION = 5
CURRENT_UNIT_MA = 2.69

# (관절, ID, 부호, URDF 하한, 상한). joint_calibration.json 을 실기에서 확정한 값.
# goto_ready.py / play_ref_gait.py 와 같은 표다 -- Jetson 에는 패키지가 없어
# 각 스크립트가 자립형이다.
JOINTS = [
    ("left_hip_yaw",     3, -1, -0.5236, 0.5236),
    ("left_hip_roll",    8, -1, -0.4363, 0.4363),
    ("left_hip_pitch",   9, -1, -0.5236, 1.2217),
    ("left_knee",       10, +1, -2.0944, 2.0944),
    ("left_ankle",      11, +1, -1.5708, 1.5708),
    ("neck_pitch",       2, +1, -0.3491, 1.1345),
    ("head_pitch",      12, -1, -0.7854, 0.7854),
    ("head_yaw",        13, -1, -2.7925, 2.7925),
    ("head_roll",       14, -1, -0.5236, 0.5236),
    ("right_hip_yaw",    1, -1, -0.5236, 0.5236),
    ("right_hip_roll",   4, +1, -0.4363, 0.4363),
    ("right_hip_pitch",  5, +1, -0.5236, 1.2217),
    ("right_knee",       6, -1, -2.0944, 2.0944),
    ("right_ankle",      7, -1, -1.5708, 1.5708),
]
NAMES = [j[0] for j in JOINTS]
IDS = [j[1] for j in JOINTS]

_stop = {"v": False}


def tick_of(k, rad):
    _, _, sign, lo, hi = JOINTS[k]
    a, b = CENTER + sign * lo / TICK_RAD, CENTER + sign * hi / TICK_RAD
    lo_t, hi_t = int(round(min(a, b))), int(round(max(a, b)))
    return max(lo_t, min(hi_t, int(round(CENTER + sign * rad / TICK_RAD))))


def rad_of(k, tick):
    return JOINTS[k][2] * (tick - CENTER) * TICK_RAD


class Bridge:
    def __init__(self, args):
        self.a = args
        self.cmd = [0.0] * 14          # 클라이언트가 요청한 목표
        self.out = [0.0] * 14          # 슬루 제한을 거친 실제 목표
        self.meas = [0.0] * 14
        self.clamped = [False] * 14
        self.last_rx = 0.0
        self.lock = threading.Lock()
        self.port = self.packet = None

    # ── 하드웨어 ──────────────────────────────────────────────────────
    def open(self):
        self.port, self.packet = PortHandler(self.a.port), PacketHandler(2.0)
        if not self.port.openPort() or not self.port.setBaudRate(BAUD):
            raise SystemExit(f"포트 열기 실패: {self.a.port}")
        self.read_now()
        with self.lock:
            self.cmd = list(self.meas)
            self.out = list(self.meas)
        for i in IDS:
            self.packet.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 0)
            self.packet.write1ByteTxRx(self.port, i, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
            self.packet.write2ByteTxRx(self.port, i, ADDR_GOAL_CURRENT, self.a.current)
            self.packet.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 1)
        print(f"토크 켬 14축 · 전류 상한 {self.a.current * CURRENT_UNIT_MA/1000:.2f} A · "
              f"슬루 {self.a.slew:.0f} °/s · 데드맨 {self.a.deadman:.1f} s")

    def read_now(self):
        gr = GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, 4)
        for i in IDS:
            gr.addParam(i)
        if gr.txRxPacket() != COMM_SUCCESS:
            return False
        m = []
        for k, i in enumerate(IDS):
            t = gr.getData(i, ADDR_PRESENT_POSITION, 4)
            m.append(rad_of(k, t))
        with self.lock:
            self.meas = m
        return True

    def write_now(self):
        gw = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 4)
        with self.lock:
            tgt = list(self.out)
        for k, i in enumerate(IDS):
            t = tick_of(k, tgt[k])
            gw.addParam(i, bytes([t & 0xFF, (t >> 8) & 0xFF,
                                  (t >> 16) & 0xFF, (t >> 24) & 0xFF]))
        gw.txPacket()
        gw.clearParam()

    # ── 제어 루프 ────────────────────────────────────────────────────
    def loop(self):
        dt = 1.0 / self.a.rate
        slew = math.radians(self.a.slew)
        next_read = 0.0
        while not _stop["v"]:
            t0 = time.time()
            with self.lock:
                stale = (t0 - self.last_rx) > self.a.deadman and self.last_rx > 0
                goal = list(self.meas) if stale else list(self.cmd)
                # 슬루 제한: 목표를 한 스텝에 slew*dt 이상 못 움직이게 한다.
                step = slew * dt
                self.out = [o + max(-step, min(step, g - o))
                            for o, g in zip(self.out, goal)]
            if not self.a.dry_run:
                self.write_now()
            if t0 >= next_read:
                next_read = t0 + self.a.read_period
                if not self.a.dry_run:
                    self.read_now()
            time.sleep(max(0.0, dt - (time.time() - t0)))

    # ── TCP ──────────────────────────────────────────────────────────
    def serve(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.a.host, self.a.port_tcp))
        s.listen(1)
        s.settimeout(1.0)
        print(f"대기 중 tcp://{self.a.host}:{self.a.port_tcp}")
        while not _stop["v"]:
            try:
                conn, addr = s.accept()
            except socket.timeout:
                continue
            print(f"접속 {addr[0]}")
            self.session(conn)
            print("연결 종료 — 홀드")
        s.close()

    def session(self, conn):
        conn.settimeout(2.0)
        buf = b""
        with conn:
            while not _stop["v"]:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        req = json.loads(line)
                    except ValueError:
                        continue
                    self.handle(conn, req)

    def handle(self, conn, req):
        with self.lock:
            if isinstance(req.get("q"), list) and len(req["q"]) == 14:
                cl = []
                q = []
                for k, v in enumerate(req["q"]):
                    lo, hi = JOINTS[k][3], JOINTS[k][4]
                    c = max(lo, min(hi, float(v)))
                    cl.append(abs(c - float(v)) > 1e-9)
                    q.append(c)
                self.cmd, self.clamped = q, cl
                self.last_rx = time.time()
            resp = {"q": [round(v, 5) for v in self.meas],
                    "out": [round(v, 5) for v in self.out],
                    "tick": [tick_of(k, self.out[k]) for k in range(14)],
                    "clamped": self.clamped, "names": NAMES, "ok": True}
        try:
            conn.sendall((json.dumps(resp) + "\n").encode())
        except OSError:
            pass

    def hold(self):
        if self.a.dry_run or self.port is None:
            return
        if self.read_now():
            with self.lock:
                self.out = list(self.meas)
            self.write_now()
        print("현재 위치 홀드. 토크 해제는 python3 ~/home_position.py --release")
        self.port.closePort()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port-tcp", type=int, default=5599)
    ap.add_argument("--rate", type=float, default=20.0, help="제어 루프 Hz")
    ap.add_argument("--read-period", type=float, default=0.25,
                    help="실측을 몇 초마다 읽을지. 14축 SyncRead 는 64 ms 라 매 루프는 무리다")
    ap.add_argument("--slew", type=float, default=60.0, help="최대 각속도 °/s")
    ap.add_argument("--current", type=int, default=350)
    ap.add_argument("--deadman", type=float, default=0.5,
                    help="이 시간 동안 명령이 없으면 그 자리 홀드")
    ap.add_argument("--arm", action="store_true", help="실제로 모터를 움직인다")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.arm or args.dry_run):
        raise SystemExit("실물이 움직인다. 로봇을 매달아 놓고 --arm 을 명시할 것 "
                         "(모터 없이 프로토콜만 보려면 --dry-run).")

    b = Bridge(args)
    if not args.dry_run:
        b.open()
    else:
        print("[dry-run] 모터를 열지 않는다. TCP 프로토콜만 돈다.")

    def stop(*_):
        _stop["v"] = True
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)

    th = threading.Thread(target=b.loop, daemon=True)
    th.start()
    try:
        b.serve()
    finally:
        _stop["v"] = True
        th.join(timeout=2.0)
        b.hold()


if __name__ == "__main__":
    main()
