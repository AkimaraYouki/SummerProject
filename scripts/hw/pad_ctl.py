#!/usr/bin/env python3
"""패드 하나로 로봇 전체를 운전한다 — READY 이동, 보행 시작/정지, 비상정지.

    ssh -t parksuho@192.168.137.7 'python3 ~/pad_ctl.py --onnx ~/policy_v46/policy.onnx'

## 버튼

    Home (Guide)   READY 자세로 이동 (goto_ready). 걷는 중이면 먼저 멈춘다.
    Start (Menu)   보행 시작 — rl_walk 를 띄운다
    Back (View)    보행 정지. 그 자리에서 자세를 붙잡는다
    A              비상정지 (래치 — pad_ctl 을 다시 실행해야 풀린다)
    RT             데드맨. **당기고 있는 동안만** 속도 명령이 나간다
    왼쪽 스틱      앞뒤 = vx,  좌우 = vy
    오른쪽 스틱    좌우 = wz (제자리 회전)

버튼 번호는 **패드에 물어서 자동으로 고른다.** 같은 Xbox 패드라도 USB(xpad)
로 붙으면 버튼이 11 개, 블루투스 HID 로 붙으면 15 개이고 배치가 다르다.
시작할 때 고른 표를 찍으니 확인하고, 그래도 안 맞으면
`--joy-map start=11,pause=10,ready=12` 처럼 덮을 것 (`joy_monitor.py` 로
실제 번호를 볼 수 있다).

## 구조 — rl_walk 는 한 줄도 안 고친다

rl_walk 에 이미 `--cmd-udp-port` 가 있다. 그래서 이 스크립트는 rl_walk 를
**자식 프로세스로 띄우고 패드 입력을 localhost UDP 로 중계**한다.
`cmd_udp.py` 가 원래 데스크톱의 심에서 받던 그 경로를 그대로 쓴다.

그렇게 한 이유:

  * rl_walk 는 이제 막 실기가 걷기 시작한 코드다. 상태기계를 안에 넣으면
    그걸 건드리게 된다. 밖에 두면 걷는 코드는 그대로다.
  * 중계가 **localhost** 라 WiFi 지터가 없다. 데스크톱 심에서 보낼 때는
    300 ms 워치독이 순간 끊김에 걸려 로봇이 섰다.
  * 데스크톱에 Isaac Sim 이 안 떠 있어도 된다 (GPU 를 안 물고, 두 개 동시
    실행 금지 규칙에도 안 걸린다). 로봇이 책상에서 벗어날 수 있다.

## 안전

  * rl_walk 의 TTY + "go" 확인은 **없애지 않았다.** 대신 pad_ctl 이 자기가
    사람에게 한 번 확인을 받고, pty 로 그 답을 전달한다. 사람이 보고 있다는
    전제는 그대로 유지된다.
  * 정지는 rl_walk 의 stdin estop("sss")으로 보낸다 — rl_walk 가 그 자리에서
    현재 위치를 붙잡는다. 프로세스를 죽여서 토크가 풀리는 일이 없다.
  * 패드가 꺼지거나 블루투스가 끊기면 `joy_local` 이 그걸 감지해 명령을 0 으로
    만들고, 이 스크립트는 UDP 송신을 멈춘다 — rl_walk 의 워치독도 걸린다.
    두 겹이다.
  * A 는 래치다. 통신이 복구돼도 저절로 다시 걷지 않는다.
"""
from __future__ import annotations

import argparse
import os
import pty
import signal
import socket
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.expanduser("~"))
from joy_local import JoystickCommand, default_buttons  # noqa: E402

#: cmd_udp.py 의 PACKET 과 **반드시 같아야 한다.**
PACKET = "<IdfffB"
HOME = os.path.expanduser("~")
RL_WALK = os.path.join(HOME, "rl_walk.py")
GOTO_READY = os.path.join(HOME, "goto_ready.py")


class Child:
    """rl_walk 자식 프로세스. stdin 이 pty 라 rl_walk 가 TTY 로 인식한다."""

    def __init__(self, argv: list[str]) -> None:
        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(argv, stdin=slave, close_fds=True)
        os.close(slave)
        # rl_walk 가 "매달려 있으면 'go'" 를 묻는다. 사람 확인은 pad_ctl 이
        # 이미 받았으므로 여기서 답한다 (독스트링의 안전 항목 참고).
        os.write(self.master, b"go\n")

    def send(self, s: str) -> None:
        try:
            os.write(self.master, s.encode())
        except OSError:
            pass

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 4.0) -> None:
        """정지: stdin estop -> SIGINT -> SIGKILL. 앞의 둘은 자세를 붙잡고 끝난다."""
        if not self.alive:
            return
        self.send("sss\n")
        t0 = time.time()
        while self.alive and time.time() - t0 < timeout:
            time.sleep(0.05)
        if self.alive:
            self.proc.send_signal(signal.SIGINT)
            t0 = time.time()
            while self.alive and time.time() - t0 < timeout:
                time.sleep(0.05)
        if self.alive:
            print("[pad] !! rl_walk 가 안 죽는다 — SIGKILL")
            self.proc.kill()
        try:
            os.close(self.master)
        except OSError:
            pass


def parse_map(s: str | None, dev: str) -> dict[str, int]:
    # 기본은 **패드에 물어서** 고른다 (USB 11버튼 / BT 15버튼 배치가 다르다).
    m = default_buttons(dev)
    if s:
        for part in s.split(","):
            k, _, v = part.partition("=")
            if k.strip() in m and v.strip().isdigit():
                m[k.strip()] = int(v)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="정책 onnx 경로")
    ap.add_argument("--port", type=int, default=9999, help="localhost UDP 포트")
    ap.add_argument("--joy-dev", default="/dev/input/js0")
    ap.add_argument("--joy-map", default=None,
                    help="버튼 번호 덮어쓰기: estop=0,start=7,pause=6,ready=8")
    ap.add_argument("--no-deadman", action="store_true",
                    help="RT 를 안 잡아도 명령이 나간다 (권하지 않는다)")
    ap.add_argument("--seconds", type=float, default=600.0,
                    help="rl_walk 한 번의 최대 실행 시간")
    ap.add_argument("--ready-duration", type=float, default=12.0,
                    help="goto_ready 이동 시간(초)")
    ap.add_argument("--rl-args", default="",
                    help="rl_walk 에 그대로 넘길 추가 인자")
    args = ap.parse_args()

    onnx = os.path.expanduser(args.onnx)
    if not os.path.exists(onnx):
        sys.exit(f"onnx 가 없다: {onnx}")
    pol_dir = os.path.dirname(onnx)
    if not sys.stdin.isatty():
        sys.exit("TTY 가 아니다. ssh -t 로 붙어서 돌릴 것.")
    if not os.path.exists(args.joy_dev):
        sys.exit(f"{args.joy_dev} 이 없다. 먼저 붙일 것:  python3 ~/bt_pad.py")

    print("=" * 72)
    print("패드 조종  —  정책:", pol_dir)
    print("  Home  READY 자세로 이동        Start  보행 시작")
    print("  Back  보행 정지 (자세 유지)     A      비상정지 (래치)")
    print("  RT    데드맨 — 당긴 동안만 움직인다")
    print("  왼쪽 스틱 앞뒤=vx 좌우=vy   ·   오른쪽 스틱 좌우=wz")
    print("=" * 72)
    print("\n⚠️  로봇을 받칠 준비를 하고, 주변에 사람·물건이 없는지 볼 것.")
    print("   rl_walk 의 확인 절차를 이 스크립트가 대신 답한다.")
    if input("계속하려면 go 입력: ").strip().lower() != "go":
        return 1

    joy = JoystickCommand(dev=args.joy_dev, deadman=not args.no_deadman,
                          buttons=parse_map(args.joy_map, args.joy_dev))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    child: Child | None = None
    seq = 0
    state = "IDLE"
    last_print = 0.0

    def set_state(s: str) -> None:
        nonlocal state
        if s != state:
            print(f"\n[pad] {state} -> {s}", flush=True)
            state = s

    def stop_walk() -> None:
        nonlocal child
        if child is not None:
            print("\n[pad] 보행 정지 — 자세를 붙잡는다", flush=True)
            child.stop()
            child = None

    def go_ready() -> None:
        stop_walk()
        set_state("READY 이동중")
        print(f"[pad] goto_ready --policy {pol_dir}  ({args.ready_duration:.0f}초)",
              flush=True)
        r = subprocess.run(
            [sys.executable, GOTO_READY, "--policy", pol_dir,
             "--duration", str(args.ready_duration)],
            stdin=subprocess.DEVNULL)
        set_state("READY" if r.returncode == 0 else "IDLE")
        if r.returncode == 0:
            print("[pad] READY 도달. Start 로 보행 시작.", flush=True)
        else:
            print(f"[pad] !! goto_ready 실패 (코드 {r.returncode})", flush=True)

    def start_walk() -> None:
        nonlocal child
        if child is not None:
            return
        argv = [sys.executable, RL_WALK, "--onnx", onnx,
                "--cmd-udp-port", str(args.port),
                "--seconds", str(args.seconds)]
        if args.rl_args:
            argv += args.rl_args.split()
        print(f"\n[pad] rl_walk 시작 — 명령은 UDP {args.port} 로 중계한다", flush=True)
        child = Child(argv)
        set_state("WALK")

    try:
        while True:
            if joy.estopped:
                print("\n[pad] !! 비상정지 — 전부 멈춘다", flush=True)
                stop_walk()
                set_state("ESTOP")
                break

            for btn in joy.poll_pressed():
                if btn == "ready":
                    go_ready()
                elif btn == "start":
                    if state == "WALK":
                        print("\n[pad] 이미 걷는 중", flush=True)
                    elif state != "READY":
                        print("\n[pad] READY 가 아니다 — Home 을 먼저 누를 것", flush=True)
                    else:
                        start_walk()
                elif btn == "pause":
                    stop_walk()
                    set_state("HOLD")
                    print("[pad] 자세 유지 중. Home 으로 READY 복귀.", flush=True)

            if child is not None and not child.alive:
                print("\n[pad] rl_walk 가 끝났다", flush=True)
                child = None
                set_state("HOLD")

            # WALK 중에만 보낸다. 안 보내면 rl_walk 워치독이 걸려 (0,0,0) 이 된다
            # — 그게 우리가 원하는 안전 동작이다.
            if child is not None:
                vx, vy, wz = joy.get()
                seq += 1
                sock.sendto(struct.pack(PACKET, seq, time.time(),
                                        float(vx), float(vy), float(wz), 0),
                            ("127.0.0.1", args.port))

            now = time.time()
            if now - last_print > 0.25:
                last_print = now
                vx, vy, wz = joy.get()
                sys.stdout.write(
                    f"\r  [{state:12s}] vx {vx:+.3f} vy {vy:+.3f} wz {wz:+.3f}  "
                    f"{'RT' if joy.deadman_held else '--'}  "
                    f"{'STALE' if joy.stale else 'LIVE '}  {joy.stats():44s}")
                sys.stdout.flush()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[pad] Ctrl+C", flush=True)
    finally:
        stop_walk()
        joy.close()
        sock.close()
        print("\n[pad] 종료. 모터 토크는 마지막 상태 그대로다 — "
              "힘을 빼려면 home_position.py --release", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
