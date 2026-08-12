#!/usr/bin/env python3
"""젯슨에 Xbox 패드를 블루투스로 붙인다. 스캔 -> 페어링 -> 연결 -> js0 확인.

    ssh -t parksuho@192.168.137.7 'python3 ~/bt_pad.py'          # 스캔해서 붙인다
    ssh -t parksuho@192.168.137.7 'python3 ~/bt_pad.py --status' # 지금 상태만
    ssh -t parksuho@192.168.137.7 'python3 ~/bt_pad.py --forget' # 등록 지우고 처음부터

패드를 페어링 모드로: Xbox 버튼을 눌러 켠 뒤, 위쪽 작은 **연결 버튼**을
로고가 빠르게 깜빡일 때까지 3 초 누른다.

## 이 파일이 있는 이유

2026-08-10 에 붙일 때 걸렸던 것들을 매번 다시 헤매지 않으려고 절차로 굳혔다.

  * **ERTM 을 꺼야 페어링이 붙는다.** 리눅스 블루투스 스택의 ERTM 과 Xbox
    패드가 안 맞는다.

    2026-08-13 정정: 이걸 `/etc/modprobe.d/99-xbox-ertm.conf` 에 넣어 뒀는데
    **처음부터 효과가 없었다.** 이 젯슨은 bluetooth 가 커널 내장이라
    (`lsmod | grep bluetooth` 가 비어 있다) modprobe.d 가 적용되지 않는다.
    그래서 재부팅할 때마다 ERTM 이 켜진 채였고, "지난번엔 붙었는데 오늘은
    안 붙는다" 를 반복했다. 지금은
    `/etc/tmpfiles.d/xbox-ertm.conf` 가 부팅 때 sysfs 에 직접 쓴다.
  * **joydev 모듈이 있어야 `/dev/input/js0` 이 생긴다.** 모듈이 없으면
    페어링은 되는데 장치 노드가 안 나와서 "붙었는데 안 읽힌다" 가 된다
    (`/etc/modules-load.d/joydev.conf`).
  * **패드가 재연결마다 MAC 끝자리를 바꾼다** (…1A -> …1B). 그래서 주소를
    적어 두고 쓰면 다음에 안 붙는다 — 매번 스캔해서 찾아야 한다.

## 붙고 나서

    python3 ~/joy_monitor.py     # 축·버튼 번호 확인 (패드마다 다르다)
    python3 ~/joy_local.py       # 속도 명령으로 어떻게 변환되는지
    python3 ~/pad_ctl.py --onnx ~/policy_v46/policy.onnx    # 패드로 로봇 운전
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ERTM = "/sys/module/bluetooth/parameters/disable_ertm"
DEV = "/dev/input/js0"
#: 이름에 이게 들어가면 Xbox 패드로 본다.
NAME_HINTS = ("xbox", "controller", "gamepad", "wireless")


def sh(cmd: str, timeout: float = 15.0) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return ""


def bctl(*lines: str, wait: float = 2.0) -> str:
    """bluetoothctl 에 명령을 순서대로 넣고 출력을 받는다."""
    script = "\n".join(lines) + "\nquit\n"
    try:
        p = subprocess.run(["bluetoothctl"], input=script, capture_output=True,
                           text=True, timeout=wait + 10.0)
        return (p.stdout or "") + (p.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def check_prereqs() -> bool:
    ok = True
    if os.path.exists(ERTM):
        v = open(ERTM).read().strip()
        if v in ("Y", "1"):
            print(f"  ERTM 꺼짐 (disable_ertm={v})  OK")
        else:
            print(f"  !! ERTM 이 켜져 있다 (disable_ertm={v}) — 페어링이 실패한다.")
            print(f"     지금 끄기:  sudo sh -c 'echo Y > {ERTM}'")
            print(f"     영구 적용:  /etc/tmpfiles.d/xbox-ertm.conf 에")
            print(f"                 w {ERTM} - - - - Y")
            print(f"     (modprobe.d 는 안 먹는다 — 이 젯슨은 bluetooth 가 커널 내장이다)")
            ok = False
    else:
        print("  ERTM 노드 없음 — 블루투스 모듈이 안 올라왔을 수 있다")
    mods = sh("lsmod")
    if re.search(r"^joydev", mods, re.M):
        print("  joydev 모듈 있음  OK")
    else:
        print("  !! joydev 모듈이 없다 — 페어링돼도 /dev/input/js0 이 안 생긴다.")
        print("     sudo modprobe joydev   (영구: /etc/modules-load.d/joydev.conf)")
        ok = False
    if "yes" in sh("systemctl is-active bluetooth").lower() or \
       sh("systemctl is-active bluetooth").strip() == "active":
        print("  bluetooth 서비스 active  OK")
    else:
        print("  !! bluetooth 서비스가 안 돌고 있다 — sudo systemctl start bluetooth")
        ok = False
    return ok


def connected_pads() -> list[tuple[str, str]]:
    out = bctl("devices Connected")
    return [(m.group(1), m.group(2).strip())
            for m in re.finditer(r"Device ([0-9A-F:]{17}) (.+)", out)]


def status() -> None:
    print("=== 사전 조건 ===")
    check_prereqs()
    print("\n=== 연결된 장치 ===")
    pads = connected_pads()
    print("  " + ("\n  ".join(f"{a}  {n}" for a, n in pads) if pads else "없음"))
    print(f"\n=== {DEV} ===")
    if os.path.exists(DEV):
        print(f"  있음. joy_monitor.py 로 축/버튼을 확인할 것.")
    else:
        print("  없음 — 패드가 안 붙었거나 joydev 가 없다.")


def scan_and_connect(secs: float, name_filter: str | None) -> int:
    print("=== 사전 조건 ===")
    if not check_prereqs():
        print("\n위 항목을 먼저 고치고 다시 실행할 것.")
        return 1

    pads = connected_pads()
    if pads and os.path.exists(DEV):
        print(f"\n이미 붙어 있다: {pads[0][0]}  {pads[0][1]}")
        print(f"{DEV} 도 있다. 할 일 없음.")
        return 0

    print(f"\n=== {secs:.0f}초 스캔 ===")
    print("패드를 페어링 모드로 두어라 — Xbox 버튼으로 켠 뒤 위쪽 작은 연결")
    print("버튼을 로고가 **빠르게 깜빡일 때까지** 3초 누른다.")
    bctl("power on", "agent on", "default-agent")
    proc = subprocess.Popen(["bluetoothctl", "scan", "on"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(secs)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    out = bctl("devices")
    found = [(m.group(1), m.group(2).strip())
             for m in re.finditer(r"Device ([0-9A-F:]{17}) (.+)", out)]
    if name_filter:
        cand = [(a, n) for a, n in found if name_filter.lower() in n.lower()]
    else:
        cand = [(a, n) for a, n in found
                if any(h in n.lower() for h in NAME_HINTS)]
    print(f"\n발견 {len(found)}대, 패드 후보 {len(cand)}대")
    for a, n in found:
        mark = " <-- 후보" if (a, n) in cand else ""
        print(f"  {a}  {n}{mark}")
    if not cand:
        print("\n후보가 없다. 패드가 페어링 모드인지 확인하고 다시 실행할 것.")
        print("이름으로 못 거르면 --name 으로 직접 지정: --name 'Xbox'")
        return 1

    # 패드는 재연결마다 MAC 끝자리를 바꾸므로 **가장 최근에 보인 것부터** 시도한다.
    for addr, name in cand:
        print(f"\n--- {addr} ({name}) 시도 ---")
        out = bctl(f"pair {addr}", f"trust {addr}", f"connect {addr}", wait=8.0)
        good = ("Connection successful" in out or "Paired: yes" in out
                or "already" in out.lower())
        for line in out.splitlines():
            if re.search(r"(Failed|successful|Paired|Trusted|Connected)", line):
                print("   " + line.strip())
        # bluetoothctl 의 출력은 믿을 게 못 된다. 장치 노드로 확인한다.
        for _ in range(20):
            if os.path.exists(DEV):
                print(f"\n붙었다. {DEV} 생성 확인.")
                print("\n다음:")
                print("  python3 ~/joy_monitor.py    # 축·버튼 번호 확인")
                print("  python3 ~/joy_local.py      # 속도 명령 변환 확인")
                return 0
            time.sleep(0.5)
        print(f"   {DEV} 이 안 생겼다" + ("" if good else " (연결도 실패)"))

    print("\n전부 실패. --forget 으로 등록을 지우고 처음부터 해 볼 것.")
    return 1


def forget() -> int:
    out = bctl("devices")
    n = 0
    for m in re.finditer(r"Device ([0-9A-F:]{17}) (.+)", out):
        addr, name = m.group(1), m.group(2).strip()
        if any(h in name.lower() for h in NAME_HINTS):
            print(f"  제거 {addr}  {name}")
            bctl(f"disconnect {addr}", f"remove {addr}")
            n += 1
    print(f"{n}대 제거. 이제 bt_pad.py 를 다시 실행할 것.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="지금 상태만 본다")
    ap.add_argument("--forget", action="store_true", help="패드 등록을 지운다")
    ap.add_argument("--secs", type=float, default=15.0, help="스캔 시간")
    ap.add_argument("--name", default=None, help="이 문자열이 든 이름만 후보로")
    args = ap.parse_args()
    if args.status:
        status()
        return 0
    if args.forget:
        return forget()
    return scan_and_connect(args.secs, args.name)


if __name__ == "__main__":
    sys.exit(main())
