#!/usr/bin/env python3
"""14축 다이나믹셀의 **컨트롤 테이블 전체**(EEPROM+RAM)를 읽고 좌우를 비교한다.

    ssh parksuho@192.168.137.7 'python3 ~/dump_ctrl_table.py'
    ssh parksuho@192.168.137.7 'python3 ~/dump_ctrl_table.py --all'   # 같은 값도 전부
    ssh parksuho@192.168.137.7 'python3 ~/dump_ctrl_table.py --csv ~/ctrl_table.csv'

토크를 켜지 않는다. 읽기만 한다.

## 왜

2026-08-12 실기 보행에서 왼무릎이 목표보다 22~32도 더 접힌 채로 끌려다녔다.
그런데 하드웨어 시험은 두 번 다 통과했다 (주파수 응답 이득 0.99, 좌우차 2 %;
손으로 들고 보행하면 정상 추종). 즉 **모터 자체는 멀쩡하다.**

그렇다면 남은 후보 중 하나가 서보 **설정값**이다. Position P Gain, Profile
Velocity, Drive Mode, Current Limit, Homing Offset 같은 값이 좌우가 다르면
정확히 이 증상이 난다 — 무부하에서는 둘 다 잘 따라가지만 부하가 걸리면
한쪽만 뒤처진다. 이 값들은 EEPROM 에 있어서 한 번 잘못 쓰이면 전원을 껐다
켜도 그대로 남는다. 지금까지 한 번도 확인한 적이 없다.

## 읽는 법

기본 출력은 **좌우가 다른 항목만** 보여준다. 아무것도 안 나오면 설정은
좌우 동일하다는 뜻이고, 그러면 원인은 설정이 아니라 기구/하중 쪽이다.
전체를 보려면 `--all`.
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time

sys.path.insert(0, os.path.expanduser("~"))
from rustypot_hwi import BY_NAME, HWI, NAMES  # noqa: E402

# (주소, 바이트, 이름, 부호있음, 단위환산 or None)
# XM430-W350 프로토콜 2.0 컨트롤 테이블.
EEPROM = [
    (0, 2, "Model Number", False, None),
    (2, 4, "Model Information", False, None),
    (6, 1, "Firmware Version", False, None),
    (7, 1, "ID", False, None),
    (8, 1, "Baud Rate", False, None),
    (9, 1, "Return Delay Time", False, ("us", 2.0)),
    (10, 1, "Drive Mode", False, None),
    (11, 1, "Operating Mode", False, None),
    (12, 1, "Secondary(Shadow) ID", False, None),
    (13, 1, "Protocol Type", False, None),
    (20, 4, "Homing Offset", True, ("tick", 1.0)),
    (24, 4, "Moving Threshold", False, ("rpm", 0.229)),
    (31, 1, "Temperature Limit", False, ("degC", 1.0)),
    (32, 2, "Max Voltage Limit", False, ("V", 0.1)),
    (34, 2, "Min Voltage Limit", False, ("V", 0.1)),
    (36, 2, "PWM Limit", False, ("%", 0.113)),
    (38, 2, "Current Limit", False, ("mA", 2.69)),
    (44, 4, "Velocity Limit", False, ("rpm", 0.229)),
    (48, 4, "Max Position Limit", True, ("tick", 1.0)),
    (52, 4, "Min Position Limit", True, ("tick", 1.0)),
    (63, 1, "Shutdown", False, None),
]

RAM = [
    (64, 1, "Torque Enable", False, None),
    (65, 1, "LED", False, None),
    (68, 1, "Status Return Level", False, None),
    (69, 1, "Registered Instruction", False, None),
    (70, 1, "Hardware Error Status", False, None),
    (76, 2, "Velocity I Gain", False, None),
    (78, 2, "Velocity P Gain", False, None),
    (80, 2, "Position D Gain", False, None),
    (82, 2, "Position I Gain", False, None),
    (84, 2, "Position P Gain", False, None),
    (88, 2, "Feedforward 2nd Gain", False, None),
    (90, 2, "Feedforward 1st Gain", False, None),
    (98, 1, "BUS Watchdog", True, ("ms", 20.0)),
    (100, 2, "Goal PWM", True, ("%", 0.113)),
    (102, 2, "Goal Current", True, ("mA", 2.69)),
    (104, 4, "Goal Velocity", True, ("rpm", 0.229)),
    (108, 4, "Profile Acceleration", False, ("rpm2", 214.577)),
    (112, 4, "Profile Velocity", False, ("rpm", 0.229)),
    (116, 4, "Goal Position", True, ("tick", 1.0)),
    (122, 1, "Moving", False, None),
    (123, 1, "Moving Status", False, None),
    (124, 2, "Present PWM", True, ("%", 0.113)),
    (126, 2, "Present Current", True, ("mA", 2.69)),
    (128, 4, "Present Velocity", True, ("rpm", 0.229)),
    (132, 4, "Present Position", True, ("tick", 1.0)),
    (144, 2, "Present Input Voltage", False, ("V", 0.1)),
    (146, 1, "Present Temperature", False, ("degC", 1.0)),
    (147, 1, "Backup Ready", False, None),
]

# 읽을 때마다 값이 달라지는 항목 — 좌우 비교에서 제외한다. 비교하고 싶은 것은
# '설정' 이지 '지금 상태' 가 아니다.
VOLATILE = {
    "Realtime Tick", "Moving", "Moving Status", "Present PWM", "Present Current",
    "Present Velocity", "Present Position", "Present Input Voltage",
    "Present Temperature", "Goal Position", "Goal PWM", "Goal Current",
    "Goal Velocity", "LED", "Registered Instruction",
}

DRIVE_MODE_BITS = {0: "역방향", 1: "슬레이브", 2: "시간기반프로파일", 3: "토크온시 goal=현재"}
OP_MODE = {0: "전류", 1: "속도", 3: "위치", 4: "확장위치", 5: "전류기반위치", 16: "PWM"}
ERR_BITS = {0: "InputVoltage", 2: "OverHeating", 3: "MotorEncoder",
            4: "ElectricalShock", 5: "Overload"}


def decode(name: str, val: int) -> str:
    """사람이 읽을 수 있는 부가 설명. 없으면 빈 문자열."""
    if name == "Operating Mode":
        return OP_MODE.get(val, "?")
    if name == "Drive Mode":
        on = [t for b, t in DRIVE_MODE_BITS.items() if val >> b & 1]
        return "+".join(on) if on else "정방향/기본"
    if name in ("Shutdown", "Hardware Error Status"):
        on = [t for b, t in ERR_BITS.items() if val >> b & 1]
        return "+".join(on) if on else "없음"
    if name == "Baud Rate":
        return {0: "9600", 1: "57.6k", 2: "115.2k", 3: "1M", 4: "2M",
                5: "3M", 6: "4M", 7: "4.5M"}.get(val, "?")
    if name == "Torque Enable":
        return "켬" if val else "끔"
    return ""


def unpack(buf: bytes, size: int, signed: bool) -> int:
    fmt = {1: "b" if signed else "B", 2: "h" if signed else "H",
           4: "i" if signed else "I"}[size]
    return struct.unpack("<" + fmt, buf[:size])[0]


def read_block(io, ids, addr, length, retries=5):
    """sync_read 는 가끔 짧은 응답을 준다. 온전한 응답을 받을 때까지 재시도하고,
    그래도 안 되면 그 ID 만 개별로 읽는다."""
    for _ in range(retries):
        try:
            raw = io.sync_read_raw_data(ids, addr, length)
        except Exception:
            time.sleep(0.02)
            continue
        if all(len(r) == length for r in raw):
            return list(raw)
        time.sleep(0.02)
    out = []
    for i in ids:
        got = None
        for _ in range(retries):
            try:
                r = io.sync_read_raw_data([i], addr, length)
                if len(r[0]) == length:
                    got = bytes(r[0])
                    break
            except Exception:
                pass
            time.sleep(0.02)
        out.append(got)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="좌우 같은 항목도 전부 출력")
    ap.add_argument("--csv", default="", help="CSV 로도 저장")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    args = ap.parse_args()

    names = list(NAMES)
    ids = [BY_NAME[n][1] for n in names]

    hwi = HWI(port=args.port)
    io = hwi.io
    # 토크는 건드리지 않는다. 지금 상태 그대로 읽는다.

    print(f"14축 컨트롤 테이블 읽는 중 ({len(EEPROM)+len(RAM)} 항목)...", flush=True)
    table: dict[str, dict[str, int | None]] = {}
    for addr, size, fname, signed, unit in EEPROM + RAM:
        bufs = read_block(io, ids, addr, size)
        row = {}
        for n, b in zip(names, bufs):
            row[n] = None if b is None else unpack(bytes(b), size, signed)
        table[fname] = row

    bad = [f for f, r in table.items() if any(v is None for v in r.values())]
    if bad:
        print(f"  !! 못 읽은 항목: {', '.join(bad)}")

    # 좌우 짝 만들기
    pairs = []
    for n in names:
        if n.startswith("left_"):
            r = "right_" + n[5:]
            if r in table["ID"]:
                pairs.append((n, r))

    def fmt(fname, val, unit):
        if val is None:
            return "?"
        s = str(val)
        if unit:
            u, k = unit
            s += f" ({val*k:.4g}{u})" if k != 1.0 else f" {u}"
        d = decode(fname, val)
        return f"{s} [{d}]" if d else s

    meta = {f: (a, s, u) for a, s, f, _sg, u in EEPROM + RAM}
    area = {f: ("EEPROM" if (a, s, f, sg, u) in EEPROM else "RAM")
            for a, s, f, sg, u in EEPROM + RAM}

    print("\n" + "=" * 96)
    print("좌우가 다른 항목  (설정값만 비교 — Present_* / Goal_* 처럼 매번 변하는 값은 제외)")
    print("=" * 96)
    diffs = 0
    for fname, row in table.items():
        if fname in VOLATILE:
            continue
        addr, size, unit = meta[fname]
        for L, R in pairs:
            if row[L] is None or row[R] is None or row[L] == row[R]:
                continue
            if fname in ("ID", "Homing Offset", "Max Position Limit", "Min Position Limit"):
                # 이 넷은 좌우가 다른 게 **정상**이다 (ID 는 당연히, 나머지는
                # 좌우 거울 대칭이라 부호가 반대). 그래도 눈으로 보게 찍어 준다.
                tag = "  (정상: 좌우 대칭이라 다름)"
            else:
                tag = "  <<< 이건 같아야 한다"
                diffs += 1
            print(f"\n  [{area[fname]} {addr:3d}] {fname}{tag}")
            print(f"      {L:18s} {fmt(fname, row[L], unit)}")
            print(f"      {R:18s} {fmt(fname, row[R], unit)}")

    if diffs == 0:
        print("\n  없음 — 좌우 설정은 완전히 동일하다.")
        print("  => 왼무릎 처짐의 원인은 서보 **설정** 이 아니다. 기구/하중 쪽을 봐야 한다.")
    else:
        print(f"\n  좌우 불일치 {diffs} 건. 위 항목을 맞추면 증상이 사라질 수 있다.")

    # 다리 6축 그룹 안에서 튀는 값도 잡는다 (좌우 짝이 아니라 전체 중 소수파).
    print("\n" + "-" * 96)
    print("다리 10축 중 **혼자 다른 값**  (짝 비교로는 안 걸리는 경우)")
    leg = [n for n in names if not n.startswith(("neck", "head"))]
    odd = 0
    for fname, row in table.items():
        if fname in VOLATILE or fname in ("ID", "Homing Offset",
                                          "Max Position Limit", "Min Position Limit"):
            continue
        vals = [row[n] for n in leg if row[n] is not None]
        if not vals:
            continue
        counts: dict[int, list[str]] = {}
        for n in leg:
            if row[n] is not None:
                counts.setdefault(row[n], []).append(n)
        if len(counts) < 2:
            continue
        major = max(counts, key=lambda k: len(counts[k]))
        for v, who in counts.items():
            if v != major and len(who) <= 2:
                addr, size, unit = meta[fname]
                print(f"  [{area[fname]} {addr:3d}] {fname}: {', '.join(who)} = "
                      f"{fmt(fname, v, unit)}   (나머지 {len(counts[major])}축 = "
                      f"{fmt(fname, major, unit)})")
                odd += 1
    if odd == 0:
        print("  없음 — 다리 10축 설정이 전부 동일하다.")

    if args.all:
        print("\n" + "=" * 96)
        print("전체 덤프")
        print("=" * 96)
        w = 13
        print(f"  {'addr':>4} {'항목':26}" + "".join(f"{n[:12]:>{w}}" for n in names))
        for fname, row in table.items():
            addr, size, unit = meta[fname]
            print(f"  {addr:4d} {fname:26}"
                  + "".join(f"{('?' if row[n] is None else row[n]):>{w}}" for n in names))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["area", "addr", "size", "field"] + names)
            for fname, row in table.items():
                addr, size, unit = meta[fname]
                wr.writerow([area[fname], addr, size, fname]
                            + [row[n] for n in names])
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
