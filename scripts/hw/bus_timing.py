#!/usr/bin/env python3
"""제어 루프의 시간이 **어디로 가는지** 잰다 (Jetson). 토크는 켜지 않는다.

    ssh -t parksuho@192.168.137.7 'python3 ~/bus_timing.py'
    ssh -t parksuho@192.168.137.7 'python3 ~/bus_timing.py --n 300'

## 왜

2026-08-14 에 6 방향 추종을 역추적하니 **회전이 v42 0.0040 -> v47 0.0672 로
17 배 나빠졌고**, 그 사이 유일한 변경이 액션 지연 0~3 -> 2~3 스텝이었다.
지연을 늘린 근거는 실측이다 — `track_stats.py` 로 잰 실기 지연이 43 ms,
무릎은 64 ms 로 심보다 한 스텝 이상 느렸다.

그러면 답은 둘 중 하나다. 지연을 심에서 더 정확히 모델링하거나, **지연
자체를 줄이거나.** 후자가 되면 회전을 공짜로 되찾는다.

그래서 43 ms 가 어디로 가는지 나눠 본다. 통신이 아니라 서보 내부 응답이면
버스를 아무리 빠르게 해도 안 줄어든다.

## 무엇을 재는가

    sync_write 목표위치   루프마다 한 번. 14 축.
    sync_read  현재위치   루프마다 한 번. 14 축.
    블록 read (64,7)      에러/전압 등 상태 블록.
    루프 합계             위 셋을 순서대로 한 번씩. 예산은 20 ms (50 Hz).

Return Delay Time(주소 9) 도 같이 읽는다. 기본 250 은 축당 **0.5 ms** 이고,
14 축 sync_read 면 그것만으로 7 ms 다. 0 으로 두면 사라진다. EEPROM 이라
한 번 쓰면 남고, 되돌릴 수 있다.

## 안전

토크를 켜지 않는다. 목표위치 쓰기는 **현재 위치를 그대로** 쓰므로 토크가
켜져 있어도 움직이지 않는다. 그래도 매달아 놓고 하는 편이 낫다.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from rustypot_hwi import BAUD, IDS, LEG_IDS, NAMES, HWI  # noqa: F401

#: Return Delay Time 레지스터 (EEPROM). 값 1 = 2 us.
ADDR_RETURN_DELAY = 9
#: 목표 제어 주기 (s).
BUDGET = 0.02


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "mean": statistics.fmean(ts), "med": ts[len(ts) // 2],
        "p95": ts[int(0.95 * (len(ts) - 1))], "max": ts[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    hwi = HWI(args.port)
    io = hwi.io

    rd = [b[0] for b in io.sync_read_raw_data(IDS, ADDR_RETURN_DELAY, 1)]
    print("=" * 74)
    print(f"버스 타이밍  BAUD {BAUD:,}  ·  {args.n} 회  ·  예산 {BUDGET*1000:.0f} ms (50 Hz)")
    print("=" * 74)
    print(f"Return Delay Time: {sorted(set(rd))}  (1 단위 = 2 us, 기본 250 = 500 us)")
    print(f"  14 축 응답 대기만 = {sum(rd) * 2 / 1000.0:.2f} ms / read")
    print()

    pos = io.sync_read_present_position(IDS)

    cases = [
        ("sync_read  현재위치 (14)", lambda: io.sync_read_present_position(IDS)),
        ("sync_write 목표위치 (14)", lambda: io.sync_write_goal_position(IDS, pos)),
        ("블록 read  (64,7) 다리10", lambda: io.sync_read_raw_data(LEG_IDS, 64, 7)),
        ("블록 read  (144,3) 다리10", lambda: io.sync_read_raw_data(LEG_IDS, 144, 3)),
    ]
    print(f"  {'항목':<26}{'평균':>9}{'중앙':>9}{'p95':>9}{'최대':>9}")
    print("  " + "-" * 62)
    tot = 0.0
    for name, fn in cases:
        try:
            r = timeit(fn, args.n)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {name:<26}  실패: {e}")
            continue
        print(f"  {name:<26}{r['mean']:>9.2f}{r['med']:>9.2f}{r['p95']:>9.2f}{r['max']:>9.2f}")
        if "144" not in name:      # 전압 블록은 25 스텝에 한 번이라 상시 비용이 아니다
            tot += r["mean"]

    print("  " + "-" * 62)
    print(f"  {'매 루프 합계 (전압 제외)':<26}{tot:>9.2f} ms")
    print(f"  {'예산 대비':<26}{100*tot/(BUDGET*1000):>8.0f} %")
    print()
    if tot > BUDGET * 1000 * 0.5:
        print("  통신이 예산의 절반을 넘는다 — 여기가 지연의 주범이다.")
        print("  줄일 수단: BAUD 상향 (1M -> 3M/4M), Return Delay Time 250 -> 0.")
    else:
        print("  통신은 예산에 여유가 있다. 실기 지연 43 ms 의 대부분은 통신이")
        print("  아니라 **서보 내부 응답**이라는 뜻이고, 버스를 빠르게 해도")
        print("  거의 안 줄어든다. 그때는 심의 지연 모델을 유지하는 편이 맞다.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
