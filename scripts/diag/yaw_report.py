#!/usr/bin/env python3
"""실기 로그(`rl_walk_log.csv`)에서 방향 문제를 가른다.  2026-09-17.

사용자 증상: 스로틀을 계속 당기면 조금씩 옆으로 새고, 회전은 되지만 잘 안 된다.
"자이로 적분이 잘못된 건가, path frame 과 관련이 있나?"

이 도구가 답하는 것:

1. **전진 구간** (|vx| 명령 > 0.05, |wz| 명령 < 0.05)
   자이로 z 를 적분한 방향 변화. 이것이 거의 0 인데 눈으로는 옆으로 샜다면
   **몸통 방향은 그대로인 채 옆으로 밀린 것**이다 — 실기에 없는 횡방향 path
   오차(늘 0) 쪽 문제다. 방향이 크게 바뀌었다면 **휘어서 간 것**이다.
2. **회전 구간** (|wz| 명령 > 0.1)
   실제 회전 / 명령 비율, 방위 오차(`path_yaw_err`)의 최대 크기, 그리고
   ±π 근처에서 **부호가 뒤집힌 횟수**. 뒤집힘이 있으면 `--path-yaw-clip` 이
   필요하다.

    python3 scripts/diag/yaw_report.py rl_walk_log.csv
"""
from __future__ import annotations

import csv
import math
import sys


def segments(rows, pred, min_len=25):
    """pred 가 연속으로 참인 구간들 [(i0, i1), ...]. 0.5 초 미만은 버린다."""
    out, start = [], None
    for i, r in enumerate(rows):
        if pred(r):
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(rows) - start >= min_len:
        out.append((start, len(rows)))
    return out


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit("빈 로그")
    # 옛 로그(2026-08-18 이전)에는 path_yaw_err 열이 없다 — 0 으로 읽는다.
    f = lambda r, k: float(r.get(k) or 0.0)                          # noqa: E731
    dt = lambda i: max(0.0, f(rows[i], "t") - f(rows[i - 1], "t")) if i else 0.0  # noqa: E731
    has_back = "policy_back" in rows[0]
    print(f"{path}: {len(rows)} 스텝, {f(rows[-1], 't'):.1f} 초"
          + ("  (policy_back 열 있음)" if has_back else ""))

    print("\n[전진·후진 구간] 자이로 적분 방향 변화")
    fwd = segments(rows, lambda r: abs(f(r, "cmd_vx")) > 0.05 and abs(f(r, "cmd_wz")) < 0.05)
    if not fwd:
        print("  없음")
    for a, b in fwd:
        yaw = sum(f(rows[i], "gyro_z") * dt(i) for i in range(a + 1, b))
        secs = f(rows[b - 1], "t") - f(rows[a], "t")
        vx = sum(f(rows[i], "cmd_vx") for i in range(a, b)) / (b - a)
        pe = max(abs(f(rows[i], "path_yaw_err")) for i in range(a, b))
        print(f"  t={f(rows[a], 't'):6.1f}~{f(rows[b - 1], 't'):6.1f}s  vx {vx:+.3f}  "
              f"방향 변화 {math.degrees(yaw):+6.1f}도 ({math.degrees(yaw) / max(secs, 1e-6):+5.2f}도/s)"
              f"  |path_yaw_err| 최대 {math.degrees(pe):5.1f}도")

    print("\n[회전 구간] 명령 대비 실제 회전")
    turn = segments(rows, lambda r: abs(f(r, "cmd_wz")) > 0.1)
    if not turn:
        print("  없음")
    for a, b in turn:
        cmd = sum(f(rows[i], "cmd_wz") * dt(i) for i in range(a + 1, b))
        act = sum(f(rows[i], "gyro_z") * dt(i) for i in range(a + 1, b))
        errs = [f(rows[i], "path_yaw_err") for i in range(a, b)]
        flips = sum(1 for e0, e1 in zip(errs, errs[1:])
                    if e0 * e1 < 0 and abs(e0) > 2.5 and abs(e1) > 2.5)
        ratio = act / cmd if abs(cmd) > 1e-6 else float("nan")
        print(f"  t={f(rows[a], 't'):6.1f}~{f(rows[b - 1], 't'):6.1f}s  명령 {math.degrees(cmd):+7.1f}도"
              f"  실제 {math.degrees(act):+7.1f}도  비율 {ratio:4.2f}"
              f"  |오차| 최대 {math.degrees(max(map(abs, errs))):5.1f}도"
              f"  ±π 부호뒤집힘 {flips}회")

    if all(abs(f(r, "path_yaw_err")) < 1e-9 for r in rows):
        print("\n!! path_yaw_err 가 전부 0 이다 — --path-imu 없이 돌린 로그다.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rl_walk_log.csv")
