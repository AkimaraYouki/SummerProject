#!/usr/bin/env python3
"""정책을 **사용자가 정한 우선순위 그대로** 판정한다. 6 방향 전부.

    python3 scripts/diag/verdict.py ~/odm_out/gait_v55.npz
    python3 scripts/diag/verdict.py ~/odm_out/gait_v48.npz ~/odm_out/gait_v55.npz \
        --labels v48 v55

입력은 `odm measure` 가 남기는 `~/odm_out/gait_<ver>.npz` 다.

## 왜 이게 필요한가

2026-08-14 에 사용자가 순위를 못박았다:

    1 순위  6 방향 추종 성능 — 그 안에서도 **앞뒤 > 회전 > 완전 옆걸음**
    2 순위  보행 안정성 (덜컹거림, 좌우 흔들림, 발 쿵쾅거림)
    3 순위  보행 효율 (토크 사용, 진동)

그런데 그날 판정은 `gait_quality.py` 로 **vx 0.15 전진 하나만** 재고 했다.
1 순위가 6 방향인데 그중 하나만 본 것이다. 실제로 v55 의 요 추종 오차는
v48 보다 나빴는데(0.437 vs 0.395) 전진 지표가 좋아서 묻혔다.

`gait_compare.py` 는 처음부터 6 방향을 돌리고 있었다 — 자세·토크가 안 남아
2·3 순위를 못 봤을 뿐이다. 그 셋을 추가하고, 여기서 순위대로 찍는다.

## 각 줄이 무엇에 대응하는가

    [1] 속도 오차      명령한 몸통 속도와 실제의 차이. 방향마다 따로 본다.
                       평균이 아니라 **최악 방향**이 그 정책의 실력이다.
        요 오차        회전 명령 추종. 직진 명령에서는 0 이어야 한다.
    [2] roll/pitch     좌우·앞뒤 흔들림. 사용자가 말한 "덜컹거림".
        착지 하강속도  발이 지면 10 mm 안에서 내려오는 속도. "발 쿵쾅거림".
        접지 바운스    한 스텝 안에서 접지가 튀는 정도. 디바운스 전후 주기 비.
    [3] 토크 제곱합    모터 부하. 스톨(4.1 N·m)에 닿는 축이 있는지도 본다.
        관절 가속도    진동의 물리적 정체.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

#: 6 방향 가중치. 사용자 순위 (2026-08-14): 앞뒤 > 회전 > 완전 옆걸음.
#: 최악 방향만 보면 가장 덜 중요한 옆걸음이 판정을 지배한다 — 실제로
#: 모든 버전에서 좌/우가 최악이라 그 줄만으로는 순위가 안 갈렸다.
PRIO_W = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}

#: 스톨 토크 (N·m). robot_cfg.py 의 effort_limit_sim 과 같은 값.
STALL_NM = 4.1
#: 착지 벌점이 걸리는 높이 (m). joystick_env_cfg.foot_impact_band 와 맞춘다.
IMPACT_BAND = 0.010
#: 과도구간을 버릴 스텝 수.
SKIP = 100


def load(path):
    z = np.load(path, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    dt = float(z["ctrl_dt"])
    out = {}
    for c in conds:
        d = {}
        for k in ("v_base", "w_base", "cmd", "grav", "tau", "foot_z", "foot_v", "feet"):
            key = f"{c}__{k}"
            if key in z:
                d[k] = z[key]
        out[c] = d
    return conds, dt, out


def analyse(path):
    conds, dt, S = load(path)
    r = {"conds": conds, "dt": dt, "per": {}}
    tau_all, acc_all = [], []
    for c in conds:
        d = S[c]
        if "v_base" not in d:
            continue
        cmd = d["cmd"]
        v = d["v_base"][SKIP:]                       # (T, N, 2) 몸통 기준
        w = d["w_base"][SKIP:]                       # (T, N)
        e_v = float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2]))
        e_w = float(abs(w.mean() - cmd[2]))
        # 회전 명령의 추종은 **요레이트**로 본다. 선속도 오차로 보면 "제자리에
        # 잘 머무는가" 를 재는 것이지 "명령한 속도로 도는가" 가 아니다.
        p = {"v_err": e_w if c == "turn" else e_v, "w_err": e_w, "lin_err": e_v}

        if "grav" in d:
            g = d["grav"][SKIP:]                     # (T, N, 3)
            roll = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
            pitch = np.degrees(np.arctan2(g[..., 0], -g[..., 2]))
            p["roll_pp"] = float(roll.max() - roll.min())
            p["roll_rms"] = float(roll.std())
            p["pitch_mean"] = float(pitch.mean())

        if "foot_z" in d and "foot_v" in d:
            z = d["foot_z"][SKIP:]                   # (T, N, 2)
            z = z - z.min(axis=2, keepdims=True)     # 접지 발 기준 여유
            vz = d["foot_v"][SKIP:][..., 2]          # (T, N, 2) 몸통 기준 z 속도
            near = np.clip(1.0 - z / IMPACT_BAND, 0.0, None)
            down = np.clip(-vz, 0.0, None)
            p["impact"] = float((down * near).max())          # 최악 착지 속도
            p["impact_rms"] = float(np.sqrt((down ** 2 * near).mean()))
            p["clear"] = float(np.median(z.max(axis=2)) * 1000)  # 스윙 여유 mm

        if "feet" in d:
            f = d["feet"][SKIP:]                     # (T, N, 2)
            p["bounce"] = _bounce(f, dt)

        if "tau" in d:
            t = d["tau"][SKIP:]
            p["tau2"] = float((t ** 2).sum(axis=-1).mean())
            p["tau_max"] = float(np.abs(t).max())
            p["stall_pct"] = float(100.0 * (np.abs(t) >= STALL_NM * 0.99).mean())
            tau_all.append(p["tau2"])
            dq = np.diff(t, axis=0)                  # 토크 변화 = 진동 대용
            acc_all.append(float((dq ** 2).sum(axis=-1).mean()))
        r["per"][c] = p
    if tau_all:
        r["tau2_mean"] = float(np.mean(tau_all))
        r["jitter"] = float(np.mean(acc_all))
    moving = [c for c in conds if c != "stop" and c in r["per"]]
    if moving:
        r["v_err_worst"] = max(r["per"][c]["v_err"] for c in moving)
        r["v_err_mean"] = float(np.mean([r["per"][c]["v_err"] for c in moving]))
        r["w_err_worst"] = max(r["per"][c]["w_err"] for c in moving)
        # 우선순위 가중 점수. 이게 최종 판정 숫자다.
        num = sum(PRIO_W[c] * r["per"][c]["v_err"] for c in PRIO_W if c in r["per"])
        den = sum(PRIO_W[c] for c in PRIO_W if c in r["per"])
        if den:
            r["prio"] = num / den
        fb = [r["per"][c]["v_err"] for c in ("forward", "backward") if c in r["per"]]
        lr = [r["per"][c]["v_err"] for c in ("left", "right") if c in r["per"]]
        if fb:
            r["fb"] = float(np.mean(fb))
        if lr:
            r["lr"] = float(np.mean(lr))
        for k in ("roll_pp", "roll_rms", "impact", "impact_rms", "bounce", "clear"):
            vals = [r["per"][c][k] for c in moving if k in r["per"][c]]
            if vals:
                r[k + "_worst"] = max(vals)
                r[k + "_mean"] = float(np.mean(vals))
    return r


def _bounce(f, dt):
    """디바운스 전후 걸음 주기의 비. 1.0 이면 안 튄다.

    접지 전환 횟수만 세면 속는다 — 2026-08-14 에 v53 이 주기 195 ms 로 나와
    "7 Hz 로 떨고 있다" 고 판단했는데, 40 ms 디바운스하니 540 ms 로 정상이었다.
    걸음이 아니라 한 스텝 안에서 발이 튄 것이다. 그 비를 직접 잰다.
    """
    def cyc(sig, deb):
        segs, val, start = [], sig[0], 0
        for i in range(1, len(sig)):
            if sig[i] != val:
                segs.append((val, start, i)); val, start = sig[i], i
        segs.append((val, start, len(sig)))
        m = []
        for v, a, b in segs:
            if m and (b - a) * dt * 1000 < deb:
                m[-1] = (m[-1][0], m[-1][1], b)
            elif m and m[-1][0] == v:
                m[-1] = (v, m[-1][1], b)
            else:
                m.append((v, a, b))
        st = [(b - a) * dt for v, a, b in m if v > 0.5]
        sw = [(b - a) * dt for v, a, b in m if v <= 0.5]
        if not st or not sw:
            return None
        return float(np.median(st) + np.median(sw))
    rat = []
    for n in range(f.shape[1]):
        for s in range(f.shape[2]):
            sig = f[:, n, s]
            a, b = cyc(sig, 0), cyc(sig, 40)
            if a and b and a > 0:
                rat.append(b / a)
    return float(np.median(rat)) if rat else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()
    labs = args.labels or [os.path.basename(p).replace("gait_", "").replace(".npz", "")
                           for p in args.npz]
    res = []
    for p, l in zip(args.npz, labs):
        try:
            res.append((l, analyse(p)))
        except Exception as e:                       # noqa: BLE001
            print(f"  {l}: 읽기 실패 ({e})")
    if not res:
        return
    W = 11

    def row(name, key, fmt, note=""):
        cells = "".join(f"{fmt.format(a[key]):>{W}}" if key in a and a[key] == a[key]
                        else f"{'—':>{W}}" for _, a in res)
        print(f"  {name:24}{cells}   {note}")

    def prow(name, cond, key, fmt, note=""):
        cells = "".join(
            f"{fmt.format(a['per'][cond][key]):>{W}}"
            if cond in a["per"] and key in a["per"][cond] else f"{'—':>{W}}"
            for _, a in res)
        print(f"  {name:24}{cells}   {note}")

    print("\n  " + " " * 24 + "".join(f"{l:>{W}}" for l, _ in res))
    print("  " + "-" * (24 + W * len(res) + 3))
    print("  [1순위] 추종 — 앞뒤 > 회전 > 옆걸음 순으로 중요")
    row("  ★ 우선순위 점수", "prio", "{:.4f}", "가중 3:2:1 · 최종 판정값")
    row("  1) 앞뒤 평균", "fb", "{:.4f}", "m/s")
    prow("     전진", "forward", "v_err", "{:.4f}", "m/s")
    prow("     후진", "backward", "v_err", "{:.4f}", "m/s")
    prow("  2) 회전", "turn", "v_err", "{:.4f}", "rad/s · 요레이트 오차")
    row("  3) 옆걸음 평균", "lr", "{:.4f}", "m/s")
    prow("     좌", "left", "v_err", "{:.4f}", "m/s")
    prow("     우", "right", "v_err", "{:.4f}", "m/s")
    prow("  정지 유지", "stop", "v_err", "{:.4f}", "m/s")

    print("  [2순위] 보행 안정성")
    row("  roll 진폭 최악", "roll_pp_worst", "{:.1f}", "도 · 좌우 흔들림")
    row("  roll RMS 평균", "roll_rms_mean", "{:.2f}", "도")
    row("  착지 하강속도 최악", "impact_worst", "{:.3f}", "m/s · 발 쿵쾅거림")
    row("  착지 하강 RMS", "impact_rms_mean", "{:.4f}", "m/s")
    row("  접지 바운스", "bounce_mean", "{:.2f}", "1.0 = 안 튐, 클수록 튐")
    row("  스윙 여유", "clear_mean", "{:.1f}", "mm")

    print("  [3순위] 보행 효율")
    row("  토크 제곱합", "tau2_mean", "{:.2f}", "N·m^2 · 낮을수록 효율")
    row("  토크 떨림", "jitter", "{:.3f}", "스텝간 토크 변화 제곱합")
    prow("  스톨 도달 비율", "forward", "stall_pct", "{:.1f}", "% · 4.1 N·m 접촉")
    prow("  최대 토크(전진)", "forward", "tau_max", "{:.2f}", "N·m")
    print()


if __name__ == "__main__":
    main()
