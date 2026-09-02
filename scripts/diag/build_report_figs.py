#!/usr/bin/env python3
"""KSPE 창의경진대회 중간보고서 본문 삽입용 그림을 생성한다.

    $IL -p scripts/diag/build_report_figs.py                 # docs/kspe/figures 에
    ODM_FIG_OUT=/tmp/fig $IL -p scripts/diag/build_report_figs.py

`~/odm_out/rep/<ver>_<i>.npz` (= `odm measure <ver> --repeat=N` 산출물)와
`~/odm_logs/train_<ver>.log` 를 읽는다.

## 규약

`docs/graph_conventions.md` 를 따른다. 특히 두 가지를 지킨다.

  · 같은 물리량 패널은 **축을 공유한다**. 그림 1 초판이 6 패널의 y 축을 각각
    잡았다가 비교가 불가능해졌고, 백분위 띠가 축을 지배해 신호가 납작해졌다.
  · 유니코드 마이너스(U+2212)를 쓰지 않는다. NanumGothic 에 글리프가 없어
    두부(□)로 나간다.

## 그림 3 의 판정 규칙

두 정책의 측정 범위가 겹치면 **개선폭을 주장하지 않는다.** 보고서 3 절에서
"회전 추종의 흩어짐이 223 % 라 판정에 쓸 수 없다" 고 적어 놓고 표에서는 개선
실적으로 제시하면 자기모순이므로, 그림이 스스로 그 규칙을 지키게 했다.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

for _f in ("NanumGothic", "AppleGothic", "Apple SD Gothic Neo"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "source")
from open_duck_mini_isaaclab.joint_order import (          # noqa: E402
    ACTUATOR_JOINT_NAMES, ACT_LEG_JOINT_IDX,
)

OUT = os.environ.get("ODM_FIG_OUT", "docs/kspe/figures")
REP = os.path.expanduser("~/odm_out/rep")
WARM, TAU = 100, 3.16
BLUE, ORANGE, GREEN, GRAY = "#2b6cb0", "#c2542a", "#2f7d5c", "#8b929a"
PRIO = {"forward": 3., "backward": 3., "turn": 2., "left": 1., "right": 1.}
BASE, BEST = "v61", "v75"            # 기준(실기 검증) / 개선 정책
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight",
                pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print("  ", name)


def metrics(path):
    z = np.load(path, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    dt = float(z["ctrl_dt"])
    e, rr, st, wt, pj = {}, [], [], [], []
    for c in conds:
        v = z[f"{c}__v_base"][WARM:]; w = z[f"{c}__w_base"][WARM:]; cmd = z[f"{c}__cmd"]
        e[c] = (float(abs(w.mean() - cmd[2])) if c == "turn"
                else float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2])))
        g = z[f"{c}__grav"][WARM:]
        r = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
        if c == "stop":
            continue
        rr.append(float(r.std()))
        t = z[f"{c}__tau"][WARM:][..., ACT_LEG_JOINT_IDX]
        a = np.abs(t)
        st.append(float(100 * (a > TAU * .995).mean()))
        pj.append(np.percentile(a, 99, axis=(0, 1)))
        d = z[f"{c}__dq"][WARM:]
        n = min(len(t), len(d))
        wt.append(float(np.abs(t[:n] * d[:n]).sum(-1).mean()))
    return dict(sat=np.mean(st), watt=np.mean(wt), roll=np.mean(rr),
                turn=e["turn"], fb=(e["forward"] + e["backward"]) / 2,
                lr=(e["left"] + e["right"]) / 2, pj=np.mean(pj, 0))


def agg(ver):
    rs = [metrics(p) for p in sorted(glob.glob(f"{REP}/{ver}_*.npz"))]
    o = {k: (lambda v: (v.mean(), v.min(), v.max()))(
             np.array([r[k] for r in rs]))
         for k in ("sat", "watt", "roll", "turn", "fb", "lr")}
    o["pj"] = np.mean([r["pj"] for r in rs], 0)
    o["n"] = len(rs)
    return o


# ── 그림 1. 6방향 추종 평가 ──────────────────────────────────────────────
def fig1():
    z = np.load(f"{REP}/{BEST}_1.npz", allow_pickle=True)
    dt = float(z["ctrl_dt"])
    LIN = [("forward", "전진", 0, 0.15), ("backward", "후진", 0, -0.15),
           ("left", "좌 게걸음", 1, 0.20), ("right", "우 게걸음", 1, -0.20),
           ("stop", "정지", 0, 0.0)]
    fig = plt.figure(figsize=(13.4, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axs = [fig.add_subplot(gs[0, 0])]
    axs += [fig.add_subplot(gs[0, i], sharey=axs[0]) for i in (1, 2)]
    axs += [fig.add_subplot(gs[1, i], sharey=axs[0]) for i in (0, 1)]
    axt = fig.add_subplot(gs[1, 2])

    def band(ax, y, tgt, ko):
        t = np.arange(y.shape[0]) * dt
        med = np.median(y, 1)
        ax.axvspan(0, WARM * dt, color=GRAY, alpha=0.13, lw=0)
        ax.fill_between(t, np.percentile(y, 25, 1), np.percentile(y, 75, 1),
                        color=BLUE, alpha=0.20, lw=0)
        ax.plot(t, med, color=BLUE, lw=1.5)
        ax.axhline(tgt, color=ORANGE, ls="--", lw=1.6)
        m = float(med[WARM:].mean())
        ax.set_title(f"{ko}   명령 {tgt:+.2f} -> 실측 {m:+.3f}   오차 {abs(m - tgt):.4f}",
                     fontsize=10.5)
        ax.grid(alpha=0.22)
        return med

    for ax, (c, ko, comp, tgt) in zip(axs, LIN):
        band(ax, z[f"{c}__v_base"][:, :, comp], tgt, ko)
    meds = [np.median(z[f"{c}__v_base"][:, :, comp], 1)[WARM:] for c, _, comp, _ in LIN]
    axs[0].set_ylim(min(m.min() for m in meds) - 0.06, max(m.max() for m in meds) + 0.06)

    band(axt, z["turn__w_base"], 1.0, "제자리 회전")
    mt = np.median(z["turn__w_base"], 1)[WARM:]
    axt.set_ylim(min(mt.min(), 0) - 0.25, max(mt.max(), 1.0) + 0.25)

    for a in (axs[3], axs[4], axt):
        a.set_xlabel("시간 [s]")
    axs[0].set_ylabel("몸통 기준 선속도 [m/s]")
    axs[3].set_ylabel("몸통 기준 선속도 [m/s]")
    axt.set_ylabel("요레이트 [rad/s]")
    axs[0].plot([], [], color=BLUE, lw=1.5, label="실측 (중앙값)")
    axs[0].plot([], [], color=ORANGE, ls="--", lw=1.6, label="명령")
    axs[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("6방향 명령 추종 평가 — 선속도 5개 패널은 축을 공유한다. "
                 "띠 25~75 백분위, 회색은 과도구간(집계 제외)", fontsize=12)
    save(fig, "그림1_6방향_추종평가.png")


# ── 그림 2. 학습 곡선 ────────────────────────────────────────────────────
def curve(v, off=0):
    t = open(os.path.expanduser(f"~/odm_logs/train_{v}.log"), errors="ignore").read()
    it = [int(m.group(1)) for m in re.finditer(r"Learning iteration (\d+)/", t)]
    rw = [float(m.group(1)) for m in re.finditer(r"Mean reward:\s*([-\d.]+)", t)]
    ep = [float(m.group(1)) for m in re.finditer(r"Mean episode length:\s*([-\d.]+)", t)]
    n = min(len(it), len(rw), len(ep))
    return np.array(it[:n]) + off, np.array(rw[:n]), np.array(ep[:n])


def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for v, ko, col, off in ((BEST, "개선 정책", BLUE, 700), (BASE, "기준 정책", GRAY, 0)):
        try:
            it, rw, ep = curve(v, off)
        except Exception:                                    # noqa: BLE001
            continue
        k = 21
        ax[0].plot(it[k - 1:], np.convolve(rw, np.ones(k) / k, "valid"),
                   color=col, lw=1.7, label=f"{ko} ({v})")
        ax[1].plot(it[k - 1:], np.convolve(ep, np.ones(k) / k, "valid"),
                   color=col, lw=1.7, label=f"{ko} ({v})")
    ax[0].set_ylabel("에피소드 누적 보상")
    ax[1].set_ylabel("에피소드 길이 [스텝] (최대 1000)")
    for a in ax:
        a.set_xlabel("학습 반복 (iteration)")
        a.grid(alpha=0.25)
        a.legend(fontsize=9.5)
    ax[0].set_title("학습 곡선 — 보상", fontsize=11)
    ax[1].set_title("학습 곡선 — 에피소드 길이 (넘어지지 않고 버틴 시간)", fontsize=11)
    fig.suptitle("PPO 학습 진행 (병렬 환경 4,096개) — 이동평균 21", fontsize=12)
    save(fig, "그림2_학습곡선.png")


# ── 그림 3. 기준 대비 개선 ───────────────────────────────────────────────
def fig3(A, B):
    items = [("sat", "토크 포화율", "%"), ("watt", "기계적 일률", "W"),
             ("roll", "몸통 흔들림", "°"), ("turn", "회전 추종 오차", "")]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9), constrained_layout=True)
    for ax, (k, ko, u) in zip(axes, items):
        for i, (d, col) in enumerate(((A, GRAY), (B, BLUE))):
            m, lo, hi = d[k]
            ax.bar(i, m, 0.55, color=col)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none",
                        ecolor="#2d3339", capsize=6, lw=1.3)
            ax.text(i, hi + (hi if hi else 1) * 0.06, f"{m:.4g}", ha="center",
                    fontsize=10.5, color=col, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"기준\n({BASE})", f"개선\n({BEST})"], fontsize=10)
        unit = f" [{u}]" if u else ""
        # 범위가 겹치면 개선폭을 주장하지 않는다 (모듈 독스트링 참조).
        if not (A[k][1] > B[k][2] or B[k][1] > A[k][2]):
            ax.set_title(f"{ko}{unit}\n범위 겹침 — 유의한 차이로 보기 어려움",
                         fontsize=10.5, color=ORANGE)
        else:
            r = A[k][0] / B[k][0]
            sub = f"{r:.1f}배 감소" if r > 1.5 else f"{100 * (1 - B[k][0] / A[k][0]):.0f} % 감소"
            ax.set_title(f"{ko}{unit}\n{sub}", fontsize=11)
        ax.set_ylim(0, max(A[k][2], B[k][2]) * 1.28)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("실기 검증 기준 정책 대비 개선 — 막대는 3회 측정 평균, 오차막대는 최소~최대.\n"
                 "오차막대가 겹치는 지표는 개선폭을 주장하지 않는다.", fontsize=12)
    save(fig, "그림3_성능개선_비교.png")


# ── 그림 4. 측정 재현성 ──────────────────────────────────────────────────
def fig4():
    rs = [metrics(p) for p in sorted(glob.glob(f"{REP}/{BEST}_*.npz"))]
    keys = [("watt", "기계적 일률"), ("roll", "몸통 흔들림"), ("sat", "토크 포화율"),
            ("fb", "앞뒤 추종"), ("lr", "옆걸음 추종"), ("turn", "회전 추종")]
    sp = []
    for k, ko in keys:
        v = np.array([r[k] for r in rs])
        sp.append((ko, 100 * (v.max() - v.min()) / v.mean()))
    sp.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9.4, 4.2), constrained_layout=True)
    cols = [GREEN if s < 30 else (ORANGE if s > 100 else GRAY) for _, s in sp]
    ax.barh([x[0] for x in sp], [x[1] for x in sp], color=cols, height=0.6)
    for i, (_, s) in enumerate(sp):
        ax.text(s + 4, i, f"{s:.0f} %", va="center", fontsize=10.5,
                color=cols[i], fontweight="bold")
    ax.axvline(100, color=ORANGE, ls="--", lw=1.2)
    ax.text(101, -0.65, "흩어짐이 평균보다 큰 영역", fontsize=9, color=ORANGE)
    ax.set_xlabel("동일 체크포인트 반복 측정의 흩어짐  (최대-최소) / 평균  [%]")
    ax.set_xlim(0, max(s for _, s in sp) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("측정 재현성 — 같은 정책을 반복 측정했을 때 지표가 얼마나 흔들리는가",
                 fontsize=12)
    save(fig, "그림4_측정_재현성.png")


# ── 그림 5. 관절별 피크 토크 ─────────────────────────────────────────────
def fig5(A, B):
    J = [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX]
    KO = {"hip_yaw": "고관절 yaw", "hip_roll": "고관절 roll",
          "hip_pitch": "고관절 pitch", "knee": "무릎", "ankle": "발목"}
    lab = [("왼 " if j.startswith("left_") else "오른 ") + KO[j.split("_", 1)[1]] for j in J]
    fig, ax = plt.subplots(figsize=(10.6, 4.4), constrained_layout=True)
    x = np.arange(len(J))
    ax.bar(x - 0.2, A["pj"], 0.4, color=GRAY, label=f"기준 정책 ({BASE})")
    ax.bar(x + 0.2, B["pj"], 0.4, color=BLUE, label=f"개선 정책 ({BEST})")
    ax.axhline(TAU, color=ORANGE, ls="--", lw=1.5)
    ax.text(len(J) - 0.4, TAU + 0.05, f"실효 토크 한계 {TAU} N·m",
            ha="right", color=ORANGE, fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylabel("관절별 |tau| 의 p99  [N·m]")
    ax.set_ylim(0, TAU * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("관절별 피크 토크 — 무릎만 한계에 도달한다 (액추에이터 선정 근거)",
                 fontsize=12)
    save(fig, "그림5_관절별_피크토크.png")


# ── 그림 6. 정책 성능 분포 ───────────────────────────────────────────────
def fig6():
    vers = sorted({os.path.basename(p).rsplit("_", 1)[0]
                   for p in glob.glob(f"{REP}/*_*.npz")},
                  key=lambda v: int(re.sub(r"\D", "", v) or 0))
    D = {v: agg(v) for v in vers}
    fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    for v, d in D.items():
        w, r = d["watt"][0], d["roll"][0]
        hot = v in (BASE, BEST)
        ax.scatter(w, r, s=92 if hot else 42,
                   color=(ORANGE if v == BASE else BLUE) if hot else GRAY,
                   alpha=1 if hot else .55, zorder=3 if hot else 2,
                   edgecolor="white" if hot else "none", linewidth=1.2)
        if hot or v in ("v55", "v65", "v73", "v82"):
            ax.annotate(v, (w, r), textcoords="offset points", xytext=(7, 4),
                        fontsize=9.5,
                        color=(ORANGE if v == BASE else BLUE) if hot else "#5c646c",
                        fontweight="bold" if hot else "normal")
    ax.annotate("", xy=(D[BEST]["watt"][0], D[BEST]["roll"][0]),
                xytext=(D[BASE]["watt"][0], D[BASE]["roll"][0]),
                arrowprops=dict(arrowstyle="->", color="#2d3339", lw=1.4, ls="--", alpha=.7))
    ax.set_xlabel("기계적 일률 [W]  <-  낮을수록 효율적")
    ax.set_ylabel("몸통 흔들림 (roll RMS) [°]  <-  낮을수록 안정")
    ax.grid(alpha=0.25)
    ax.set_title(f"학습한 정책 {len(D)}종의 성능 분포 — 각 점은 3회 측정 평균", fontsize=12)
    save(fig, "그림6_정책_성능분포.png")


if __name__ == "__main__":
    A, B = agg(BASE), agg(BEST)
    fig1(); fig2(); fig3(A, B); fig4(); fig5(A, B); fig6()
    print(f"[ok] {OUT}")
