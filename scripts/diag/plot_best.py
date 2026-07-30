"""최고 성능 정책의 보행 그림. `docs/graph_conventions.md` 규약을 따른다.

`odm measure` 가 떨어뜨린 npz 와 레퍼런스 pkl 만 읽는다 — Isaac Sim 도 GPU 도
쓰지 않으므로 학습 중에 돌려도 된다.

    isaaclab.sh -p scripts/diag/plot_best.py --ver v28

규약에서 특히 신경 쓴 것 (번호는 graph_conventions.md 기준):

  0  파일명 v{번호}_{한글설명}.png — 기존 30장과 정렬·비교되게
  1  같은 물리량 패널은 sharey, y 범위는 굵은 선(평균) 기준으로 잡고
     띠는 잘리게 둔다. 과도구간은 축 계산에서도 뺀다 (9번)
  2  좌우를 겹칠 때 오른쪽 부호를 뒤집는다. 부호는 추측이 아니라
     scripts/diag/derive_mirror.py 가 레퍼런스에서 유도하고, URDF 원점 rpy 와
     실측 홈 자세로 교차 확인한 값이다
  3  모든 그림에 레퍼런스를 겹친다. 실선=정책 / 파선=레퍼런스, 같은 색 alpha 0.6
  4  제목·패널에 숫자를 박는다. RMS 는 rad 과 도를 함께
  7  본문 한글, 코드 심볼과 정착 용어(path frame 등)만 영문
  8  한글 폰트 + axes.unicode_minus = False
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 규약 8 — plot_gait_compare.py 서두를 그대로 재사용
for _f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

WARM = 100           # 규약 9 — 과도구간, 오차에서도 축에서도 제외
L_COLOR = "#2a78d6"  # 왼쪽
R_COLOR = "#eb6834"  # 오른쪽
INK = "#222222"
MUTED = "#7a7a75"

CONDS = ["stop", "forward", "backward", "left", "right", "turn"]
KO = {"stop": "정지", "forward": "앞", "backward": "뒤",
      "left": "좌", "right": "우", "turn": "회전"}
JOINT_KO = {"hip_yaw": "고관절 yaw", "hip_roll": "고관절 roll",
            "hip_pitch": "고관절 pitch", "knee": "무릎", "ankle": "발목"}

# 규약 2 — derive_mirror.py 유도값. 오른쪽을 왼쪽에 겹칠 때 곱할 부호.
# URDF 원점 rpy(무릎·발목 체인이 180도 뒤집힘)와 실측 홈 자세
# (좌 무릎 -1.785 / 우 +1.816)가 같은 결론을 준다.
RIGHT_SIGN = {"hip_yaw": -1.0, "hip_roll": 1.0, "hip_pitch": 1.0,
              "knee": -1.0, "ankle": -1.0}


def _save(fig, out, fname, title, lines, panel_title_lines=0):
    """제목 블록을 인치 단위로 쌓고 그 아래에서 축이 시작하게 한다.

    `panel_title_lines` 는 각 패널이 제 위에 얹는 제목 줄 수다. 축 위로 뻗는
    그 높이를 빼 두지 않으면 패널 제목이 그림 부제를 뚫고 올라온다.

    그림 비율로 간격을 잡으면 낮은 그림에서 줄이 겹친다 (규약 5). 글자 크기는
    포인트 = 절대 길이이므로 간격도 절대 길이로 잡아야 높이와 무관하게 맞는다.
    """
    H = fig.get_size_inches()[1]
    y = 1.0 - 0.18 / H
    fig.text(0.011, y, title, ha="left", va="top", fontsize=13,
             fontweight="bold", color=INK)
    y -= 0.34 / H
    for ln in lines:
        fig.text(0.011, y, ln, ha="left", va="top", fontsize=8.5, color=MUTED)
        y -= 0.235 / H
    reserve = 0.10 + 0.175 * panel_title_lines
    fig.subplots_adjust(top=max(y - reserve / H, 0.30))
    path = os.path.join(out, fname)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print(f"[ok] {path}")


def _ylim_from(traces, pad=0.35):
    """굵은 선(평균)만으로 y 범위를 잡는다. 띠와 과도구간은 넣지 않는다."""
    lo = min(float(np.nanmin(t)) for t in traces)
    hi = max(float(np.nanmax(t)) for t in traces)
    span = max(hi - lo, 1e-3)
    return lo - span * pad, hi + span * pad


def plot_tracking(g, out, ver):
    dt = 0.02
    lin = [c for c in CONDS if c != "turn"]
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.22)

    series = {}
    for c in CONDS:
        cmd = g[f"{c}__cmd"]
        if c == "turn":
            series[c] = (g[f"{c}__w_base"], float(cmd[2]), "rad/s")
        elif abs(float(cmd[1])) > 1e-6:
            series[c] = (g[f"{c}__v_base"][:, :, 1], float(cmd[1]), "m/s")
        else:
            series[c] = (g[f"{c}__v_base"][:, :, 0], float(cmd[0]), "m/s")

    lin_lim = _ylim_from([series[c][0][WARM:].mean(axis=1) for c in lin]
                         + [np.array([series[c][1]]) for c in lin])

    axes, first, clipped = {}, None, False
    for k, c in enumerate(lin):
        ax = fig.add_subplot(gs[k // 3, k % 3], sharey=first)
        first = first or ax
        axes[c] = ax
    axes["turn"] = fig.add_subplot(gs[1, 2])

    for c in CONDS:
        ax = axes[c]
        s, target, unit = series[c]
        t = np.arange(s.shape[0]) * dt
        mean = s.mean(axis=1)
        lo = np.percentile(s, 10, axis=1)
        hi = np.percentile(s, 90, axis=1)

        ax.fill_between(t, lo, hi, color=L_COLOR, alpha=0.16, linewidth=0)
        ax.plot(t, mean, color=L_COLOR, linewidth=1.7)
        ax.axhline(target, color=INK, linewidth=1.1, linestyle="--", alpha=0.6)
        ax.axvspan(0, WARM * dt, color="#e9e7e0", linewidth=0)

        ylim = _ylim_from([mean[WARM:], np.array([target])]) if c == "turn" else lin_lim
        ax.set_ylim(*ylim)
        if lo[WARM:].min() < ylim[0] or hi[WARM:].max() > ylim[1]:
            clipped = True

        err = abs(float(mean[WARM:].mean()) - target)
        ax.set_title(f"[{KO[c]}]  목표 {target:+.2f} {unit}   오차 {err:.3f} {unit}",
                     fontsize=9.5, loc="left", pad=5)
        ax.set_ylabel(unit, fontsize=8.5)
        ax.set_xlabel("시간 [s]", fontsize=8.5)
        ax.grid(alpha=0.25)
        ax.margins(x=0)

    lines = [
        f"{ver} · 방향마다 500스텝 · 굵은 선 = 4개 환경 평균 · 띠 = 10~90 백분위 · 파선 = 명령",
        "회색 = 과도구간(오차·축 계산에서 제외) · 앞·뒤·좌·우·정지는 y축 공유, 회전만 rad/s 라 분리",
    ]
    if clipped:
        lines.append("띠 일부가 축 밖으로 잘림 — 축은 평균선 기준이다 "
                     "(한 환경의 순간 휘청임이 축을 지배하지 않게)")
    _save(fig, out, f"{ver}_6방향_속도추종.png", "명령 속도 추종", lines,
          panel_title_lines=1)


def plot_gait(j, ref_contact, out, ver):
    feet = j["feet"]
    phase = j["phase"].astype(int)
    period = int(j["gait_period_steps"])
    dt = float(j["ctrl_dt"])
    env, T = 0, 240
    t = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(13, 4.0))
    for idx, side, color, y in [(0, "왼발", L_COLOR, 1), (1, "오른발", R_COLOR, 0)]:
        rc = ref_contact[phase[:T, env] % period, idx] > 0.5
        ax.broken_barh([(t[i], dt) for i in range(T) if rc[i]], (y - 0.40, 0.80),
                       facecolors=color, alpha=0.22, edgecolor="none")
        pc = feet[:T, env, idx] > 0.5
        ax.broken_barh([(t[i], dt) for i in range(T) if pc[i]], (y - 0.22, 0.44),
                       facecolors=color, edgecolor="none")
        ax.text(-0.012, y, side, ha="right", va="center", fontsize=9.5,
                color=INK, transform=ax.get_yaxis_transform())

    ax.set_yticks([]); ax.set_ylim(-0.75, 1.75)
    ax.set_xlabel("시간 [s]", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25); ax.margins(x=0)

    pol = feet[WARM:, :, :] > 0.5
    ref = ref_contact[phase[WARM:] % period, :] > 0.5
    agree = float((pol == ref).mean())
    dl, dr = float(pol[:, :, 0].mean()), float(pol[:, :, 1].mean())
    rl, rr = float(ref[:, :, 0].mean()), float(ref[:, :, 1].mean())

    lines = [
        f"{ver} · 앞 0.15 m/s · 진한 막대 = 정책 / 옅은 막대 = 레퍼런스 · 색칠 = 발이 지면에 닿아 있음",
        f"보행 주기 {period}스텝 ({period*dt:.2f} s) · 디딤 비율 정책 왼 {dl*100:.0f}% / 오른 {dr*100:.0f}%"
        f" · 레퍼런스 왼 {rl*100:.0f}% / 오른 {rr*100:.0f}%",
        f"정책과 레퍼런스의 접지 상태가 일치하는 비율 {agree*100:.0f}%",
    ]
    _save(fig, out, f"{ver}_보행_접지.png", "발 접지 — 정책과 레퍼런스", lines)


def _shift_half(y, period):
    """한 주기로 접은 곡선을 반주기만큼 옮긴다 (주기 27이 홀수라 13.5스텝 보간).

    두 다리는 반주기 어긋나 걷는다. 이 정렬 없이 좌우를 빼면 비대칭이 아니라
    위상차를 재게 된다 — 레퍼런스(정의상 좌우 대칭)로 확인한 값:
    이동 없음 hip_roll 14.0도 / 발목 9.8도, 반주기 이동 2.8도 / 2.2도.
    """
    x = np.arange(period, dtype=float)
    return np.interp((x + period / 2.0) % period, x, y, period=period)


def plot_joints(j, out, ver):
    q, qr = j["qpos"], j["qref"]
    names = [str(x) for x in j["leg_names"]]
    period = int(j["gait_period_steps"])
    idx = j["phase"].astype(int) % period   # phase 는 0..period-1 정수 카운터다
    DEG = 180 / np.pi

    def fold(arr):
        o = np.full((period, arr.shape[-1]), np.nan)
        for b in range(period):
            m = idx[WARM:] == b
            if m.any():
                o[b] = arr[WARM:][m].mean(axis=0)
        return o

    qa, qra = fold(q), fold(qr)
    x = np.arange(period) / period * 100

    order = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
    fig, axs = plt.subplots(1, 5, figsize=(15, 5.2))
    for ax, jn in zip(axs, order):
        li, ri = names.index(f"left_{jn}"), names.index(f"right_{jn}")
        s = RIGHT_SIGN[jn]
        # 오른쪽: 부호 반전 + 반주기 이동해야 왼쪽과 같은 자리에 온다
        rp = _shift_half(qa[:, ri], period) * s
        rr = _shift_half(qra[:, ri], period) * s
        for yy, color, side in [(qa[:, li], L_COLOR, "왼"), (rp, R_COLOR, "오른")]:
            ax.plot(x, yy * DEG, color=color, linewidth=1.8, label=f"{side} 정책")
        for yy, color, side in [(qra[:, li], L_COLOR, "왼"), (rr, R_COLOR, "오른")]:
            ax.plot(x, yy * DEG, color=color, linewidth=1.2, linestyle="--",
                    alpha=0.6, label=f"{side} 레퍼런스")
        asym = float(np.nanmean(np.abs(qa[:, li] - rp))) * DEG
        floor = float(np.nanmean(np.abs(qra[:, li] - rr))) * DEG  # 지표의 바닥
        rms = float(np.sqrt(np.nanmean(np.concatenate(
            [(qa[:, li] - qra[:, li]) ** 2, (qa[:, ri] - qra[:, ri]) ** 2]))))
        ax.set_title(f"{JOINT_KO[jn]}\n추종 RMS {rms:.3f} rad ({rms*DEG:.1f}°)\n"
                     f"좌우 차이 {asym:.1f}° (레퍼런스 {floor:.1f}°)",
                     fontsize=9, loc="left", pad=6)
        ax.set_xlabel("보행 주기 [%]", fontsize=8.5)
        ax.grid(alpha=0.25); ax.margins(x=0)
    axs[0].set_ylabel("관절각 [°]", fontsize=8.5)
    axs[0].legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.9)

    lines = [
        f"{ver} · 한 보행 주기로 접어 평균 · 실선 = 정책 / 파선 = 레퍼런스, 같은 색끼리 짝",
        "오른쪽은 부호 반전(hip_yaw · 무릎 · 발목) 후 반주기 이동해 겹쳤다 — 두 다리는 반주기 어긋나 걷는다",
        "부호는 derive_mirror.py 유도값을 URDF 원점 rpy 와 실측 홈 자세로 교차 확인 · "
        "정렬이 맞으므로 두 선이 겹칠수록 좌우 대칭이다",
    ]
    _save(fig, out, f"{ver}_관절위치.png", "관절 궤적 — 좌우 겹침", lines,
          panel_title_lines=3)


def plot_path(g, out, ver):
    order = [c for c in CONDS if c != "stop"]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for i, c in enumerate(order):
        lat = np.abs(g[f"{c}__path_err"][WARM:, :, 0]) * 1000
        m, s = float(lat.mean()), float(lat.std())
        ax.plot([max(m - s, 0), m + s], [i, i], color=L_COLOR, alpha=0.30,
                linewidth=7, solid_capstyle="round")
        ax.scatter([m], [i], s=62, color=L_COLOR, zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.text(m, i + 0.30, f"{m:.0f} mm", ha="center", fontsize=9, color=INK)

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([KO[c] for c in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("path frame 대비 횡방향 이탈 [mm] — 낮을수록 직진", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25); ax.set_xlim(left=0); ax.margins(y=0.2)

    lines = [f"{ver} · 점 = 평균 · 띠 = ±1σ · 과도구간 {WARM}스텝 제외",
             "정지 명령은 경로가 정의되지 않아 뺐다"]
    _save(fig, out, f"{ver}_경로이탈.png", "경로 이탈", lines)


def load_ref_contact(pkl, cmd):
    """레퍼런스 접지를 pkl 에서 직접 복원한다 (위상 -> 접지, GPU 불필요).

    joint npz 는 레퍼런스 접지를 저장하지 않지만, 레퍼런스는 (명령, 위상)의
    결정적 함수라 pkl 만 있으면 정확히 재현된다 — 지어내는 것이 아니다.
    접지는 레퍼런스 프레임의 28:30 인덱스다 (imit_internals2.py 와 동일).
    """
    import torch
    from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import (
        PolyReferenceMotion,
    )
    prm = PolyReferenceMotion(pkl, device="cpu")
    n = prm.nb_steps_in_period
    f = prm.get_reference_motion(torch.full((n,), cmd[0]), torch.full((n,), cmd[1]),
                                 torch.full((n,), cmd[2]), torch.arange(n))
    return f[:, 28:30].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v28")
    ap.add_argument("--src", default=os.path.expanduser("~/odm_out"))
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/openduck_graphs"))
    ap.add_argument("--ref", default=None, help="레퍼런스 pkl (기본: 버전에 맞춰 추정)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    g = np.load(os.path.join(args.src, f"gait_{args.ver}.npz"))
    j = np.load(os.path.join(args.src, f"joint_{args.ver}.npz"))
    print(f"[info] 체크포인트: {str(j['checkpoint'])}")

    ref = args.ref or os.path.join(
        os.path.dirname(__file__), "..", "..", "source", "open_duck_mini_isaaclab",
        "reference_motion", "data",
        "ref_h175.pkl" if args.ver in ("v28", "v29") else "polynomial_coefficients.pkl")
    print(f"[info] 레퍼런스:   {os.path.normpath(ref)}")
    rc = load_ref_contact(ref, (float(j["cmd_fwd"]), 0.0, 0.0))

    plot_tracking(g, args.out, args.ver)
    plot_gait(j, rc, args.out, args.ver)
    plot_joints(j, args.out, args.ver)
    plot_path(g, args.out, args.ver)


if __name__ == "__main__":
    main()
