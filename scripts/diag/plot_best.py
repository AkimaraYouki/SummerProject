"""최고 성능 정책의 보행을 그림으로 남긴다.

`odm measure` 가 떨어뜨린 npz 만 읽는다 — Isaac Sim 도 GPU 도 쓰지 않으므로
학습이 도는 중에도 돌릴 수 있다.

    isaaclab.sh -p scripts/diag/plot_best.py --ver v28 --out ~/Desktop/openduck_graphs

그리는 것 (각각 데이터의 역할에 맞춰 형태를 골랐다):

  1. 명령 추종      6방향 각각 시간에 따른 실제 속도 + 명령선.
                    막대로 평균만 찍으면 "얼마나 흔들리는지"가 사라진다.
  2. 보행 다이어그램 좌우 발 접지 패턴과 레퍼런스의 그것.
                    주기성·듀티비·좌우 위상차가 한 장에 보인다.
  3. 관절 궤적       다리 10관절의 정책 vs 레퍼런스, 한 보행 주기 평균.
  4. 경로 이탈       path frame 기준 횡방향 오차. 직진성의 직접 지표.

색은 눈대중하지 않고 dataviz 검증기를 통과한 값을 쓴다
(slot1 #2a78d6, slot2 #eb6834 — CVD ΔE 24.7, 전 항목 PASS).
레퍼런스는 "따라가야 할 목표"라 계열색을 주지 않고 중성 파선으로 둔다 —
색은 실측값 하나에만 쓴다.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# dataviz 기본 팔레트 (검증 통과)
S1 = "#2a78d6"   # 실측 (정책)
S2 = "#eb6834"   # 두 번째 정체성 (오른발)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

CMDS = ["stop", "forward", "backward", "left", "right", "turn"]
CMD_KO = {"stop": "정지", "forward": "전진", "backward": "후진",
          "left": "좌", "right": "우", "turn": "회전"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "font.size": 9,
        # 한글이 들어가므로 한글 글리프가 있는 폰트를 앞에 둔다. DejaVu Sans 만
        # 쓰면 라벨이 통째로 두부(□)가 된다 — 처음 돌렸을 때 실제로 그랬다.
        "font.family": ["NanumBarunGothic", "NanumGothic", "Noto Sans CJK JP", "DejaVu Sans"],
        # 한글 폰트는 유니코드 마이너스(U+2212)를 대개 갖고 있지 않다. ASCII 로 쓴다.
        "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _finish(fig, path, title, subtitle, note=None):
    """제목 블록을 figure 위에 별도로 얹는다.

    처음엔 suptitle 과 부제를 가까운 y 에 두었다가 서로 겹쳤다. 축 영역을
    rect 로 눌러 두고 그 위 여백에만 글자를 놓는다."""
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13,
             fontweight="bold", color=INK)
    fig.text(0.012, 0.930, subtitle, ha="left", va="top", fontsize=8.5, color=MUTED)
    if note:
        fig.text(0.012, 0.885, note, ha="left", va="top", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[ok] {path}")


def plot_tracking(g, out, tag):
    """6방향 × (실제 속도 vs 명령). 축은 명령 성분에 맞춰 고른다."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    dt = 0.02
    for ax, name in zip(axes.ravel(), CMDS):
        cmd = g[f"{name}__cmd"]
        vb = g[f"{name}__v_base"]        # [T, envs, 2]
        wb = g[f"{name}__w_base"]        # [T, envs]
        turning = abs(cmd[2]) > 1e-6
        if turning:
            series, target, unit, lab = wb, cmd[2], "rad/s", "요 각속도"
        elif abs(cmd[1]) > 1e-6:
            series, target, unit, lab = vb[:, :, 1], cmd[1], "m/s", "횡방향 속도"
        else:
            series, target, unit, lab = vb[:, :, 0], cmd[0], "m/s", "전진 속도"

        t = np.arange(series.shape[0]) * dt
        mean = series.mean(axis=1)
        # min~max 를 쓰면 한 환경의 한 번 휘청임이 축 전체를 장악해
        # 정작 추종이 눌린다. 10~90 백분위로 바꾸고 캡션에 밝힌다.
        lo = np.percentile(series, 10, axis=1)
        hi = np.percentile(series, 90, axis=1)

        ax.fill_between(t, lo, hi, color=S1, alpha=0.14, linewidth=0)
        ax.plot(t, mean, color=S1, linewidth=1.6)
        ax.axhline(target, color=INK2, linewidth=1.2, linestyle=(0, (5, 3)))
        ax.axvspan(0, 100 * dt, color=GRID, alpha=0.55, linewidth=0)

        err = abs(mean[100:].mean() - target)
        ax.set_title(f"{CMD_KO[name]}   목표 {target:+.2f} {unit}   오차 {err:.3f}",
                     fontsize=9.5, loc="left", pad=6)
        ax.set_ylabel(f"{lab} [{unit}]", fontsize=8)
        ax.grid(axis="y", alpha=0.9)
        ax.margins(x=0)

    for ax in axes[1]:
        ax.set_xlabel("시간 [s]", fontsize=8)

    note = ("굵은 선 = 4개 환경 평균 · 옅은 띠 = 10~90 백분위 · 파선 = 명령 · "
            "회색 구간 = 과도구간(오차 계산에서 제외) · "
            "아래로 튀는 구간은 한 환경이 잠깐 휘청인 것")
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    _finish(fig, os.path.join(out, f"1_command_tracking_{tag}.png"),
            "명령 추종", f"{tag} · 6방향 각 500스텝 · 낮은 오차가 좋음", note)


def plot_gait(j, out, tag):
    """발 접지 다이어그램 + 레퍼런스 위상.

    처음엔 위상에서 레퍼런스 접지를 복원해 아래 행에 그렸는데, `phase` 가
    라디안이 아니라 **0..period-1 정수 카운터**(_imitation_i)라 그 행은 의미
    없는 값이었다. 레퍼런스의 실제 접지는 이 npz 에 없다(레퍼런스 프레임의
    28:30 인덱스인데 joint_periodicity 가 저장하지 않는다). 지어내는 대신
    가지고 있는 위상 신호를 그대로 그린다 — "정책의 스텝이 레퍼런스 위상에
    물려 있는가" 라는 질문에는 이쪽이 정직하게 답한다.
    """
    feet = j["feet"]            # [T, envs, 2]  1 = 접지
    phase = j["phase"]          # [T, envs]  0..period-1 정수
    period = int(j["gait_period_steps"])
    dt = float(j["ctrl_dt"])
    env = 0
    T = 240                     # 약 9주기
    t = np.arange(T) * dt

    fig, axes = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.5})

    ax = axes[0]
    for idx, label, color, y in [(0, "왼발", S1, 1), (1, "오른발", S2, 0)]:
        c = feet[:T, env, idx] > 0.5
        ax.broken_barh([(t[i], dt) for i in range(T) if c[i]], (y - 0.34, 0.68),
                       facecolors=color, edgecolor="none")
        ax.text(-0.06, y, label, ha="right", va="center", fontsize=9,
                color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks([]); ax.set_ylim(-0.6, 1.6)

    ax = axes[1]
    ax.plot(t, phase[:T, env], color=MUTED, linewidth=1.3, drawstyle="steps-post")
    ax.set_ylabel("레퍼런스 위상", fontsize=8, rotation=0, ha="right",
                  va="center", labelpad=12)
    ax.set_yticks([0, period - 1])
    ax.set_xlabel("시간 [s]", fontsize=8)

    duty_l = (feet[100:, :, 0] > 0.5).mean()
    duty_r = (feet[100:, :, 1] > 0.5).mean()
    both = ((feet[100:, :, 0] > 0.5) & (feet[100:, :, 1] > 0.5)).mean()
    air = ((feet[100:, :, 0] < 0.5) & (feet[100:, :, 1] < 0.5)).mean()

    for ax in axes:
        ax.margins(x=0); ax.grid(axis="x", alpha=0.7)

    note = (f"보행 주기 {period}스텝 ({period*dt:.2f} s) · "
            f"디딤 비율 왼발 {duty_l*100:.0f}% / 오른발 {duty_r*100:.0f}% · "
            f"양발 동시 접지 {both*100:.0f}% · 양발 공중 {air*100:.0f}%")
    fig.tight_layout(rect=[0.055, 0, 1, 0.80])
    _finish(fig, os.path.join(out, f"2_gait_diagram_{tag}.png"),
            "보행 다이어그램", f"{tag} · 전진 0.15 m/s · 색칠 = 발이 지면에 닿아 있음", note)


def plot_joints(j, out, tag):
    """다리 10관절, 한 주기 평균. 정책 vs 레퍼런스."""
    q, qr = j["qpos"], j["qref"]        # [T, envs, 10]
    names = [str(x) for x in j["leg_names"]]
    period = int(j["gait_period_steps"])
    phase = j["phase"]

    # 위상으로 접어 한 주기 평균을 만든다 (시간축 평균은 주기가 섞여 뭉개진다).
    # phase 는 라디안이 아니라 0..period-1 정수 카운터(_imitation_i)다.
    # 처음에 라디안으로 보고 mod 2pi 를 씌웠다가 binning 이 통째로 틀렸다.
    bins = period
    idx = phase.astype(int) % bins
    qa = np.full((bins, 10), np.nan)
    qra = np.full((bins, 10), np.nan)
    qs = np.full((bins, 10), np.nan)
    for b in range(bins):
        m = idx[100:] == b
        if m.any():
            qa[b] = q[100:][m].mean(axis=0)
            qra[b] = qr[100:][m].mean(axis=0)
            qs[b] = q[100:][m].std(axis=0)

    x = np.arange(bins) / bins * 100
    fig, axes = plt.subplots(2, 5, figsize=(13, 4.6), sharex=True)
    DEG = 180 / np.pi
    for k, (ax, nm) in enumerate(zip(axes.ravel(), names)):
        ax.fill_between(x, (qa[:, k] - qs[:, k]) * DEG, (qa[:, k] + qs[:, k]) * DEG,
                        color=S1, alpha=0.15, linewidth=0)
        ax.plot(x, qra[:, k] * DEG, color=INK2, linewidth=1.2, linestyle=(0, (5, 3)))
        ax.plot(x, qa[:, k] * DEG, color=S1, linewidth=1.7)
        err = np.nanmean(np.abs(qa[:, k] - qra[:, k])) * DEG
        ax.set_title(f"{nm.replace('_', ' ')}\n오차 {err:.1f}°", fontsize=8.5,
                     loc="left", pad=4, color=INK)
        ax.grid(axis="y", alpha=0.9)
        ax.margins(x=0)
        if k % 5 == 0:
            ax.set_ylabel("관절각 [°]", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("보행 주기 [%]", fontsize=8)

    total = np.nanmean(np.abs(qa - qra)) * DEG
    note = (
             f"실선 = 정책 · 파선 = 레퍼런스 · 띠 = ±1σ · "
             f"전체 평균 관절 오차 {total:.1f}°")
    fig.tight_layout(rect=[0, 0, 1, 0.845])
    _finish(fig, os.path.join(out, f"3_joint_tracking_{tag}.png"),
            "관절 궤적 — 정책 vs 레퍼런스", f"{tag} · 한 보행 주기로 접어 평균", note)


def plot_path(g, out, tag):
    """path frame 기준 횡방향 이탈. 직진성의 직접 지표."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    dt = 0.02
    order = ["forward", "backward", "left", "right", "turn"]
    xs, means, spans = [], [], []
    for name in order:
        pe = g[f"{name}__path_err"]      # [T, envs, 3]  0 = 횡방향 오차 [m]
        lat = np.abs(pe[100:, :, 0]) * 1000     # mm
        xs.append(CMD_KO[name])
        means.append(lat.mean())
        spans.append((lat.mean() - lat.std(), lat.mean() + lat.std()))

    y = np.arange(len(xs))
    for i, (lo, hi) in enumerate(spans):
        ax.plot([max(lo, 0), hi], [y[i], y[i]], color=S1, alpha=0.3, linewidth=6,
                solid_capstyle="round")
    ax.scatter(means, y, s=64, color=S1, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for i, m in enumerate(means):
        ax.text(m, y[i] + 0.28, f"{m:.0f} mm", ha="center", fontsize=8.5, color=INK)

    ax.set_yticks(y); ax.set_yticklabels(xs, fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("path frame 대비 횡방향 이탈 [mm] — 낮을수록 직진", fontsize=8)
    ax.grid(axis="x", alpha=0.9)
    ax.set_xlim(left=0)
    ax.margins(y=0.18)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    _finish(fig, os.path.join(out, f"4_path_drift_{tag}.png"),
            "경로 이탈", f"{tag} · 명령 속도를 적분한 경로 대비",
            "점 = 평균 · 띠 = ±1σ · 정지 명령은 경로가 정의되지 않아 제외")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v28")
    ap.add_argument("--src", default=os.path.expanduser("~/odm_out"))
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/openduck_graphs"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    style()

    g = np.load(os.path.join(args.src, f"gait_{args.ver}.npz"))
    j = np.load(os.path.join(args.src, f"joint_{args.ver}.npz"))
    print(f"[info] 체크포인트: {str(j['checkpoint'])}")

    plot_tracking(g, args.out, args.ver)
    plot_gait(j, args.out, args.ver)
    plot_joints(j, args.out, args.ver)
    plot_path(g, args.out, args.ver)


if __name__ == "__main__":
    main()
