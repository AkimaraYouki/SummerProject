"""학습 중인 런의 건강 상태를 텐서보드 이벤트만 읽어 판정한다.

왜 이벤트 파일인가: 학습 중에는 Isaac Sim을 두 개 띄울 수 없어 `odm measure`를
못 쓴다. 그래서 감시는 이미 기록된 스칼라만으로 해야 한다. Isaac Sim을 켜지
않으므로 학습에 아무 영향이 없다.

판정 기준은 상상이 아니라 **이 프로젝트가 실제로 겪은 실패**에서 왔다
(docs/training_log.md, docs/handoff/README.md):

  lr 바닥      Upstream 런에서 num_minibatches=32가 iteration당 128회 업데이트가
               되어 KL이 초과했고, adaptive 스케줄이 lr을 하한 1e-5까지 깎아
               iter 150부터 학습이 동결됐다. 곡선은 그때도 "오르는 중"이었다.
  std 붕괴     탐험이 사라지면 국소 최적에 갇힌다. 반대로 entropy가 너무 커서
               std가 0.75까지 밀려 올라가 정체한 적도 있다(BigNet).
  alive 파먹기 에피소드 길이가 상한에 붙었는데 스텝당 리워드가 정체하면,
               "넘어지지만 않는" 정책일 수 있다. 이 프로젝트의 v4 "플랭크"가
               정확히 그것이었다 — 접촉력 0.000 N으로 모든 종료를 빠져나갔다.

**중요한 한계**: 지금 환경은 리워드 총합만 기록하고 항목별 분해를 남기지 않는다
(`_get_rewards`가 7~8개 항을 더해서 반환하고 끝난다). 리워드 해킹은 "어느 항만
치솟고 나머지는 정체"로 나타나므로, 총합만으로는 확정할 수 없고 **의심 신호까지만**
잡을 수 있다. 항목별 로깅이 붙으면 이 스크립트도 항별 비교로 올려야 한다.

그리고 이 스크립트의 판정은 **곡선 기반이라 그 자체로 성공/실패 판정이 아니다**.
경고가 없다고 "잘 학습됐다"고 말하면 안 된다 — 판정은 끝난 뒤 `odm measure`와
`odm play`(눈)로 한다.
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# adaptive 스케줄의 하한. 여기에 붙어 있으면 사실상 학습이 멈춘 것이다.
LR_FLOOR = 1.1e-5
# 액션 노이즈 std 하한/상한. 아래로 가면 탐험 소멸, 위로 가면 잡음이 신호를 덮는다.
STD_COLLAPSE = 0.08
STD_TOO_WIDE = 0.75
# 에피소드 길이가 상한의 이 비율을 넘으면 "포화"로 본다.
EP_SATURATED = 0.92
# 최근 구간에서 스텝당 리워드가 이 비율보다 덜 오르면 "정체"로 본다.
PLATEAU_REL = 0.01


def _scalars(acc, tag):
    return [(s.step, s.value) for s in acc.Scalars(tag)] if tag in acc.Tags()["scalars"] else []


def _last(series, default=None):
    return series[-1][1] if series else default


def _trend(series, frac=0.3):
    """최근 frac 구간의 시작 대비 끝 변화율. 표본이 모자라면 None."""
    if len(series) < 6:
        return None
    n = max(2, int(len(series) * frac))
    a, b = series[-n][1], series[-1][1]
    if a == 0:
        return None
    return (b - a) / abs(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True, help="런 디렉터리")
    p.add_argument("--max_ep_len", type=float, default=500.0, help="에피소드 길이 상한 (스텝)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.run, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"이벤트 파일 없음: {args.run}")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()

    reward = _scalars(acc, "Train/mean_reward")
    ep_len = _scalars(acc, "Train/mean_episode_length")
    std = _scalars(acc, "Policy/mean_std")
    lr = _scalars(acc, "Loss/learning_rate")
    v_loss = _scalars(acc, "Loss/value")
    entropy = _scalars(acc, "Loss/entropy")

    if not reward:
        raise SystemExit("Train/mean_reward 가 없습니다 — 아직 첫 기록 전일 수 있습니다.")

    it = reward[-1][0]
    r, e = _last(reward), _last(ep_len, 0.0)
    per_step = r / e if e else float("nan")

    # 스텝당 리워드 시계열 (버전 간 비교는 못 해도, 한 런 안에서는 유효하다)
    ps_series = [(s, rv / ev) for (s, rv), (_, ev) in zip(reward, ep_len) if ev] if ep_len else []

    print(f"[{os.path.basename(args.run)}]  iter {it}")
    print(f"  스텝당 리워드   {per_step:.4f}   (총합 {r:.2f} / 에피 {e:.1f})")
    print(f"  액션 std        {_last(std, float('nan')):.3f}")
    print(f"  learning_rate   {_last(lr, float('nan')):.2e}")
    print(f"  value loss      {_last(v_loss, float('nan')):.4f}")
    print(f"  entropy         {_last(entropy, float('nan')):.3f}")

    warn = []

    cur_lr = _last(lr)
    if cur_lr is not None and cur_lr <= LR_FLOOR:
        warn.append(
            f"learning_rate 가 하한({cur_lr:.1e})에 붙었다 — adaptive 스케줄이 KL 초과로 "
            "계속 깎아내린 것이다. Upstream 런이 iter 150부터 이렇게 동결됐다. "
            "곡선이 오르는 것처럼 보여도 실질 학습은 멈춘 상태일 수 있다."
        )

    cur_std = _last(std)
    if cur_std is not None:
        if cur_std < STD_COLLAPSE:
            warn.append(f"액션 std 붕괴({cur_std:.3f}) — 탐험이 사라졌다. 국소 최적 의심.")
        elif cur_std > STD_TOO_WIDE:
            warn.append(
                f"액션 std 과대({cur_std:.3f}) — 잡음이 관절 추종 오차보다 커서 "
                "학습이 정체할 수 있다(BigNet에서 겪음)."
            )

    ep_ratio = e / args.max_ep_len if args.max_ep_len else 0.0
    ps_trend = _trend(ps_series)
    if ep_ratio >= EP_SATURATED and ps_trend is not None and ps_trend < PLATEAU_REL:
        warn.append(
            f"에피소드 길이가 상한에 포화({e:.0f}/{args.max_ep_len:.0f})했는데 스텝당 "
            f"리워드는 최근 {ps_trend*100:+.1f}%로 정체다 — '넘어지지만 않는' 정책일 수 "
            "있다(v4 '플랭크'와 같은 양상). 확정하려면 항목별 리워드가 필요하다."
        )

    if ps_trend is not None and ps_trend < -0.05:
        warn.append(f"스텝당 리워드가 최근 {ps_trend*100:+.1f}%로 하락 중이다.")

    print()
    if warn:
        print("  경고:")
        for w in warn:
            print(f"   - {w}")
    else:
        print("  경고 없음.")
    print(
        "\n  주의: 곡선만 본 판정이다. 리워드 항목별 분해가 기록되지 않아 리워드 해킹은 "
        "의심까지만 잡힌다. 성공 판정은 학습 후 odm measure 와 odm play(눈)로 한다."
    )


if __name__ == "__main__":
    main()
