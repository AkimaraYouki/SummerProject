"""여러 런을 **같은 iteration에서** 맞대어 본다.

왜 필요한가. 학습 중에는 `odm measure`를 못 쓰므로(Isaac Sim 두 개 금지) 진행
중인 실험이 기준선을 이기고 있는지 알 방법이 곡선뿐이다. 그런데 "지금 0.51"
같은 최종값 비교는 무의미하다 — 런마다 iteration이 다르면 그냥 더 오래 돈 쪽이
높다. 같은 iteration에서 비교해야 한다.

**주의 두 가지.**

1. rsl-rl 2.x 로 돌린 v25 이하와 5.0.1 로 돌린 v26 이상은 같은 그래프에 놓으면
   안 된다. PPO 구현이 세 세대 다르다. 이 스크립트는 막지 않으니 부르는 쪽이
   지킬 것.
2. 리워드 항이 다른 버전끼리는 총합 비교가 무의미하다("스케일 착시" — 인수인계
   문서 참고). 항 구성이 같을 때만 총합을 믿고, 아니면 항목별로 본다.
   구성이 다르면 아래 표에서 한쪽만 값이 있는 항으로 드러난다.

Isaac Sim 을 켜지 않으므로 학습 중에도 그냥 돌려도 된다.
"""

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"이벤트 파일 없음: {run_dir}")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def series(acc, tag):
    if tag not in acc.Tags()["scalars"]:
        return {}
    return {s.step: s.value for s in acc.Scalars(tag)}


def at(step_map, it):
    """it 이하의 가장 가까운 기록. 없으면 None."""
    keys = [k for k in step_map if k <= it]
    return step_map[max(keys)] if keys else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="런 디렉터리들 (첫 번째가 기준선)")
    p.add_argument("--at", type=int, nargs="*", default=None,
                   help="비교할 iteration들. 없으면 공통 구간을 5등분")
    args = p.parse_args()

    accs = [load(r) for r in args.runs]
    names = [os.path.basename(r).split("_imitation_")[-1] or os.path.basename(r) for r in args.runs]

    rewards = [series(a, "Train/mean_reward") for a in accs]
    eplens = [series(a, "Train/mean_episode_length") for a in accs]
    per_step = [
        {s: rewards[i][s] / eplens[i][s] for s in rewards[i] if eplens[i].get(s)}
        for i in range(len(accs))
    ]

    last_common = min(max(ps) for ps in per_step if ps)
    its = args.at or [int(last_common * f / 5) for f in range(1, 6)]

    w = 11
    print("\n스텝당 리워드 (같은 iteration에서 비교)")
    print(f"{'iter':>7} " + "".join(f"{n:>{w}}" for n in names) + f"{'차이':>{w}}")
    print("-" * (7 + w * (len(names) + 1)))
    for it in its:
        vals = [at(ps, it) for ps in per_step]
        row = f"{it:>7} " + "".join(f"{v:>{w}.4f}" if v is not None else f"{'-':>{w}}" for v in vals)
        if vals[0] is not None and vals[-1] is not None and vals[0] != 0:
            row += f"{(vals[-1] - vals[0]) / abs(vals[0]) * 100:>{w-1}.1f}%"
        print(row)

    # 항목별 — 마지막 공통 iteration 에서
    term_sets = [
        {t.split("/", 1)[1] for t in a.Tags()["scalars"] if t.startswith("Episode_Reward/")}
        for a in accs
    ]
    all_terms = sorted(set().union(*term_sets)) if term_sets else []
    if all_terms:
        it = int(last_common)
        print(f"\n리워드 항목별 (iter {it})")
        print(f"{'항목':<20}" + "".join(f"{n:>{w}}" for n in names))
        print("-" * (20 + w * len(names)))
        for term in all_terms:
            vals = [at(series(a, f"Episode_Reward/{term}"), it) for a in accs]
            print(f"{term:<20}" + "".join(f"{v:>{w}.4f}" if v is not None else f"{'-':>{w}}" for v in vals))

        # 항목이 통째로 없는 런과, 항 구성이 실제로 다른 런은 전혀 다른 이야기다.
        # v26 은 전자다 — 리워드는 v27 과 같고 항목별 로깅이 그때 없었을 뿐이라
        # 총합 비교는 유효하다. 이걸 뭉뚱그려 "구성이 다르다"고 경고했다가
        # v27 비교를 스스로 깎아내릴 뻔했다 (2026-07-30).
        no_terms = [names[i] for i, s in enumerate(term_sets) if not s]
        with_terms = [i for i, s in enumerate(term_sets) if s]
        if no_terms:
            print(f"\n  참고: {', '.join(no_terms)} 에는 항목별 기록이 없다(v26 이전 — 당시 로깅 없음).")
            print("        리워드 구성이 다른 것이 아니므로 위 총합 비교는 유효하다.")
        if len(with_terms) > 1 and len({frozenset(term_sets[i]) for i in with_terms}) > 1:
            print("\n  ! 기록이 있는 런들끼리 항 구성이 다르다 — 총합 비교는 무의미하다")
            print("    (인수인계 문서의 '스케일 착시'). 항목별로만 볼 것.")

    print()


if __name__ == "__main__":
    main()
