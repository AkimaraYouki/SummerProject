"""관절각 CSV -> 몸통 기준 발 궤적. **보폭과 발 들림 높이를 실측한다.**

    python3 scripts/diag/foot_traj.py sim.csv real.csv ...

심(`play_fixed_cmd --track-csv`)과 실기(`~/rl_walk_log.csv`) 둘 다 읽는다.
URDF FK 는 scripts/leg_fk.py (의존성 numpy 뿐).

## 왜

2026-08-13, 사용자가 "보폭이 너무 크거나 발을 너무 위로 들어서 그런 걸
수도 있다" 고 했다. 재 보니 맞았다.

    다리 마디      78.65 mm x 2 = 157 mm
    보폭(실기)     75.7 / 71.1 mm
    placo 설정     walk_foot_height = 0.04 (40 mm)

⚠️ **2026-08-14 정정 — 이 스크립트가 내는 "발 들림" 은 발 여유가 아니다.**
여기서 재는 것은 몸통 기준 좌표에서 발 하나의 z 범위인데, 거기에는

  * 발이 실제로 뜨는 양
  * 몸통이 위아래로 출렁이는 양 (실측 표준편차 4.6~5.3 mm)
  * 몸통이 기울며 발이 상대적으로 움직이는 양

이 전부 섞인다. 이 값으로 "다리 길이의 23 % 를 든다" 고 며칠 판단했는데
틀렸다. **스윙 발이 접지 발보다 얼마나 뜨는지**로 다시 재면:

    v48  중앙값 5.2 mm (다리의 3.3 %) · 스윙당 최고 14.2 mm
    v51  중앙값 4.5 mm · 18.3 mm
    v52  중앙값 4.1 mm · 19.4 mm

사람 보행(1~2 %)에 가깝다. **발 높이는 이미 문제가 아니었다.** 진짜 문제는
좌우다 — 세로로 14 mm 뜨는 동안 옆으로 28~41 mm 간다.
`scripts/diag/gait_quality.py` 의 스윙 지표를 볼 것.

그리고 **보폭은 줄일 수 없다** — 76 mm x (1/0.54 s) = 0.14 m/s 로 명령 속도
0.15 와 맞는다. 속도가 보폭을 정한다. **발 들림만 자유 변수다.**

스윙 0.18 초에 36 mm 를 올렸다 내리면 평균 0.4 m/s, 피크 약 0.63 m/s 로
착지한다. 이것이 착지 충격 2.4 g, 모터 속도 한계 상시 포화, roll +-10 도
(무게를 한쪽 다리로 완전히 옮겨야 한다) 의 공통 원인이다.

원본 리포 비교: v1(open_duck_mini) 은 walk_foot_height 0.03 / feet_spacing
0.14, v2 는 0.04 / 0.18 이다.
"""
import csv, sys, math
sys.path.insert(0, "scripts")
import numpy as np
from leg_fk import foot_in_trunk

NAMES = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
         "neck_pitch","head_pitch","head_yaw","head_roll",
         "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]

def load(path, pre="pos_", filt=None):
    rows = list(csv.DictReader(open(path)))
    if filt: rows = [r for r in rows if filt(r)]
    out = []
    for r in rows:
        try:
            ja = {n: float(r[pre + n]) for n in NAMES}
        except (KeyError, ValueError):
            continue
        out.append(foot_in_trunk(ja))
    return out

def report(traj, lab):
    if len(traj) < 30:
        print(f"  {lab}: 샘플 부족 {len(traj)}"); return
    for side in ("left", "right"):
        P = np.array([t[side] for t in traj])
        x, y, z = P[:,0], P[:,1], P[:,2]
        # 가장 낮은 z 를 접지면으로 본다 (5 퍼센타일로 이상치 배제)
        ground = np.percentile(z, 5)
        lift = np.percentile(z, 95) - ground
        stride = np.percentile(x, 95) - np.percentile(x, 5)
        lateral = np.percentile(y, 95) - np.percentile(y, 5)
        print(f"  {lab:10} {side:5}  발들림 {lift*1000:5.1f} mm   보폭(x) {stride*1000:5.1f} mm"
              f"   좌우폭(y) {lateral*1000:5.1f} mm   발간격 {abs(np.mean(y))*1000:5.1f} mm")

def f(r,k):
    try: return float(r[k])
    except: return float('nan')

print("몸통 기준 발 궤적 (URDF FK)")
J = "/home/parksuho/.claude/jobs/75c2022c/tmp"
report(load(f"{J}/sim_v46_body.csv", "pos_", lambda r: float(r['t'])>4.0), "심 v46")
report(load(f"{J}/sim_v46_body.csv", "goal_", lambda r: float(r['t'])>4.0), "심 목표")
report(load(f"{J}/rl_walk_log5.csv", "pos_", lambda r: abs(f(r,'cmd_vx'))>0.05), "실기")
print()
print("  발들림 = 보행 중 발이 지면에서 뜨는 최대 높이")
print("  placo 설정 walk_foot_height = 40 mm")
