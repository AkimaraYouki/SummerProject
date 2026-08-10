#!/usr/bin/env python3
"""잿슨 rl_walk_log.csv 를 심 측정과 **같은 항목**으로 잰다.

같은 정의를 쓰는 게 핵심이다. 심에서는 standstill_pose.py / smooth_sweep.py /
leg_symmetry.py 가 재는 양을 여기서 손으로 다시 계산해, v40 심 수치와 나란히
놓을 수 있게 한다.
"""
import csv, math, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/home/parksuho/.claude/jobs/75c2022c/tmp/rl_walk_log.csv"
rows = list(csv.DictReader(open(path)))
f = lambda r, k: float(r[k])

JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
ALL = ["left_" + j for j in JOINTS] + ["neck_pitch", "head_pitch", "head_yaw", "head_roll"] \
    + ["right_" + j for j in JOINTS]
# joint_order.py 에서 뽑은 거울 규칙 (FK 로 32조합 전수탐색해 얻은 것)
MIRROR = {"hip_yaw": -1.0, "hip_roll": +1.0, "hip_pitch": +1.0, "knee": -1.0, "ankle": -1.0}
LIM = {"hip_yaw": (-30, 30), "hip_roll": (-25, 25), "hip_pitch": (-30, 70),
       "knee": (-120, 120), "ankle": (-90, 90)}

d = math.degrees
n = len(rows)
t0, t1 = f(rows[0], "t"), f(rows[-1], "t")
dts = [f(rows[i + 1], "t") - f(rows[i], "t") for i in range(n - 1)]
dts_s = sorted(dts)
print("=" * 78)
print(f"잿슨 실기 로그 — {n} 스텝 · {t1 - t0:.2f} s")
print("-" * 78)
print(f"  제어 주기   평균 {sum(dts)/len(dts)*1000:5.1f} ms ({1/(sum(dts)/len(dts)):4.1f} Hz)"
      f"   중앙 {dts_s[len(dts_s)//2]*1000:5.1f} ms"
      f"   최대 {max(dts)*1000:5.1f} ms")
jit = [abs(x - sum(dts)/len(dts)) for x in dts]
print(f"  주기 흔들림 평균 {sum(jit)/len(jit)*1000:4.1f} ms   "
      f"20 ms 초과 {sum(1 for x in dts if x > 0.020)/len(dts)*100:4.1f} %")

# ── 자세 (IMU) ────────────────────────────────────────────────────────────
# proj_grav 는 몸통 좌표계에서 본 중력 단위벡터. 심의 projected_gravity_b 와 같다.
# 피치 = atan2(-gx, -gz), 롤 = atan2(gy, -gz)  (수직일 때 g=(0,0,-1) -> 0,0)
pit = [d(math.atan2(-f(r, "proj_grav_x"), -f(r, "proj_grav_z"))) for r in rows]
rol = [d(math.atan2(f(r, "proj_grav_y"), -f(r, "proj_grav_z"))) for r in rows]
mean = lambda v: sum(v) / len(v)
std = lambda v: math.sqrt(sum((x - mean(v)) ** 2 for x in v) / len(v))

# 걷는 중인지: phase 가 도는가 (정지에서는 standstill_hold 로 위상이 0 에 묶인다)
ph = [(f(r, "phase_cos"), f(r, "phase_sin")) for r in rows]
moving = [i for i in range(n) if abs(ph[i][1]) > 1e-6 or ph[i][0] < 0.999]
print(f"  위상이 도는 스텝 {len(moving)}/{n}  → {'보행' if len(moving) > n*0.5 else '정지'} 구간으로 본다")

WARM = min(30, n // 5)   # 과도구간 제외 (심에서도 150/400 을 버린다)
sl = slice(WARM, n)
print("-" * 78)
print(f"  몸통 피치   평균 {mean(pit[sl]):+7.2f}°   표준편차 {std(pit[sl]):5.2f}°"
      f"   범위 {min(pit[sl]):+.2f} ~ {max(pit[sl]):+.2f}")
print(f"  몸통 롤     평균 {mean(rol[sl]):+7.2f}°   표준편차 {std(rol[sl]):5.2f}°")
gy = [math.sqrt(f(r,"gyro_x")**2 + f(r,"gyro_y")**2 + f(r,"gyro_z")**2) for r in rows]
print(f"  몸통 각속도 RMS {math.sqrt(mean([x*x for x in gy[sl]])):6.4f} rad/s")

# ── 액션 매끄러움 ────────────────────────────────────────────────────────
# smooth_sweep.py 의 '액션요동' 과 같은 정의: 14축 1차차분 제곱합의 평균.
A = [[f(r, "action_" + j) for j in ALL] for r in rows]
k = len(ALL)
d1 = [sum((A[i][c] - A[i-1][c]) ** 2 for c in range(k)) for i in range(1, n)]
d2 = [sum((A[i][c] - 2*A[i-1][c] + A[i-2][c]) ** 2 for c in range(k)) for i in range(2, n)]
rev = sum(1 for i in range(2, n) for c in range(k)
          if (A[i][c]-A[i-1][c]) * (A[i-1][c]-A[i-2][c]) < 0) / ((n-2) * k)
print("-" * 78)
print(f"  액션 1차차분²  {mean(d1):7.4f}      2차차분²  {mean(d2):7.4f}"
      f"      비율 {mean(d2)/max(mean(d1),1e-9):4.2f}")
print(f"  액션 방향반전  {rev*100:5.1f} %   (100% = 매 스텝 뒤집힘, 0% = 한 방향)")
print("     비율 해석: 매끄러운 램프면 0, 순수 진동이면 4.")

# ── 목표 대비 실제 (모터가 밀리는가) ──────────────────────────────────────
print("-" * 78)
print(f"  {'관절':16}{'목표평균':>9}{'실제평균':>9}{'추종오차RMS':>12}{'최대':>8}{'한계여유':>9}")
sag = {}
for j in ALL:
    tg = [d(f(r, "target_" + j)) for r in rows][WARM:]
    ps = [d(f(r, "pos_" + j)) for r in rows][WARM:]
    err = [p - t for p, t in zip(ps, tg)]
    base = j.replace("left_", "").replace("right_", "")
    lim = LIM.get(base)
    margin = ""
    if lim:
        m = min(lim[1] - max(tg), min(tg) - lim[0])
        margin = f"{m:+8.1f}"
    sag[j] = math.sqrt(mean([e * e for e in err]))
    print(f"  {j:16}{mean(tg):+8.2f}°{mean(ps):+8.2f}°{sag[j]:11.2f}°{max(abs(e) for e in err):7.2f}°{margin:>9}")

# ── 좌우 대칭 ────────────────────────────────────────────────────────────
print("-" * 78)
print(f"  좌우 대칭 (거울 규칙 적용 후 관절각 차)")
print(f"  {'관절':10}{'좌':>10}{'우':>10}{'거울기대':>10}{'어긋남':>10}")
for j in JOINTS:
    L = mean([d(f(r, "pos_left_" + j)) for r in rows][WARM:])
    R = mean([d(f(r, "pos_right_" + j)) for r in rows][WARM:])
    exp = MIRROR[j] * L
    print(f"  {j:10}{L:+9.2f}°{R:+9.2f}°{exp:+9.2f}°{R - exp:+9.2f}°"
          f"{'  <<<' if abs(R - exp) > 1.0 else ''}")

# ── 접지 ─────────────────────────────────────────────────────────────────
cl = [f(r, "contact_l") for r in rows]
cr = [f(r, "contact_r") for r in rows]
tog = sum(1 for i in range(1, n) if cl[i] != cl[i-1] or cr[i] != cr[i-1]) / (n-1)
print("-" * 78)
print(f"  접지  좌 {mean(cl)*100:5.1f} %   우 {mean(cr)*100:5.1f} %   "
      f"양발 {sum(1 for i in range(n) if cl[i]>0.5 and cr[i]>0.5)/n*100:5.1f} %   "
      f"바뀜 {tog:.3f}/스텝")
print("=" * 78)
