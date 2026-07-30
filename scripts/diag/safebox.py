"""레퍼런스 관절 박스 안이 전부 5 mm 이상 안전한지 검사한다.

간격은 다리 관절각만의 함수이므로 관절 공간에 안전 영역이 존재한다. 박스가
안전하면 PhysX 관절 제한으로 강제할 수 있고, 그때는 위반이 물리적으로 불가능해진다
(소프트 벌점과 달리 보장이 된다).
"""
import sys, numpy as np, pinocchio as pin
sys.path.insert(0, ".")
from leg_trunk_clearance import build, min_clearance, TRUNK_LINKS, LEG_LINKS

URDF = "robot/robot.urdf"
model, geom, pairs = build(URDF, "robot")
data, gdata = model.createData(), geom.createData()

d = np.load("gait_v28.npz")
names = [str(x) for x in d["leg_names"]]
slot = [model.idx_qs[model.getJointId(n)] for n in names]

# 레퍼런스가 실제로 쓰는 관절별 min/max
qr = np.concatenate([d[f"{c}__qr"][100:].reshape(-1, len(names))
                     for c in ["stop","forward","backward","left","right","turn"]])
lo, hi = qr.min(0), qr.max(0)
D = 180/np.pi
print("[레퍼런스 관절 박스]")
for i, n in enumerate(names):
    print(f"  {n:18s} [{lo[i]*D:+7.1f}, {hi[i]*D:+7.1f}]°")

rng = np.random.default_rng(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
worst, worst_q = 1e9, None
viol = 0
for k in range(N):
    s = rng.uniform(lo, hi)
    q = pin.neutral(model)
    for j, sl in enumerate(slot):
        q[sl] = s[j]
    c = min_clearance(model, geom, data, gdata, q)
    if c < 0.005:
        viol += 1
        if c < worst:
            worst, worst_q = c, s.copy()
print(f"\n박스 안 무작위 {N}개 · 5 mm 위반 {100*viol/N:.1f}% · 최소 간격 {worst*1000 if worst_q is not None else 999:.1f} mm")
if worst_q is not None:
    print("가장 위험한 자세:")
    for i, n in enumerate(names):
        f = (worst_q[i]-lo[i])/(hi[i]-lo[i]+1e-9)
        print(f"  {n:18s} {worst_q[i]*D:+7.1f}°  (범위의 {f*100:3.0f}%)")
