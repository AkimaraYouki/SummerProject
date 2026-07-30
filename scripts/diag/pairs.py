"""5 mm 를 깎아먹는 링크 쌍이 몇 개인지 센다.

형상쌍 4872 개를 50 Hz 로 다 계산할 수는 없다. 소수의 쌍이 지배적이면 그것만
캡슐로 근사해 해석적으로 풀 수 있고, 넓게 퍼져 있으면 학습된 거리장이 필요하다.
"""
import numpy as np, pinocchio as pin, collections, sys
from leg_trunk_clearance import build

model, geom, pairs = build("robot/robot.urdf", "robot")
data, gdata = model.createData(), geom.createData()
d = np.load("gait_v28.npz")
names = [str(x) for x in d["leg_names"]]
slot = [model.idx_qs[model.getJointId(n)] for n in names]
gn = [model.frames[g.parentFrame].name for g in geom.geometryObjects]

cnt = collections.Counter()
tot = 0
for c in ["stop","forward","backward","left","right","turn"]:
    traj = d[f"{c}__q"][100::10, 0]
    for t in range(traj.shape[0]):
        q = pin.neutral(model)
        for j, s in enumerate(slot):
            q[s] = traj[t, j]
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geom, gdata)
        pin.computeDistances(model, data, geom, gdata, q)
        r = [(x.min_distance, i) for i, x in enumerate(gdata.distanceResults)]
        r.sort()
        tot += 1
        if r[0][0] < 0.005:                      # 위반 자세에서 누가 최소였나
            cp = geom.collisionPairs[r[0][1]]
            cnt[(gn[cp.first], gn[cp.second])] += 1
print(f"@@자세 {tot}개 · 5 mm 위반 {sum(cnt.values())}개")
print("@@지배적인 링크 쌍:")
for (a, b), n in cnt.most_common(8):
    print(f"@@  {a:26s} <-> {b:26s} {n:4d}회 ({100*n/max(sum(cnt.values()),1):.0f}%)")
