"""가정 두 개를 검증한다.
 (1) 몸통<->한쪽 정강이 거리는 그 다리 5관절만의 함수인가 (반대쪽 무관)
 (2) 좌우가 미러 대칭인가 (함수 하나로 양쪽 처리 가능)
둘 다 맞으면 장벽함수가 10차원이 아니라 5차원이 된다.
"""
import numpy as np, pinocchio as pin
from leg_trunk_clearance import build

model, geom, _ = build("robot/robot.urdf", "robot")
data, gdata = model.createData(), geom.createData()
gn = [model.frames[g.parentFrame].name for g in geom.geometryObjects]
L = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle"]
R = ["right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
SGN = np.array([-1.,1.,1.,-1.,-1.])          # 검증된 미러 부호
sl = {n: model.idx_qs[model.getJointId(n)] for n in L+R}
LEFT = {"knee_and_ankle_assembly", "knee_and_ankle_assembly_2"}

pl, pr = [], []
for i, p in enumerate(geom.collisionPairs):
    a, b = gn[p.first], gn[p.second]
    s = {a, b} & LEFT
    (pl if s else pr).append(i)

def dist(qv):
    q = pin.neutral(model)
    for n, v in qv.items(): q[sl[n]] = v
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geom, gdata)
    pin.computeDistances(model, data, geom, gdata, q)
    r = [x.min_distance for x in gdata.distanceResults]
    return min(r[i] for i in pl), min(r[i] for i in pr)

d = np.load("gait_v28.npz")
names = [str(x) for x in d["leg_names"]]
tr = np.concatenate([d[f"{c}__q"][100::30, 0] for c in
                     ["stop","forward","backward","left","right","turn"]])
iL = [names.index(n) for n in L]; iR = [names.index(n) for n in R]
rng = np.random.default_rng(0)

e1, e2 = [], []
for k in range(min(60, len(tr))):
    ql, qr_ = tr[k, iL], tr[k, iR]
    dl, _ = dist({**dict(zip(L, ql)), **dict(zip(R, qr_))})
    # (1) 반대쪽 다리를 딴 자세로 바꿔도 왼쪽 거리가 같은가
    other = tr[rng.integers(len(tr)), iR]
    dl2, _ = dist({**dict(zip(L, ql)), **dict(zip(R, other))})
    e1.append(abs(dl - dl2))
    # (2) 왼쪽 자세를 미러해 오른쪽에 넣으면 오른쪽 거리가 왼쪽과 같은가
    _, dr = dist({**dict(zip(L, ql)), **dict(zip(R, ql * SGN))})
    e2.append(abs(dl - dr))
e1, e2 = np.array(e1)*1000, np.array(e2)*1000
print(f"@@(1) 반대쪽 다리 무관성: 평균 {e1.mean():.4f} mm · 최대 {e1.max():.4f} mm")
print(f"@@(2) 좌우 미러 대칭    : 평균 {e2.mean():.4f} mm · 최대 {e2.max():.4f} mm")
