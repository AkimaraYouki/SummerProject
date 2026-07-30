"""5차원 장벽함수 h(q_leg) = 몸통<->정강이 간격 학습 데이터를 만든다.

check5d.py 로 확인: 반대쪽 다리와 무관(오차 0.0000 mm), 좌우 미러 대칭
(최대 0.22 mm). 따라서 왼다리 5관절만 훑으면 양쪽을 덮는다.

샘플 범위는 레퍼런스가 쓰는 박스를 40% 넓힌 영역이다. 필터가 필요한 곳은
로봇이 실제로 가는 근방이고, 전 범위를 훑으면 대부분이 무의미한 자세다.
"""
import sys, numpy as np, pinocchio as pin
from multiprocessing import Pool
from leg_trunk_clearance import build

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
L = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle"]
LEFT_SHIN = {"knee_and_ankle_assembly", "knee_and_ankle_assembly_2"}

_G = {}
def init():
    model, geom, _ = build("robot/robot.urdf", "robot")
    gn = [model.frames[g.parentFrame].name for g in geom.geometryObjects]
    keep = [i for i, p in enumerate(geom.collisionPairs)
            if ({gn[p.first], gn[p.second]} & LEFT_SHIN)]
    _G.update(model=model, geom=geom, data=model.createData(),
              gdata=geom.createData(), keep=keep,
              sl=[model.idx_qs[model.getJointId(n)] for n in L])

def one(s):
    m, g, d, gd = _G["model"], _G["geom"], _G["data"], _G["gdata"]
    q = pin.neutral(m)
    for j, k in enumerate(_G["sl"]):
        q[k] = s[j]
    pin.forwardKinematics(m, d, q)
    pin.updateGeometryPlacements(m, d, g, gd)
    pin.computeDistances(m, d, g, gd, q)
    r = gd.distanceResults
    return min(r[i].min_distance for i in _G["keep"])

if __name__ == "__main__":
    dd = np.load("gait_v28.npz")
    names = [str(x) for x in dd["leg_names"]]
    idx = [names.index(n) for n in L]
    qr = np.concatenate([dd[f"{c}__qr"][100:].reshape(-1, len(names))
                         for c in ["stop","forward","backward","left","right","turn"]])[:, idx]
    lo, hi = qr.min(0), qr.max(0)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * 1.4      # 레퍼런스 박스 40% 확장
    rng = np.random.default_rng(0)
    S = rng.uniform(c - half, c + half, size=(N, 5))
    with Pool(initializer=init) as p:
        D = np.array(p.map(one, S, chunksize=200))
    np.savez_compressed("h5d.npz", S=S, D=D, lo=c - half, hi=c + half, joints=np.array(L))
    print(f"@@[ok] {N}개 · 간격 min {D.min()*1000:.1f} / 중앙 {np.median(D)*1000:.1f} "
          f"/ max {D.max()*1000:.1f} mm · 5mm 미만 {100*(D<0.005).mean():.1f}%")
