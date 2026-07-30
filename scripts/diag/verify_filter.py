"""필터를 통과한 자세를 **정확 메시**로 검증한다.

학습된 h 로 판정한 것을 학습된 h 로 확인하면 검증이 아니다. hppfcl 로 실제
간격을 재서 5 mm 를 지키는지 본다.
"""
import numpy as np, pinocchio as pin
from leg_trunk_clearance import build

model, geom, _ = build("/media/parksuho/Extreme SSD/parksuho/open_duck_mini_isaaclab/robot/robot.urdf", "/media/parksuho/Extreme SSD/parksuho/open_duck_mini_isaaclab/robot")
data, gdata = model.createData(), geom.createData()
d = np.load("/media/parksuho/Extreme SSD/parksuho/open_duck_mini_isaaclab/filtered_v28.npz")
names = [str(x) for x in d["leg_names"]]
slot = [model.idx_qs[model.getJointId(n)] for n in names]

def clear(row):
    q = pin.neutral(model)
    for j, s in enumerate(slot):
        q[s] = row[j]
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geom, gdata)
    pin.computeDistances(model, data, geom, gdata, q)
    return min(x.min_distance for x in gdata.distanceResults)

D = np.array([clear(r) for r in d["q"]]) * 1000
print(f"@@필터 통과 자세 {len(D)}개 (원본 v28: 5mm 위반 38.0%, 접촉 13.0%)")
print(f"@@  5 mm 위반 {100*(D < 5).mean():.1f}% · 접촉 {100*(D <= 0).mean():.1f}%")
print(f"@@  최소 {D.min():.2f} mm · 중앙값 {np.median(D):.1f} mm")
if (D < 5).any():
    print(f"@@  위반 자세의 간격: {np.sort(D[D < 5])[:6].round(2)}")
