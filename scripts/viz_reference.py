"""ref_h175 레퍼런스 보행을 meshcat 으로 띄운다 -- 실제로 바닥을 딛고 걷는 모습.

베이스를 고정하면 제자리걸음으로 보인다. 프리플라이어를 붙이고 **디딤발을 월드에
고정**하는 방식으로 베이스를 역산하면, 명령 속도를 따로 적분하지 않아도 발이
미끄러지지 않는다. 전진·후진·횡보·회전이 전부 같은 코드로 맞는다.

궤적은 `odm measure` 가 저장한 npz 를 쓴다: `*__qr` 이 레퍼런스 관절각,
`*__feet_ref` 가 레퍼런스 접지다 -- 정책이 실제로 모방하는 바로 그 값이다.

    mcv/bin/python viz_ref.py forward     # backward left right turn stop
"""
import sys
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

ROOT = "/media/parksuho/Extreme SSD/parksuho/open_duck_mini_isaaclab"
NPZ = "/home/parksuho/odm_out/gait_v28.npz"
PERIOD, DT = 27, 0.02
cond = sys.argv[1] if len(sys.argv) > 1 else "forward"

model, coll, vis = pin.buildModelsFromUrdf(
    f"{ROOT}/robot/robot.urdf", f"{ROOT}/robot", pin.JointModelFreeFlyer())
data = model.createData()
viz = MeshcatVisualizer(model, coll, vis)
viz.initViewer(open=False)
viz.loadViewerModel()

# 바닥 + 격자. 격자가 없으면 전진하는지 눈으로 확인이 안 된다.
viz.viewer["ground"].set_object(
    g.Box([4.0, 4.0, 0.002]), g.MeshLambertMaterial(color=0xD8D6CE))
viz.viewer["ground"].set_transform(tf.translation_matrix([0, 0, -0.0015]))
for i in range(-20, 21):
    for ax in (0, 1):
        size = [4.0, 0.004, 0.001] if ax == 0 else [0.004, 4.0, 0.001]
        pos = [0, i * 0.1, 0] if ax == 0 else [i * 0.1, 0, 0]
        node = viz.viewer[f"grid/{ax}_{i}"]
        node.set_object(g.Box(size), g.MeshLambertMaterial(color=0xB4B2A9))
        node.set_transform(tf.translation_matrix(pos))

d = np.load(NPZ)
names = [str(x) for x in d["leg_names"]]
slot = [model.idx_qs[model.getJointId(n)] for n in names]
qr = d[f"{cond}__qr"][100:100 + PERIOD, 0]
fr = d[f"{cond}__feet_ref"][100:100 + PERIOD, 0] > 0.5   # [T, 2] 좌/우 접지
cmd = d[f"{cond}__cmd"]
FID = [model.getFrameId("left_foot"), model.getFrameId("right_foot")]


def foot_in_base(k):
    """베이스를 원점에 둔 상태의 관절 벡터와 양발 placement."""
    q = pin.neutral(model)
    for j, s in enumerate(slot):
        q[s] = qr[k, j]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return q, [data.oMf[f].copy() for f in FID]


base = [foot_in_base(k) for k in range(PERIOD)]

# 베이스 이동은 **명령 속도**로 만든다. 정책이 실제로 내는 것도 명령 쪽이다
# (전진 명령 0.15 에 대해 실측 0.143).
#
# 레퍼런스가 들고 있는 v_ref 는 쓰지 않는다 -- 명령과 안 맞는다:
#   전진 cmd (+0.15, 0, 0)  ->  v_ref (+0.077, -0.043), w_ref -0.074
#   좌   cmd (0, +0.20, 0)  ->  v_ref (-0.001, +0.112), w_ref -0.074
# 직진 명령인데 yaw 가 -0.074 rad/s 다. 그 값은 레퍼런스 명령 격자에서 0 에 가장
# 가까운 지점과 같다 (격자에 0 이 없다, docs/training_log.md v31 항목).
# 시각화 문제가 아니라 레퍼런스 자체의 문제로 보이며, 따로 검증할 일이다.
vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])

poses, qs, yaw, xy = [], [], 0.0, np.zeros(2)
for k in range(PERIOD):
    q, _ = base[k]
    c, s_ = np.cos(yaw), np.sin(yaw)
    xy = xy + DT * np.array([c * vx - s_ * vy, s_ * vx + c * vy])
    yaw += DT * wz
    poses.append(pin.SE3(pin.utils.rotate("z", yaw),
                         np.array([xy[0], xy[1], 0.0])))
    qs.append(q)

# 발 최저점을 바닥(z=0)에 맞춘다. 베이스 회전이 yaw 뿐이라 상수 하나면 된다.
zmin = min(b.translation[2] for _, bb in base for b in bb)
lap = poses[0].inverse() * poses[-1]      # 한 주기당 누적 변위
stride = float(np.linalg.norm(lap.translation[:2]))

print(f"[meshcat] {viz.viewer.url()}", flush=True)
print(f"[ref] ref_h175 · {cond} "
      f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})", flush=True)
print(f"[gait] 주기 {PERIOD}스텝 {PERIOD * DT:.2f}s · 한 주기 {stride * 1000:.0f} mm "
      f"= {stride / (PERIOD * DT):.3f} m/s (명령) · 발 최저점 {zmin * 1000:+.1f} mm",
      flush=True)

acc, i = pin.SE3.Identity(), 0
while True:
    k = i % PERIOD
    if k == 0 and i:
        acc = acc * lap                   # 주기를 넘어가도 계속 걸어 나간다
    T = acc * poses[k]
    q = qs[k].copy()
    q[0:3] = T.translation - np.array([0.0, 0.0, zmin])
    q[3:7] = pin.Quaternion(T.rotation).coeffs()
    viz.display(q)
    i += 1
    time.sleep(DT)
