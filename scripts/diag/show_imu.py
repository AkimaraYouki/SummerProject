#!/usr/bin/env python3
"""IMU 가 로봇 어디에 붙어 있고 축이 어디를 향하는지 meshcat 으로 띄운다.

    ~/.odm-tools/bin/python scripts/diag/show_imu.py            # 중립 자세
    ~/.odm-tools/bin/python scripts/diag/show_imu.py --ready    # READY 자세

표시하는 것:
  * 빨강 구  = `imu_frame` (URDF/시뮬이 IMU 로 쓰는 점)
  * 노랑 구  = trunk_assembly 링크 **원점** — CAD 가 정한 임의의 점이라
               눈에 보이는 특징과 안 맞는다. 우리가 "몸통 높이" 라고 부르던 기준.
  * 축 삼각대 = 빨강 +x(앞) / 초록 +y(좌) / 파랑 +z(위)
               실기 실측 축맵이 항등이라 IMU 축 = 몸통 축이다 (imu_map.py).
  * 회색 바닥 = 발 밑창이 닿는 지면
"""

import argparse
import os
import sys
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "source"))

READY = {
    "left_hip_yaw": 0.0003, "left_hip_roll": 0.0213, "left_hip_pitch": 0.9910,
    "left_knee": -1.7852, "left_ankle": 0.8647,
    "right_hip_yaw": -0.0005, "right_hip_roll": -0.0092, "right_hip_pitch": 1.0114,
    "right_knee": 1.8163, "right_ankle": -0.8754,
}


def axes(node, origin, length=0.06, radius=0.0018):
    """+x 빨강 / +y 초록 / +z 파랑 삼각대. meshcat 실린더는 기본이 +y 방향이라 돌려 준다."""
    for name, color, rot in (
        ("x", 0xE05A4A, tf.rotation_matrix(-np.pi / 2, [0, 0, 1])),
        ("y", 0x5FB37A, np.eye(4)),
        ("z", 0x4A7FE0, tf.rotation_matrix(np.pi / 2, [1, 0, 0])),
    ):
        n = node[f"axis_{name}"]
        n.set_object(g.Cylinder(length, radius), g.MeshLambertMaterial(color=color))
        T = tf.translation_matrix(origin) @ rot @ tf.translation_matrix([0, length / 2, 0])
        n.set_transform(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready", action="store_true", help="READY 자세로 (기본은 중립)")
    args = ap.parse_args()

    urdf = os.path.join(ROOT, "robot", "robot.urdf")
    model, coll, vis = pin.buildModelsFromUrdf(urdf, os.path.join(ROOT, "robot"))
    data = model.createData()

    q = pin.neutral(model)
    if args.ready:
        for n, v in READY.items():
            q[model.idx_qs[model.getJointId(n)]] = v

    viz = MeshcatVisualizer(model, coll, vis)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    viz.display(q)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    imu = data.oMf[model.getFrameId("imu")].translation.copy()
    foot = min(data.oMf[model.getFrameId(s)].translation[2] for s in ("left_foot", "right_foot"))

    # 바닥. 발 프레임은 밑창보다 2.6 mm 위라 그만큼 더 내린다.
    ground = foot - 0.0026
    viz.viewer["floor"].set_object(g.Box([0.6, 0.6, 0.002]),
                                   g.MeshLambertMaterial(color=0xD8D6CE, opacity=0.85))
    viz.viewer["floor"].set_transform(tf.translation_matrix([0, 0, ground - 0.001]))

    viz.viewer["imu/dot"].set_object(g.Sphere(0.007), g.MeshLambertMaterial(color=0xE05A4A))
    viz.viewer["imu/dot"].set_transform(tf.translation_matrix(imu))
    axes(viz.viewer["imu"], imu)

    viz.viewer["trunk_origin"].set_object(g.Sphere(0.005),
                                          g.MeshLambertMaterial(color=0xE0C24A))
    viz.viewer["trunk_origin"].set_transform(tf.translation_matrix([0, 0, 0]))

    print(f"[meshcat] {viz.viewer.url()}")
    print(f"[자세]    {'READY' if args.ready else '중립(관절 전부 0)'}")
    print(f"[IMU]     trunk_assembly 원점 기준 "
          f"({imu[0]*1000:+.1f}, {imu[1]*1000:+.1f}, {imu[2]*1000:+.1f}) mm")
    print(f"          지면 위 {(imu[2]-ground)*1000:.1f} mm")
    print("[축]      빨강 +x 앞 · 초록 +y 좌 · 파랑 +z 위   (실측 축맵 항등)")
    print("[구]      빨강 = IMU · 노랑 = trunk_assembly 원점")
    print("Ctrl+C 로 종료")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
