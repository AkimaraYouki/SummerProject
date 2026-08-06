#!/usr/bin/env python3
"""레퍼런스 보행 pkl 을 meshcat 으로 재생한다 — 바닥을 딛고 걸어 나가는 모습.

    ~/.odm-tools/bin/python scripts/viz_ref_pkl.py --pkl .../ref_g115.pkl
    ~/.odm-tools/bin/python scripts/viz_ref_pkl.py --vx 0.15 --wz 1.0

기존 `viz_reference.py` 는 `odm measure` 가 만든 npz 를 읽는다. 그건 Isaac Sim
을 돌려야 나오고, **학습 중에는 Isaac Sim 을 두 개 못 띄운다.** 이 스크립트는
pkl 에서 직접 읽으므로 학습과 같이 돌릴 수 있다.

파이썬이 두 개 필요하다는 게 성가신 지점이다:
  * PolyReferenceMotion 은 torch 가 필요하다      -> Isaac 파이썬
  * pinocchio / meshcat 은 Isaac 쪽에 없다        -> ~/.odm-tools
그래서 이 파일은 자기 자신을 Isaac 파이썬으로 한 번 불러 npz 로 떨어뜨린 뒤
(--dump), 본체는 .odm-tools 에서 그 npz 를 읽어 그린다.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_PKL = os.path.join(ROOT, "source", "open_duck_mini_isaaclab",
                           "reference_motion", "data", "ref_g115.pkl")
ISAAC = os.path.expanduser("~/Desktop/IsaacLab/isaaclab.sh")
NAMES = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]


def dump(pkl, vx, vy, wz, out):
    """Isaac 파이썬에서 실행된다. 한 주기치 관절각과 접지를 npz 로."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "source"))
    from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import PolyReferenceMotion
    prm = PolyReferenceMotion(pkl, device="cpu")
    n = prm.nb_steps_in_period
    f = prm.get_reference_motion(torch.full((n,), vx), torch.full((n,), vy),
                                 torch.full((n,), wz), torch.arange(n))
    np.savez(out, q=f[:, 0:14].numpy(), contact=f[:, 28:30].numpy(),
             period=np.array([n]), cmd=np.array([vx, vy, wz]))
    print(f"[dump] {out}  주기 {n} 스텝")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--vx", type=float, default=0.15, help="전진 m/s")
    ap.add_argument("--vy", type=float, default=0.0, help="횡보 m/s")
    ap.add_argument("--wz", type=float, default=0.0, help="회전 rad/s")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    ap.add_argument("--dump", default="", help="내부용 — Isaac 파이썬이 쓰는 경로")
    args = ap.parse_args()

    if args.dump:
        dump(args.pkl, args.vx, args.vy, args.wz, args.dump)
        return

    npz = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "ref_viz.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    print("[1/2] 레퍼런스를 푸는 중 (Isaac 파이썬)…")
    r = subprocess.run([ISAAC, "-p", os.path.abspath(__file__),
                        "--pkl", args.pkl, "--vx", str(args.vx), "--vy", str(args.vy),
                        "--wz", str(args.wz), "--dump", npz],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ""})
    if not os.path.exists(npz):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("레퍼런스 덤프 실패")

    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat.geometry as g
    import meshcat.transformations as tf

    d = np.load(npz)
    Q, period = d["q"], int(d["period"][0])
    vx, vy, wz = d["cmd"]

    model, coll, vis = pin.buildModelsFromUrdf(
        os.path.join(ROOT, "robot", "robot.urdf"), os.path.join(ROOT, "robot"),
        pin.JointModelFreeFlyer())
    data = model.createData()
    viz = MeshcatVisualizer(model, coll, vis)
    viz.initViewer(open=False)
    viz.loadViewerModel()

    slot = [model.idx_qs[model.getJointId(n)] for n in NAMES]
    FID = [model.getFrameId("left_foot"), model.getFrameId("right_foot")]

    # 바닥 + 격자. 격자가 없으면 전진하는지 눈으로 확인이 안 된다.
    viz.viewer["ground"].set_object(g.Box([4.0, 4.0, 0.002]),
                                    g.MeshLambertMaterial(color=0xD8D6CE))
    viz.viewer["ground"].set_transform(tf.translation_matrix([0, 0, -0.0015]))
    for i in range(-20, 21):
        for ax in (0, 1):
            size = [4.0, 0.004, 0.001] if ax == 0 else [0.004, 4.0, 0.001]
            pos = [0, i * 0.1, 0] if ax == 0 else [i * 0.1, 0, 0]
            node = viz.viewer[f"grid/{ax}_{i}"]
            node.set_object(g.Box(size), g.MeshLambertMaterial(color=0xB4B2A9))
            node.set_transform(tf.translation_matrix(pos))

    # 주기별 관절 벡터와 발 최저점. 발 프레임이 아니라 **밑창**을 바닥에 맞춘다.
    frames, zmin = [], 1e9
    for k in range(period):
        q = pin.neutral(model)
        for j, s in enumerate(slot):
            q[s] = Q[k, j]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        zmin = min(zmin, min(data.oMf[f].translation[2] for f in FID))
        frames.append(q)
    zmin -= 0.0026   # 발 프레임 -> 밑창

    print(f"[meshcat] {viz.viewer.url()}")
    print(f"[레퍼런스] {os.path.basename(args.pkl)}  "
          f"cmd=({vx:+.2f}, {vy:+.2f}, {wz:+.2f})  주기 {period}스텝 "
          f"{period*args.dt:.2f}s  재생 {args.speed}x")
    print("Ctrl+C 로 종료")

    # 베이스는 명령 속도로 적분한다 (viz_reference.py 와 같은 방식).
    i, yaw, xy = 0, 0.0, np.zeros(2)
    try:
        while True:
            k = i % period
            c, s_ = np.cos(yaw), np.sin(yaw)
            xy = xy + args.dt * np.array([c * vx - s_ * vy, s_ * vx + c * vy])
            yaw += args.dt * wz
            q = frames[k].copy()
            q[0:3] = np.array([xy[0], xy[1], -zmin])
            q[3:7] = pin.Quaternion(pin.utils.rotate("z", yaw)).coeffs()
            viz.display(q)
            i += 1
            time.sleep(args.dt / max(args.speed, 1e-6))
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
