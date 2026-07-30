"""다리와 몸통이 실제로 얼마나 가까운지 정확 메시로 잰다.

**Isaac Sim 안에서는 못 돌린다.** pinocchio 임포트가 실패하고(libhpp-fcl 충돌),
PhysX 의 볼록 근사는 이 로봇에서 쓸 수 없다 -- 관절 하우징이 부모 부품 *안에*
끼워지는 구조라 어떤 볼록 껍질도 서로 겹친다 (joystick_env_cfg.py 의 자기충돌
주석). 그래서 **궤적은 Isaac 에서 npz 로 뽑고, 거리 계산은 여기서 따로 한다.**

    # 이 PC (pinocchio 없음)
    rsync -az scripts/diag/leg_trunk_clearance.py robot/ <랩PC>:<원격>/
    scp ~/odm_out/gait_v28.npz <랩PC>:<원격>/
    # 랩PC (placo 가 있으니 pinocchio 2.7 + hppfcl 이 있다)
    python3 leg_trunk_clearance.py --npz gait_v28.npz --urdf robot/robot.urdf

다리<->몸통 거리는 다리 관절각만으로 정해진다 (몸통이 기준 링크이므로 로봇의
월드 자세는 무관하다). 그래서 `gait_compare.py` 가 이미 저장한 다리 10관절
궤적이면 충분하고, 시뮬레이션을 다시 돌릴 필요가 없다.

레퍼런스 궤적(`*__qr`)도 같이 계산한다. 기존 실측이 "레퍼런스는 접촉 0%,
최소 간격 7.1 mm" 였으므로 그 값이 재현되면 이 파이프라인을 믿을 수 있다.
"""

import argparse
import os

import numpy as np


# 몸통 쪽 / 다리 쪽 링크. 이름은 robot/robot.urdf 에서 확인한 것 그대로다.
TRUNK_LINKS = {"trunk", "trunk_assembly", "imu"}
LEG_LINKS = {
    "hip_roll_assembly", "left_roll_to_pitch_assembly",
    "knee_and_ankle_assembly", "knee_and_ankle_assembly_2",
    "foot_assembly", "left_foot",
    "hip_roll_assembly_2", "right_roll_to_pitch_assembly",
    "knee_and_ankle_assembly_3", "knee_and_ankle_assembly_4",
    "foot_assembly_2", "right_foot",
}
CONDS = ["stop", "forward", "backward", "left", "right", "turn"]
KO = {"stop": "정지", "forward": "앞", "backward": "뒤",
      "left": "좌", "right": "우", "turn": "회전"}
WARM = 100


def build(urdf, mesh_dir):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(urdf)
    geom = pin.buildGeomFromUrdf(model, urdf, pin.GeometryType.COLLISION, mesh_dir)

    # 인접(부모-자식) 링크는 설계상 닿아 있으므로 뺀다. 그 판정을 링크가 아니라
    # **관절 사슬 거리**로 한다 -- 이 URDF 는 하우징이 부모 안에 끼워지는 구조라
    # 두 관절 떨어진 쌍도 볼록 근사로는 겹치는데, 정확 메시로는 10 mm 넘게 뜬다.
    # 관절 사슬 거리로 인접 여부를 정한다. 몸통은 베이스(관절 0)라 "부모-자식"
    # 판정만으로는 몸통<->고관절 하우징 같은 직결 쌍이 걸러지지 않는다 --
    # 실제로 처음에 그래서 레퍼런스마저 접촉 100% 로 나왔다.
    def chain(jid):
        out = []
        while jid > 0:
            out.append(jid)
            jid = model.parents[jid]
        return set(out)

    MIN_HOPS = 3   # 몸통에서 3관절 이상 떨어진 다리 링크만 본다

    pairs = []
    for i, gi in enumerate(geom.geometryObjects):
        for j, gj in enumerate(geom.geometryObjects):
            if j <= i:
                continue
            ni = model.frames[gi.parentFrame].name
            nj = model.frames[gj.parentFrame].name
            if not (({ni} & TRUNK_LINKS and {nj} & LEG_LINKS) or
                    ({nj} & TRUNK_LINKS and {ni} & LEG_LINKS)):
                continue
            ci, cj = chain(gi.parentJoint), chain(gj.parentJoint)
            if len(ci ^ cj) < MIN_HOPS:
                continue
            pairs.append((i, j))
            geom.addCollisionPair(pin.CollisionPair(i, j))
    return model, geom, pairs


def min_clearance(model, geom, data, gdata, q):
    import pinocchio as pin
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geom, gdata)
    pin.computeDistances(model, data, geom, gdata, q)
    return min(r.min_distance for r in gdata.distanceResults)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--mesh-dir", default="")
    ap.add_argument("--stride", type=int, default=25, help="몇 스텝마다 계산할지")
    # 메시 전수 거리계산이 비싸다(형상 46개). 환경을 늘려도 같은 정책의 같은
    # 명령이라 분포가 거의 겹치므로, 기본은 1개만 본다.
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--dump", default="", help="포즈별 (간격, 고관절 이탈) 을 npz 로")
    # 실기 안전 기준 (mm). 사용자 지정 5 mm -- 부품 공차를 감안하면 "닿지만
    # 않으면 된다"가 아니라 이만큼 떠 있어야 한다. 접촉률(<=0)만 보면 문제를
    # 3분의 1 크기로 과소평가한다: v28 은 접촉 13.0% 인데 5 mm 위반은 38.0% 다.
    ap.add_argument("--safe-mm", type=float, default=5.0)
    args = ap.parse_args()

    import pinocchio as pin
    mesh_dir = args.mesh_dir or os.path.dirname(os.path.abspath(args.urdf))
    model, geom, pairs = build(args.urdf, mesh_dir)
    data, gdata = model.createData(), geom.createData()
    print(f"[info] 링크 {model.nframes}개 · 다리<->몸통 형상쌍 {len(pairs)}개 · q 차원 {model.nq}")

    d = np.load(args.npz)
    names = [str(x) for x in d["leg_names"]]
    # npz 의 다리 10관절을 pinocchio 의 q 벡터 자리로 옮긴다. 나머지(머리)는 0 --
    # 머리 관절은 다리<->몸통 거리에 영향을 주지 않는다.
    slot = []
    for n in names:
        jid = model.getJointId(n)
        assert jid < model.njoints, f"URDF 에 없는 관절: {n}"
        slot.append(model.idx_qs[jid])

    SAFE = args.safe_mm / 1000.0
    print(f"\n{'명령':6s} {'':3s}{'정책 위반율':>11s} {'정책 접촉':>10s} {'정책 최소':>10s} "
          f"{'레퍼런스 위반':>13s} {'레퍼런스 최소':>13s}   (기준 {args.safe_mm:g} mm)")
    agg = {"q": [], "qr": []}
    # --dump: 어느 관절이 어느 **방향**으로 갈 때 간격이 줄어드는지 보려면
    # 포즈별로 간격과 부호 있는 이탈을 같이 남겨야 한다. v31 이 실패한 것은
    # 이 정보 없이 대칭으로 잘랐기 때문이다.
    hip_slots = [k for k, n in enumerate(names) if "hip_roll" in n or "hip_yaw" in n]
    dump = {"clear": [], "dev": [], "cmd": []}
    for c in CONDS:
        if f"{c}__q" not in d.files:
            continue
        row = {}
        for tag in ("q", "qr"):
            traj = d[f"{c}__{tag}"][WARM::args.stride, :args.envs]   # [T, N, 10]
            dists = []
            for t in range(traj.shape[0]):
                for e in range(traj.shape[1]):
                    q = pin.neutral(model)
                    for k, s in enumerate(slot):
                        q[s] = traj[t, e, k]
                    cl = min_clearance(model, geom, data, gdata, q)
                    dists.append(cl)
                    if args.dump and tag == "q":
                        dump["clear"].append(cl)
                        dump["dev"].append(traj[t, e, hip_slots]
                                           - d[f"{c}__qr"][WARM::args.stride, :args.envs][t, e, hip_slots])
                        dump["cmd"].append(CONDS.index(c))
            dists = np.array(dists)
            agg[tag].append(dists)
            row[tag] = (float((dists < SAFE).mean()) * 100,
                        float((dists <= 0).mean()) * 100, float(dists.min()) * 1000)
        import sys; sys.stdout.flush()
        print(f"{KO[c]:6s} {'':3s}{row['q'][0]:9.1f} % {row['q'][1]:8.1f} % {row['q'][2]:7.1f} mm "
              f"{row['qr'][0]:11.1f} % {row['qr'][2]:10.1f} mm")

    if args.dump:
        np.savez_compressed(args.dump, hip_names=np.array([names[k] for k in hip_slots]),
                            **{k: np.array(v) for k, v in dump.items()})
        print(f"[ok] dump -> {args.dump}")

    for tag, label in (("q", "정책"), ("qr", "레퍼런스")):
        a = np.concatenate(agg[tag])
        print(f"\n[{label}] {args.safe_mm:g} mm 위반 {float((a < SAFE).mean())*100:.1f} % · "
              f"접촉 {float((a <= 0).mean())*100:.1f} % · "
              f"최소 {a.min()*1000:.1f} mm · 중앙값 {np.median(a)*1000:.1f} mm")


if __name__ == "__main__":
    main()
