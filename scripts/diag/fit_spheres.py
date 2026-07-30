"""몸통·정강이를 구 집합으로 덮고, 정확 메시 대비 오차를 검증한다.

단일 캡슐은 실패했다 -- 몸통 반지름이 112.8 mm 로 나와, 여유가 5~9 mm 인 문제에서
모든 자세를 충돌로 판정한다. 구를 여러 개 쓰면 덮으면서도 조일 수 있다.

**보수성 규칙**: 각 구는 자기 클러스터의 정점을 모두 품는다(반지름 = 최대 거리).
그러면 구집합 거리 <= 실제 메시 거리가 보장되고, 필터는 항상 안전 쪽으로 틀린다.
개수를 늘릴수록 실제 거리에 붙는다.
"""
import sys, json, numpy as np, pinocchio as pin
from scipy.cluster.vq import kmeans2
from leg_trunk_clearance import build

TRUNK = "trunk_assembly"
SHINS = ["knee_and_ankle_assembly", "knee_and_ankle_assembly_2",
         "knee_and_ankle_assembly_3", "knee_and_ankle_assembly_4"]
K_TRUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 64
K_SHIN = int(sys.argv[2]) if len(sys.argv) > 2 else 24

model, geom, _ = build("robot/robot.urdf", "robot")
data, gdata = model.createData(), geom.createData()
gn = [model.frames[g.parentFrame].name for g in geom.geometryObjects]
FR = {n: model.getFrameId(n) for n in [TRUNK] + SHINS}


def verts(link):
    out = []
    for i, g in enumerate(geom.geometryObjects):
        if gn[i] != link:
            continue
        o = g.geometry
        try:
            V = np.array([o.vertices(k) for k in range(o.num_vertices)])
        except Exception:
            continue
        out.append(V @ g.placement.rotation.T + g.placement.translation)
    return np.vstack(out)


def cover(P, K):
    """구 K 개로 정점을 전부 덮는다. 반지름은 클러스터 내 최대 거리."""
    idx = np.random.default_rng(0).choice(len(P), min(len(P), 20000), replace=False)
    c, lab = kmeans2(P[idx], K, minit="++", seed=0)
    full = ((P[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
    r = np.zeros(K)
    for k in range(K):
        m = full == k
        if m.any():
            r[k] = np.linalg.norm(P[m] - c[k], axis=1).max()
    keep = r > 0
    return c[keep], r[keep]


spheres = {}
for link, K in [(TRUNK, K_TRUNK)] + [(s, K_SHIN) for s in SHINS]:
    P = verts(link)
    c, r = cover(P, K)
    spheres[link] = (c, r)
    print(f"@@{link:28s} 구 {len(c):3d}개 · 반지름 평균 {r.mean()*1000:5.1f} / 최대 {r.max()*1000:5.1f} mm")

# 검증: 실제 자세에서 구집합 거리 vs 정확 메시 거리
d = np.load("gait_v32.npz")
names = [str(x) for x in d["leg_names"]]
slot = [model.idx_qs[model.getJointId(n)] for n in names]
gap, viol, n = [], 0, 0
for cond in ["stop", "forward", "backward", "left", "right", "turn"]:
    traj = d[f"{cond}__q"][100::20, 0]
    for t in range(traj.shape[0]):
        q = pin.neutral(model)
        for j, s in enumerate(slot):
            q[s] = traj[t, j]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.updateGeometryPlacements(model, data, geom, gdata)
        pin.computeDistances(model, data, geom, gdata, q)
        mesh = min(x.min_distance for x in gdata.distanceResults)

        w = {k: data.oMf[v] for k, v in FR.items()}
        best = 1e9
        ct = w[TRUNK].rotation @ spheres[TRUNK][0].T + w[TRUNK].translation[:, None]
        for sh in SHINS:
            cs = w[sh].rotation @ spheres[sh][0].T + w[sh].translation[:, None]
            D = np.linalg.norm(ct[:, :, None] - cs[:, None, :], axis=0)
            best = min(best, (D - spheres[TRUNK][1][:, None] - spheres[sh][1][None, :]).min())
        gap.append(mesh - best); n += 1
        if best > mesh + 1e-9:
            viol += 1

gap = np.array(gap) * 1000
print(f"@@\n@@자세 {n}개 · 보수성 위반(구집합이 메시보다 멀다고 함) {viol}개  <- 0 이어야 안전")
print(f"@@과보수 gap: 평균 {gap.mean():.1f} mm · 중앙값 {np.median(gap):.1f} mm · 최대 {gap.max():.1f} mm")
json.dump({k: {"c": v[0].tolist(), "r": v[1].tolist()} for k, v in spheres.items()},
          open("spheres.json", "w"))
print("@@[ok] spheres.json")
