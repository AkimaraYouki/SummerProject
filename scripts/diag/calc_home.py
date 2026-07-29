"""Computes the reference gait's mean joint pose, to be used as the robot's
new default/home pose.

Why: target = default_pos + action*action_scale(0.25). With default=0 (straight
legs) the reference's crouch (knee ~2.0 rad) needs action~8, i.e. 8 sigma out
of the policy's init distribution -- unreachable. Centering default ON the
reference makes action=0 the reference's neutral pose and +-0.25 rad of action
cover the gait's actual swing amplitude.
"""
import argparse, sys, torch
sys.path.insert(0, "source")
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import PolyReferenceMotion
from open_duck_mini_isaaclab.joint_order import REF_JOINT_NAMES, REF_LEG_JOINT_IDX

# pkl 경로는 인자로 받는다. 하드코딩돼 있어서 새로 만든 레퍼런스(예: 키를 높인
# ref_h175.pkl)에는 쓸 수 없었다 (2026-07-30). 레퍼런스를 바꾸면 이 스크립트로
# READY_JOINT_POS 를 반드시 다시 뽑아야 한다 — 기본 자세와 레퍼런스가 어긋나면
# 액션이 도달 불가능해지고, 그게 v1~v9 를 아홉 번 실패시킨 원인이었다.
_ap = argparse.ArgumentParser()
_ap.add_argument("--pkl", default="source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl")
_args = _ap.parse_args()
print(f"[calc_home] 레퍼런스: {_args.pkl}")

prm = PolyReferenceMotion(_args.pkl, device="cpu")
N = prm.nb_steps_in_period
# average over a representative spread of commands, not just one, so the home
# pose is neutral across the whole command space the policy will see
cmds = [(0.0,0.0,0.0), (0.10,0.0,0.0), (-0.10,0.0,0.0), (0.0,0.10,0.0),
        (0.0,-0.10,0.0), (0.0,0.0,0.5), (0.0,0.0,-0.5), (0.10,0.0,0.5)]
allf = []
for dx,dy,dth in cmds:
    f = prm.get_reference_motion(torch.full((N,),dx), torch.full((N,),dy),
                                 torch.full((N,),dth), torch.arange(N))
    allf.append(f[:, 0:14])
ref = torch.cat(allf, dim=0)
mean_pose = ref.mean(dim=0)
amp = (ref.max(dim=0).values - ref.min(dim=0).values) / 2

print("=" * 74)
print(f"REFERENCE MEAN POSE  ({len(cmds)} commands x {N} frames = {ref.shape[0]} samples)")
print("=" * 74)
print(f"{'joint':<22}{'mean':>9}{'amplitude':>11}   {'needs |action|':>14}")
for i, n in enumerate(REF_JOINT_NAMES):
    leg = "*" if i in REF_LEG_JOINT_IDX else " "
    a = amp[i].item()
    print(f"{leg}{n:<21}{mean_pose[i].item():>9.4f}{a:>11.4f}   {a/0.25:>14.2f}")
print("=" * 74)
print("HOME_JOINT_POS = {")
for i, n in enumerate(REF_JOINT_NAMES):
    print(f'    "{n}": {mean_pose[i].item():.4f},')
print("}")
# residual error if the policy simply held this mean pose
err2 = ((ref - mean_pose) ** 2)[:, REF_LEG_JOINT_IDX].sum(dim=-1)
print("=" * 74)
print(f"sum_err^2 holding mean pose : {err2.mean():.3f} rad^2  (was 12.8 at old HOME)")
print(f"  -> mean per-joint error   : {(err2.mean()/10).sqrt()*57.3:.1f} deg  (was 64.9 deg)")
print(f"max |action| needed for gait: {(amp[REF_LEG_JOINT_IDX]/0.25).max():.2f}  (was 8.1)")
print("=" * 74)
