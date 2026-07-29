"""Compares the reference motion's commanded joint angles against the robot's
actual URDF joint limits and home pose.

Motivated by imit_internals.py measuring ~79 deg mean per-joint error between
policy and reference after a full training run -- far too large for "not yet
converged", and the signature of a mapping/convention mismatch. If the
reference asks for angles the robot physically cannot reach, the joint_pos
imitation term is permanently dead no matter how long you train.
"""
import sys, xml.etree.ElementTree as ET
import torch
sys.path.insert(0, "source")
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import PolyReferenceMotion
from open_duck_mini_isaaclab.joint_order import REF_JOINT_NAMES, REF_LEG_JOINT_IDX, HOME_JOINT_POS

prm = PolyReferenceMotion("source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl", device="cpu")
lim = {}
for j in ET.parse("robot/robot.urdf").getroot().findall("joint"):
    l = j.find("limit")
    if l is not None:
        lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))

# sample the reference across a full gait cycle for a typical forward command
N = prm.nb_steps_in_period
dx = torch.full((N,), 0.10); dy = torch.zeros(N); dth = torch.zeros(N)
idx = torch.arange(N)
frames = prm.get_reference_motion(dx, dy, dth, idx)   # [N, 36]
ref_jp = frames[:, 0:14]

print("=" * 78)
print(f"REFERENCE vs ROBOT LIMITS  (cmd dx=0.10, {N} frames = one gait cycle)")
print("=" * 78)
print(f"{'joint':<22}{'ref min':>9}{'ref max':>9} | {'urdf lo':>9}{'urdf hi':>9} | {'home':>7}  violates?")
bad = 0
for i, name in enumerate(REF_JOINT_NAMES):
    rmin, rmax = ref_jp[:, i].min().item(), ref_jp[:, i].max().item()
    lo, hi = lim.get(name, (float("nan"), float("nan")))
    home = HOME_JOINT_POS.get(name, float("nan"))
    v = ""
    if lo == lo:  # not nan
        if rmin < lo - 1e-3 or rmax > hi + 1e-3:
            v = "  <<< OUT OF LIMIT"
            bad += 1
    leg = "*" if i in REF_LEG_JOINT_IDX else " "
    print(f"{leg}{name:<21}{rmin:>9.3f}{rmax:>9.3f} | {lo:>9.3f}{hi:>9.3f} | {home:>7.3f}{v}")
print("=" * 78)
print(f"joints outside URDF limits: {bad}   (* = used by imitation reward)")
# how far is the robot's HOME pose from the reference, in the same metric the reward uses?
home_vec = torch.tensor([HOME_JOINT_POS.get(n, 0.0) for n in REF_JOINT_NAMES])
err2 = ((ref_jp - home_vec) ** 2)[:, REF_LEG_JOINT_IDX].sum(dim=-1)
print(f"sum_err^2 of HOME pose vs reference (10 leg joints): mean {err2.mean():.3f}, max {err2.max():.3f} rad^2")
print(f"  -> mean per-joint error at HOME: {(err2.mean()/10).sqrt()*57.3:.1f} deg")
print("=" * 78)
