"""Measures the reference motion's actual signal distributions, so the reward's
exp() sensitivities can be set from data instead of inherited guesses.

Each tracking term is exp(-k * err^2). k only produces a useful gradient when
k * (typical err^2) lands near ~1: too large and it saturates to 0 (flat
gradient, the failure we already measured on joint_pos), too small and it sits
near 1 for everything (no discrimination). "Typical err" for an untrained
policy is roughly the reference signal's own spread, so that's what this
measures -- per term, over the whole command space the policy will see.
"""
import sys, torch
sys.path.insert(0, "source")
from open_duck_mini_isaaclab.reference_motion.poly_reference_motion import PolyReferenceMotion
from open_duck_mini_isaaclab.joint_order import REF_LEG_JOINT_IDX

prm = PolyReferenceMotion("source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl", device="cpu")
N = prm.nb_steps_in_period
cmds = [(x, y, t) for x in (-0.15, -0.07, 0.0, 0.07, 0.15)
                  for y in (-0.2, 0.0, 0.2) for t in (-1.0, 0.0, 1.0)]
fr = []
for dx, dy, dt in cmds:
    fr.append(prm.get_reference_motion(torch.full((N,), dx), torch.full((N,), dy),
                                       torch.full((N,), dt), torch.arange(N)))
f = torch.cat(fr, 0)
jp, jv, ct, lv, av = f[:, 0:14], f[:, 14:28], f[:, 28:30], f[:, 30:33], f[:, 33:36]
jp, jv = jp[:, REF_LEG_JOINT_IDX], jv[:, REF_LEG_JOINT_IDX]

def rep(name, sig, k_now, note=""):
    # spread of the signal itself ~ the error scale an untrained policy sees
    sd = sig.std(dim=0)
    e2 = (sd ** 2).sum().item()
    print(f"{name:<18} spread(sum sd^2)={e2:9.4f} | k_now={k_now:<7} k*e2={k_now*e2:9.3f} "
          f"-> exp={torch.exp(torch.tensor(-k_now*e2)).item():.4f}  k_for_exp0.37={1/e2 if e2>0 else float('nan'):8.3f} {note}")

print("=" * 104)
print(f"REFERENCE SIGNAL SPREAD  ({len(cmds)} commands x {N} frames = {f.shape[0]} samples)")
print("=" * 104)
rep("joint_pos (10)", jp, 0.25, "<- current bounded setting")
rep("joint_vel (10)", jv, 0.0, "(unbounded -err^2 * 1e-3, no exp)")
rep("lin_vel_xy", lv[:, :2], 8.0)
rep("lin_vel_z", lv[:, 2:3], 8.0)
rep("ang_vel_xy", av[:, :2], 2.0)
rep("ang_vel_z", av[:, 2:3], 2.0)
print("-" * 104)
print(f"joint_vel abs: mean {jv.abs().mean():.3f}  max {jv.abs().max():.3f} rad/s"
      f"   -> -err^2*1e-3 at max: {-(jv.abs().max()**2 * 1e-3 * 10).item():.3f}")
print(f"lin_vel  |xy| mean {lv[:,:2].norm(dim=-1).mean():.4f} max {lv[:,:2].norm(dim=-1).max():.4f} m/s")
print(f"ang_vel  |z|  mean {av[:,2].abs().mean():.4f} max {av[:,2].abs().max():.4f} rad/s")
print(f"contact  duty: left {ct[:,0].mean():.3f}  right {ct[:,1].mean():.3f} (fraction of time in contact)")
print("=" * 104)
