---
name: measure-reward-discrimination
description: Audit what a reward pays for DOING NOTHING vs doing the task — an exp(-k·err²) term with too-small k silently stops discriminating
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-28T10:11:30.367Z
---

For any shaped reward, measure **what the degenerate behavior scores** against what the target
behavior scores. Not the weights — the realized per-term values.

The OpenDuck failure ([[project_openduck_autonomous_training]]): `lin_vel_xy`, the one term whose
job was to price walking, used `exp(-8·err²)`. Missing the reference speed (0.265 m/s) *entirely*
gives err² = 0.070, so it still paid `exp(-0.56) = 0.57` out of 1.0. Combined with `lin_vel_z`
(0.954/1.0 free) and `ang_vel_xy` (reference spread 0.0000 — always full marks by construction),
**~92% of the imitation reward was collectable while standing still**. The policy stood and
trembled, which was the correct answer to the reward as written.

**Why:** under `exp(-k·err²)`, k is a *sensitivity*, not a weight. Too small and the term is
flat across the entire range that matters (no gradient toward the task, and it doubles as free
income for the degenerate behavior); too large and it saturates to ~0 everywhere (flat gradient
again). Reading the weight tells you nothing about which regime you are in.

**How to apply:**
- Derive k from the reference signal's own spread: `k ≈ 1/spread` lands typical error near
  `exp(-1) ≈ 0.37`, the responsive middle of the curve. A term whose reference spread is ~0 has
  no discriminating power at any k — drop it rather than reweight it.
- Run a **standing/no-op measurement** (`scripts/reward_at_ready.py` pattern: action=0, sum every
  term) and a **trained-policy per-term breakdown** (`scripts/imit_internals2.py`) before and
  after each change. Track the standing-to-ideal ratio as the actual metric: 0.734 -> 0.583 ->
  0.444 per step across v10/v11/v12.
- Measure the **behavior** too, not just the reward: actual base speed vs reference speed, and
  foot-contact toggles. v11's reward terms looked mediocre-but-alive while the robot moved at
  0.064 m/s against a 0.265 m/s reference — the behavioral number is what made the diagnosis
  unambiguous.
- Do not trust a plausible-sounding culprit without measuring it. I twice named the wrong one
  (unbounded `joint_vel`, which turned out to be 2% of the total; and missing phase observation,
  which was in fact present in the obs vector).

Related: [[feedback_check_action_reachability_first]], [[feedback_verify_training_visually]].
