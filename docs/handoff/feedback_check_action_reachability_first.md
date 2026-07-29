---
name: check-action-reachability-first
description: "Before tuning RL reward weights, verify the target pose is reachable within the policy's action distribution — 9 OpenDuck runs were tuned on a system that physically couldn't do the task"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-28T10:10:06.945Z
---

When an imitation/tracking RL policy fails, check **whether the reference target is reachable
within the action distribution** before touching any reward weight.

In the OpenDuck project ([[project_openduck_autonomous_training]]), joint targets are
`target = default_joint_pos + action * action_scale`. With `default_joint_pos` = all zeros and a
reference requiring knee ~= 2.03 rad at `action_scale=0.25`, the needed action was **8.12 — over
8 sigma** outside the policy's `init_noise_std=1.0` Gaussian. Nine consecutive runs (v1-v9) were
spent tuning `alive_scale`, `w_joint_pos`, RSI on/off, and `imitation_scale` on a robot that could
not physically produce the target pose under any action.

**Why:** reward shaping only reallocates credit among behaviors the policy can actually execute.
If the target is outside the reachable set, every weight produces the same failure, and the
identical-failure pattern reads as "reward is wrong" when it is really "the task is impossible."
Worse, it silently invalidates *ablations* run during that period — the v7-vs-v8 RSI comparison
concluded "RSI doesn't help" while RSI was initializing into poses no action could hold, so it
measured nothing and had to be re-run later.

**How to apply:**
- Compute `(target - default) / action_scale` in units of `init_noise_std`. Beyond ~2-3 sigma,
  fix the offset (move `default_joint_pos`, raise `action_scale`) before tuning anything else.
- When a fix changes the reachable set, **re-open ablations that were run before it** — their
  conclusions are void, not merely stale.
- Watch for changes that are silent no-ops at the old defaults: the same fix exposed a
  *multiplicative* joint randomization that did nothing while defaults were zeros and would have
  pushed a 2.03 rad knee past its ±2.094 limit afterward.

Related: [[feedback_verify_training_visually]] — the user's visual report ("발도 붙여진 상태")
and their own hypothesis about link/joint mismatch is what triggered this investigation; the
training-log numbers alone never surfaced it.
