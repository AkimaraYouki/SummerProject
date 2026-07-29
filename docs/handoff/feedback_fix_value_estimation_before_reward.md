---
name: fix-value-estimation-before-reward
description: "When reward tuning keeps producing only local improvements, suspect the critic — bad value estimates corrupt the advantages every reward change is filtered through"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-29T05:47:11.053Z
---

If eleven reward-shaping attempts each move things only locally, stop tuning the reward and look
at what sits **upstream** of it. PPO learns from `A = R − V(s)`; a poor `V(s)` turns
situation-driven return variance into advantage noise, and no amount of reward craft survives
that.

Two findings in [[project_openduck_autonomous_training]], both on this axis, both far larger than
any reward change:

**Asymmetric critic.** The policy must see only what the real robot has (noisy IMU, joint
angles) — that constraint is correct and deliberate. The critic is training-only and never ships,
so it can see whatever the simulator knows. This port had `state_space=0`, so the critic shared
the policy's blinkered 101-dim view and could not tell "slipping" from "walking well" behind
identical observations. Giving it 205 dims (true base velocity, noiseless sensors, torques, foot
velocities, the full reference frame) **tripled per-step reward with the reward function
untouched** — which is what made the comparison clean.

**Discount factor vs the task's own period.** `gamma` 0.99 is a ~100-step horizon; 0.97 is ~33.
The gait period was 27 steps. Asking the critic to estimate three and a half cycles ahead on a
periodic task blurs it; one cycle does not. Changing only gamma gave +53%.

**How to apply:**
- Before another reward iteration, check: does the critic get privileged observations? Is
  `gamma`'s effective horizon (~1/(1−γ)) sane against the task's natural period?
- Change one thing and verify it is actually one thing. I claimed a `num_minibatches` effect that
  did not exist because reverting an inherited config block had silently changed five parameters;
  a control run (minibatch 4 → 16) landed exactly on the baseline curve and exposed it.
- **PPO hyperparameters do not port across implementations.** brax's `num_minibatches=32` becomes
  128 gradient steps per iteration in rsl_rl, KL overshoots, and rsl_rl's `schedule="adaptive"`
  drives the learning rate to its 1e-5 floor — brax uses a fixed lr with no KL adaptation. Always
  check `Loss/learning_rate` and `Policy/mean_noise_std` trajectories, not just the config values.
- Exploration noise must be smaller than the precision the reward pays for. `std 0.75 ×
  action_scale 0.25` is ~10° of joint jitter against a 7-8° tracking error — the noise swamped
  the signal and the run flatlined.
