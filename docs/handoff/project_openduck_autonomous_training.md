---
name: openduck-autonomous-training
description: "Open Duck Mini IsaacLab RL pipeline — user asked for unattended autonomous training/verification while asleep, like the TD3 project"
metadata: 
  node_type: memory
  type: project
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-29T05:46:52.052Z
---

**UPDATE 2026-07-29 ~15:00 — imitation_v24 is the current best. gamma 0.99 -> 0.97 was the second
half of the same story as the asymmetric critic.**

v20 bundled the critic fix with a literal port of brax's PPO numbers. Separating them took four
runs and one wrong attribution worth remembering: I described v22-vs-v20b as a `num_minibatches`
difference, but reverting v21's algorithm block had changed five things at once (steps 20/24,
minibatch 32/4, epochs 4/5, gamma 0.97/0.99, lr init). **v23 raised minibatch 4 -> 16 and landed
exactly on v22's curve** (0.192 at iter 160 for both), so minibatch is irrelevant here and the
earlier "+54% from minibatch" claim had no basis. Isolating gamma alone (v24) gave **+53%**
(0.228 -> 0.348 at matched iterations) and, unlike v20b, kept lr alive (4.4e-4 vs v20b's 1e-5
floor), so it passed v20b's frozen ceiling and settled at 0.366.

**Why gamma mattered is the same axis as the critic**: 0.97 is an effective horizon of ~33 steps
against this robot's 27-step gait period — one cycle. 0.99 is ~100 steps, three and a half
cycles. Asking the critic to estimate three cycles ahead on a periodic task blurs the value
estimate. Both wins were about making value estimation accurate, not about making the reward
cleverer. **Also: PPO hyperparameters do not port across implementations** — brax's
`num_minibatches=32` becomes 128 gradient steps per iteration in rsl_rl (v17 did 20), KL
overshoots, and rsl_rl's `schedule="adaptive"` crushes lr to its floor; brax uses a fixed lr with
no KL adaptation.

v24 final (model_2999, pinned-command rollouts): forward 0.148 vs 0.15 cmd (**99%**), backward
-0.131 (87%), left 0.142 / right -0.162 against ±0.2 (71% / 81%, up from 58/67), joint RMS
6.5-9.2deg, dominant frequency 1.90 Hz against the gait's 1.85, verdict WALKING-LIKE.

**Stop command** — a condition that had been missing from `gait_compare.py` entirely until the
user asked "정지는 잘됨?". Added as a sixth condition. v20b -> v24: residual speed 109 -> 46 mm/s,
yaw |mean| 0.692 -> 0.142 rad/s, foot toggles 155 -> 58/10s, feet planted 1.37 -> 1.84 of 2.
Not a true standstill (46 mm/s drift remains) but it now stops when told to, without any
stop-specific reward change.

Two corrections to earlier notes in this file: `zero_command_prob = 0.1` and 500-step command
resampling were **already implemented and match upstream** — my claim that pure-stop commands were
barely trained was wrong. And domain randomization (7 knobs incl. ±1 m/s pushes) has been on from
the start; terrain is flat, same as upstream's `flat_terrain` default.

Remaining: lateral tracking (71-81%) lags forward (99%); 46 mm/s residual drift when stopped;
multi-terrain not started.

---

**UPDATE 2026-07-29 ~09:30 — BREAKTHROUGH: the asymmetric critic was the missing piece, not the
reward.** After ~11 reward-shaping attempts moved things only locally, the user asked whether our
setup actually matches upstream. It did not, and the important gap was the critic: upstream
Open_Duck_Playground builds a `privileged_state` and feeds it to the value network
(mujoco_playground sets `value_obs_key="privileged_state"`), while this port had `state_space=0` —
documented as a deliberate v1 simplification — so **the critic saw the same noisy, partial 101-dim
view as the policy**. A bad value estimate corrupts the advantages that every reward change is
filtered through, i.e. it sits upstream of all the shaping work.

Implemented as `JoystickEnvCfg_Walk9` / imitation_v20: critic obs is 205-dim (policy state +
noiseless sensors + true base velocity + torques + foot velocities + root height + the FULL
36-dim reference frame). Reward config left at v17's values so the run isolates the change.
**Because the reward is byte-identical to v17, per-step reward is directly comparable for once**
(every earlier cross-version comparison was confounded by scale):

| iter | v17 per-step / eplen | v20 per-step / eplen |
|---|---|---|
| 120 | 0.1135 / 133 | **0.2244 / 297** |
| 240 | 0.0994 / 143 | **0.2973 / 372** |

3x per-step, 2.6x episode length. Playback with the command pinned forward: `vy = -0.005`
(v17 swung ±0.15, v16 ±0.30 — lateral drift essentially gone). **User's verdict: "그거 빼면
제일 안정적임 지금까지"** — the most stable policy so far.

Two things that came with it and are NOT wanted:
- Porting brax's `num_minibatches=32` literally means 32x4 = **128 gradient steps per iteration**
  in rsl_rl (v17 did 4x5 = 20). KL overshoots, and rsl_rl's `schedule="adaptive"` drove the
  learning rate into its 1e-5 floor by iter 235. brax uses a FIXED lr with no KL adaptation, so
  the number does not transfer. **Hyperparameters are not portable across PPO implementations.**
- `entropy_coef` 0.01 -> 0.005 collapsed `mean_noise_std` 0.99 -> 0.27 (v17 held ~0.8).
Performance still climbed regardless, which is how large the critic's contribution is. Next run
should keep the critic + network but restore v17's PPO params.

Also fixed, found via pinocchio FK on the Mac: `_reset_idx` pins root z to READY_BASE_HEIGHT
while RSI separately overwrites the legs to a random gait phase — the two never consult each
other. Required spawn height varies 116.7-126.4 mm over the cycle, so low-foot phases spawn
5.4 mm into the floor and PhysX ejects the robot (the user's "처음에 뿅하고 튀어오름").
`SPAWN_BASE_HEIGHT = 0.130` is now separate from READY_BASE_HEIGHT (which stays the
collapse-termination reference). Playback also always clears `push_robot` now.

---

**UPDATE 2026-07-29 — SECOND independent reason v1-v9 failed: upstream's imitation term nets
to ZERO in this setup.** The user noticed imitation_v14's TensorBoard curve overlaid v11's
almost exactly and asked whether that was right. It was a real signal. `reward_at_ready.py` on
all three configs at the READY pose:

| term | v11 (Walk) | v13 (Walk3) | v14 (Upstream) |
|---|---|---|---|
| alive | 0.200 | 0.060 | **0.400** |
| imitation | 0.284 | 0.254 | **0.004** |
| joint_pos_rew | 0.467 | 0.493 | **0.112** |

Upstream's imitation is **1% of alive**: the unbounded joint-position penalty
(-0.2023 x 15 = -3.03 raw) cancels the positive tracking terms (~+3.2 raw) almost exactly, so
imitation collapses to ~0 and the policy trains on `alive` alone -- the reward-hack regime by
construction. So Open_Duck_Playground's recipe cannot work here even with the pose fixed.
Stopped v14 at iter ~900 rather than run it out.

Same measurement exposed a mirror-image error of my own: v13's `imitation_w_joint_pos=4.0` was
derived from the reference's *standing* spread (0.234 rad^2), but the trained policy's
*in-motion* error is 0.684 rad^2, where exp(-4*0.684)=0.065 -- saturated, flat gradient, no pull
back toward the pose. Matches v13's symptoms (joint RMS 9.9deg -> 16.5deg, amplitude 1.31x the
reference). v15 (`JoystickEnvCfg_Walk4`) sets k=1.5 = 1/0.684. **Lesson: derive an exp
sensitivity from the error the policy actually produces, not from the reference's own spread.**

v13 results (best so far, `JoystickEnvCfg_Walk3`, the stance-violation penalty): forward
0.117 m/s (v12 0.097), left 0.091 (v12 0.005), contact toggles 122-134/10s (v12 144-319),
per-step foot toggles 0.238 (v12 0.72), base speed 0.165 (v11 0.064).

---

**UPDATE 2026-07-28 ~19:10 — ROOT CAUSE OF v1-v9 FOUND (action space physically could not reach
the reference pose); fixed via HOME/READY split; v10/v11 still failed for a SECOND, separate
reason (lin_vel_xy's exp sharpness could not tell standing from walking); imitation_v12 now
training with --livestream 2 at user's request.**

**The 8-sigma bug (why v1-v9 all failed identically).** Joint targets are
`target = default_joint_pos + action * action_scale(0.25)`. `default_joint_pos` was HOME = all
zeros (straight legs), but the reference gait needs knee ~= 2.03 rad, i.e. `action = 8.12` —
**8 sigma outside** the PPO policy's `init_noise_std=1.0` Gaussian. The reference pose was
unreachable by exploration, so **every reward-weight experiment in v1-v9 was tuning a knob on a
system that physically could not do the task.** Found from the user's own hypothesis ("이미테이션과
현실의 링크나 조인트 위치가 안 맞는 거 아냐? 확인해봐"). Fix: split HOME (physical rest, all
zeros, base 0.193 m — for real hardware) from **READY** (the gait's mean crouch: hip_pitch ~1.11,
knee ±2.03, ankle ±0.98, base 0.121 m) and make READY the articulation's `init_state`, hence
`default_joint_pos`. Measured effect: joint_pos error 79° -> 7.8°, imitation -1.011 -> +0.290/step,
clamped steps 81.6% -> 0%, required action 8.1 sigma -> 1.30 sigma. Also fixed a latent bug this
exposed: `_reset_idx`'s joint randomization was *multiplicative*, a silent no-op while HOME was
zeros, and would have scattered READY's 2.03 rad knee to [1.0, 3.0] past the ±2.094 limit — now
additive ±0.05 rad.

**The second, separate cause (why v10/v11 still failed).** User watched v11 over WebRTC: "또
부들부들거림 발 안뜸... 걸을려고 안함", and correctly spotted "지금 리워드가 v10과 같은 양상"
(both runs: episode length rising while per-step reward falls). Stopped v11 at iter ~320 rather
than burn 3 more hours confirming a known failure, and measured `model_300` with a new
`scripts/imit_internals2.py` (adds Walk-task mapping, `swing_only_contact`, and **foot-toggle /
actual-speed** measurement — the behavioral half `imit_internals.py` lacked):
```
actual speed 0.064 m/s  (reference 0.265)   <- barely moving
lin_vel_z +0.954/1.0   ang_vel_xy +0.220/0.5   lin_vel_xy +0.556/1.0
joint_pos +0.379/1.0   ang_vel_z  +0.258/0.5   contact    +0.205/~0.63
joint_vel -0.056       <- the term I had predicted was the culprit: 2% of the total, irrelevant
```
**~92% of the imitation reward was collectable while standing still.** The decisive term is
`lin_vel_xy` — the *only* term whose job is to price walking — because at sharpness `k=8`,
being wrong by the entire reference speed (err² = 0.265² = 0.070) still pays
`exp(-0.56) = 0.57`. **Two of my own hypotheses were wrong and are recorded as ruled out**:
(a) unbounded `joint_vel` suppressing motion — measured at -0.056, irrelevant; (b) policy can't
observe gait phase — `imitation_phase` (cos/sin) is present in the obs (`joystick_env.py:289-304`).

**imitation_v12 (`JoystickEnvCfg_Walk2`)**: `k_lin_vel_xy` 8->20, `w_lin_vel_z` 1.0->0.1,
`w_ang_vel_xy` 0.5->0.1 (reference spread is 0.0000 — zero discriminating power by construction),
`w_contact` 1.0->2.0, `alive_scale` 10->3, and **`use_rsi` back ON**. RSI rationale: the v7-vs-v8
"RSI makes no difference" conclusion was measured while `default_joint_pos` was still HOME, i.e.
while RSI was initializing into mid-stride poses the policy could not hold under *any* action —
that experiment measured nothing. Verified before launch (`reward_at_ready.py`): standing-still
reward 0.734 (v10) -> 0.583 (v11) -> **0.444 (v12)**, clamp 0.1%, Mac unit tests 7/7, and
`reward_imitation`'s defaults unchanged so v1-v11 stay reproducible.
**Run conditions**: user asked for visualized (not headless) training — `--livestream 2`.
`num_envs=4096` dies instantly under rendering with CUDA illegal memory access (RTX 5080, 16 GB),
so **1024**; ~5.4 s/iter, ETA ~4h30m. **Hack threshold for this run: standing = 0.444/step, so
0.40-0.50 plateau + episode length 400+ = hacking again; >=0.55 = real walking signal.**

---

**UPDATE 2026-07-28 ~07:37 — imitation_v8 (RSI OFF, Disney-literal) also FAILED (toggle 96.7,
worse than v7's 90.6), despite looking dramatically more stable during training than v6/v7 —
third confirmation that training-time aggregate stats don't predict eval-time policy quality.
Three-way v6/v7/v8 comparison table now in docs/training_log.md. All three attempts (v6/v7/v8)
have failed; stopped the autonomous chain here to report to the user rather than launch another
large experiment on my own.**
v8 completed 3000 iterations with markedly better training-time metrics than v6 or v7 throughout
(episode length 60-73 vs v6/v7's 30-65 at matching iterations; value_function loss stayed
0.06-0.14 the whole run vs v6/v7 climbing to 1.2-2.8) — genuinely looked like the best run yet
while it was training. Eval said otherwise: foot-toggle 96.7, slightly worse than v7's 90.6 and
far worse than v6's 29.4.
**Three-way summary**: v6 (RSI on, num_envs=512) = 29.4 [best by far] / v7 (RSI on, num_envs=4096)
= 90.6 / v8 (RSI off, num_envs=4096) = 96.7 [worst]. The only clean isolated comparison (v7 vs
v8, num_envs held at 4096, only RSI differs) shows RSI-on marginally beating RSI-off (90.6 vs
96.7) — a 6.7% difference, plausibly within run-to-run noise rather than a real effect. v6's
huge lead is confounded with num_envs=512, not isolated from RSI, and has no repeat run to check
if it was a lucky seed.
**Bottom line for the user's "just use Disney's simple recipe" request**: dropping RSI did NOT
fix anything — v8 is arguably the worst of the three. Simplifying toward Disney's literal
recipe was a reasonable, well-motivated thing to try (and worth having ruled out cleanly), but
it isn't the answer either. All three post-crouch-fix attempts (v6/v7/v8) have now failed.
**Decision**: stopping the autonomous launch-next-experiment chain at this point rather than
picking another direction unilaterally — three failed attempts in a row without a proposed
next step is exactly the kind of moment to report status and let the user weigh in, even under
the standing full-autonomy grant (which was about not needing to ask before EACH step, not a
mandate to never surface "we've tried three things and none worked, what now"). Candidate next
directions not yet tried, for reference if the user asks: bump `max_iterations` well beyond 3000
(literature suggests our total training scale is still tiny vs. Disney's), retry
`reward_breakdown.py` now that the GPU is idle (never got a clean measurement of the clamp's
actual impact), or the literature survey's remaining candidates (symmetry loss, gait-phase
reward) — though the user's stated preference leans against adding more custom techniques.

---

**UPDATE 2026-07-28 ~05:15 — imitation_v7 (RSI + num_envs=4096) FAILED, and surprisingly WORSE
than v6 (toggle 90.6 vs v6's 29.4, a 3x regression) — refutes the "v6 was just under-trained"
hypothesis; imitation_v8 (RSI OFF, Disney-literal A20J5) now training as the first true A/B test.**
v7 completed 3000 iterations cleanly (final training-log episode length 89-91, actually higher
than v6's 64-82) but eval came back foot-toggle **90.6** — 3x worse than v6's 29.4, back down
near v3/v4's ~79-80 range. worst-upright also slightly worse than v6. This directly contradicts
the standing hypothesis that v6's num_envs=512 (8x less data) was the bottleneck — scaling to
4096 made toggle *worse*, not better, despite the training-time episode-length metric looking
fine. Notable disconnect: training-log episode length improved but eval-time stability got worse
— a reminder that training-time aggregate stats and eval-time rollout behavior can diverge, this
project's central recurring lesson (see [[feedback_verify_training_visually]]). Can't yet
distinguish "num_envs itself caused the regression" from "this is just RL run-to-run seed
variance" — v6 and v7 differ in exactly one variable (num_envs) but RL training has high variance
even with identical configs, and no repeat runs exist to separate the two explanations.
Given this result, launched the already-planned `imitation_v8` = A20J5 config with **RSI turned
off** (`use_rsi=False`, matching Disney's literal recipe — confirmed Disney doesn't use RSI),
`num_envs=4096`, headless. Short validation (num_envs=64, 200 iter) passed first (no crash,
episode length 40-45, normal range) before committing to the full run. This is the first genuine
controlled comparison isolating RSI's own effect (v7 vs v8, same num_envs=4096, only RSI differs)
— v6 vs v7 only ever isolated num_envs (RSI was on for both), so RSI's true contribution has
still never been cleanly measured until v8 finishes.
Symlink farm `/tmp/tb_compare` now holds v3/v4/v5/v6/v7/v8.

---

**UPDATE 2026-07-28 ~04:30 — user pushed back wanting "just Disney's simple recipe, not so much
extra stuff piled on"; confirmed RSI is NOT in Disney's paper; added a `use_rsi` toggle rather
than deleting RSI, planning imitation_v8 = A20J5 with RSI off as a direct comparison against v7.**
User asked "디즈니꺼는 rsi 씀?" — checked the actual paper (WebFetch on arxiv.org/html/2501.05204v1)
and confirmed RSI/reference-state-initialization is never mentioned; Disney's only phase-related
discussion is about runtime animation-blending phase selection, not RL episode initialization.
User then said "디즈니의 그 간단한 모방학습만 쓰고싶은데 너무 거추장스러운거 다는게 아니라..." — wanting
to strip back toward Disney's literal recipe rather than keep layering on techniques beyond it.
I tried to clarify scope via AskUserQuestion; **the user rejected that tool call** and a system
instruction forced a text-only response for that turn — so I did NOT re-ask, just stated my own
interpretation directly in the next reply and proceeded: **RSI off** (confirmed not in Disney's
paper), but **contact-based termination + the crouch-fix (height ratio 0.75, knee/ankle contact)
stay on** — these aren't "extra" additions, they're literally what Disney's paper itself describes
(terminate on head/torso ground contact), so keeping them is consistent with "just Disney's
recipe," not a deviation from it. A20J5 weights also stay (alive_scale=20 was always Disney's own
number). Implemented as a `cfg.use_rsi: bool` toggle (default True) rather than deleting the RSI
code, specifically so "RSI on" (v6/v7) and "RSI off, Disney-literal" can be compared directly on
the identical codebase — new config `JoystickEnvCfg_A20J5_NoRSI`, gym task
`Isaac-OpenDuckMini-Joystick-A20J5NoRSI-v0`.
**Plan**: once `imitation_v7` (RSI on, num_envs=4096) finishes — eval + mandatory visual
confirmation per [[feedback_verify_training_visually]] as always — run a short validation
(num_envs=64, 150-200 iter) on the new no-RSI path (toggling a cfg flag is exactly the kind of
change that's bitten us with silent indexing bugs before — the actual RSI pose-write code is now
gated by `use_rsi` too, so verify it doesn't misbehave when skipped), then launch `imitation_v8`
= A20J5NoRSI on num_envs=4096, compare directly against v7's result to answer: does RSI actually
help here, or was v6/v7's improvement (toggle 29.4, ROM 1.31) really just from the num_envs
scale-up / more total training, independent of RSI? This is the first real controlled A/B test
of RSI's own contribution — everything before this only ever changed RSI and num_envs together.

---

**UPDATE 2026-07-28 ~02:45 — imitation_v6 (RSI+A20J5) is the first run to show genuine
walking-attempt behavior instead of reward-hacking; imitation_v7 (RSI + num_envs scaled back
to 4096) now training.**
`imitation_v6` completed 3000 iterations (after one mid-run crash+restart caused by the user
experimenting with USD render output over WebRTC, corrupting the render pipeline — fixed by
restarting; see below). Eval: foot-toggle **29.4** (best of ALL runs including v2's Stage-1
baseline of 43.4 — first time any run beat that), leg-joint ROM **1.31 rad** (2.3-3x every prior
run's ~0.42-0.56) — but lin-vel tracking error got WORSE (0.43 vs v4/v5's ~0.19) and the eval
script's automated verdict still says FAIL. User watched live over WebRTC: "계속 넘어짐" (keeps
falling) and asked if the IMU was broken. Re-checked the sim IMU code: accelerometer correctly
includes gravity bias (`lin_acc_w = Δv/dt + gravity_bias_w`, confirmed in IsaacLab's imu.py
source), and the missing-orientation-in-observations quirk is inherited dead-code from Playground
present in v1-v6 alike, not new to v6 — concluded this is NOT an IMU bug, it's a policy that's
attempting much bigger, more walk-like motions (matches the ROM/toggle numbers) but hasn't
learned balance yet.
**Judgment call (autonomous)**: v6 used `num_envs=512` (down from the standard 4096) specifically
so the user could watch it live — an 8x reduction in samples per iteration. Given v6 is
qualitatively the first non-reward-hacking result, the most likely explanation for "still falls"
is under-training from that 8x data cut, not a flaw in RSI itself. Rather than layering another
new lever (symmetry loss, gait-phase reward) onto an already-promising-but-underpowered setup,
launched `imitation_v7` = identical RSI+A20J5+contact-termination+crouch-fix config, just
`num_envs` back to 4096, `--headless` (traded live-viewing away for this run). No new code
changed, so skipped the usual short-validation-first step.
User then asked to "use max resources for speed" — tried `num_envs=8192` (Disney's paper scale)
first, but ETA got WORSE (2h50m → 4h48m) because the GPU was already ~70% compute-utilized at
4096 (memory had headroom, but compute was the real bottleneck) — reverted to 4096 as the
empirically-fastest point on this GPU. Worth remembering: more envs isn't free throughput once
compute-bound; check `nvidia-smi` utilization%, not just free memory, before scaling up.
Symlink farm `/tmp/tb_compare` now holds v3/v4/v5/v6/v7 for continued live TensorBoard overlay.

---

**UPDATE 2026-07-27 ~18:47 — user granted full autonomy ("실측하고 너가 방향성 정해서 알아서 돌려,
내 개입없이"); imitation_v5(A30J25) was the WORST result yet (toggle 159.1); pivoted away from
the pre-approved weight-doubling v6 to implementing RSI instead; imitation_v6(A20J5+RSI,
num_envs=512, WebRTC-rendered per user request) now training.**
`imitation_v5` (A30J25: alive_scale 20→30, w_joint_pos 5→25, on top of the crouch-fix) completed
3000 iterations cleanly but FAILED worse than every prior run: foot-toggle 159.1 (vs v2=43.4,
v3=79.1, v4=78.9) — confirms the "raise both weights further" direction the user asked to try
was actively harmful, consistent with the literature survey's warning (arXiv:2606.02636) about
generous survival rewards. Episode length was also anomalous: 6-8x shorter than v4 at matching
iterations early on (crouch-fix's stricter termination firing far more often), recovering to
~128 by the end — so the crouch exploit itself is confirmed closed, but the policy moved to a
different, worse twitching failure mode instead.
Tried to get `scripts/reward_breakdown.py` (fixed a real bug in it first: it hardcoded
`JoystickEnvCfg_A20J5` regardless of `--task`, silently using wrong alive_scale/w_joint_pos for
any other variant — now uses `parse_env_cfg`) to quantify the user's clamp-cancellation
hypothesis, but it crashed silently 3 times in a row with zero GPU contention this time —
concluded the script itself is unreliable right now and moved on without further retries rather
than blocking on it (user's explicit "don't need me for this" authorization made that the right
call).
**Implemented RSI (Reference State Initialization, DeepMimic/Peng et al. 2018)** — the literature
survey's top-ranked alternative, and something none of v1-v5 had ever touched (all varied only
alive_scale/w_joint_pos). `_reset_idx()` previously always reset `self._imitation_i` to phase 0
and always spawned from the same default pose; now samples a uniform-random phase and sets the
spawned leg joint pos/vel to match the reference frame AT that phase (not just randomizing the
reward-target phase while still spawning from the same pose, which wouldn't be real RSI). Caught
and fixed a real indexing bug in this on the first validation attempt (episode length pinned at
exactly ~8, a dead giveaway): `ACT_LEG_JOINT_IDX` is in ACTUATOR_JOINT_NAMES order but
`joint_pos`/`joint_vel` in `_reset_idx` are in the articulation's native (USD) order — needed
`self._joint_ids[i] for i in ACT_LEG_JOINT_IDX]` to map correctly. Second validation run came
back normal (episode length 35-39, matching prior healthy validations).
**Decision**: test RSI's effect isolated on the A20J5 baseline (not A30J25, which is now
confirmed the worst config) — one-variable-at-a-time discipline. Launched `imitation_v6` =
A20J5 + contact-termination + crouch-fix + RSI, `num_envs=512` (down from 4096, per user's
explicit request to trade data volume for live viewability) with `--livestream 2` instead of
`--headless` so the user (or I) can watch it live over WebRTC while it trains, ETA ~2h32m from
~18:47.
**Standing authorization note**: user explicitly said to keep deciding direction and executing
without asking, including keeping runs overlaid in the same TensorBoard (`/tmp/tb_compare`
symlink farm) so progress is visible at a glance — this expands on the earlier per-run approval
pattern (A30J25, A30J25Im2N2 were individually pre-approved; this grant is now standing until
revoked).

---

**UPDATE 2026-07-27 ~15:05 — imitation_v4 crashed near the finish (iter 2983/3000, value-function
loss exploded to NaN); WebRTC viewing of model_2900.pt revealed a NEW reward-hack (stable
"crouch" evading all 3 termination conditions at once); fixed and imitation_v5(A30J25) now
training with the fix included.**
`imitation_v4` (A20J5 + contact termination) never reached iteration 2999 — `Mean value_function
loss` exploded from 63.96 (iter 2978) to 4.93e32 (iter 2983) across 5 iterations, crashing with
`RuntimeError: normal expects all elements of std >= 0.0` (NaN action-distribution std). Root
cause of the explosion itself not investigated (deprioritized — see below for what mattered
more). Evaluated the last good checkpoint (`model_2900.pt`, 96.7% through): FAIL, foot-toggle
78.9 (matches v3's 79.1 almost exactly) — so the contact-termination fix alone did NOT change
the twitching failure mode's magnitude.
User then watched `model_2900.pt` live via `play.py --livestream 2` and reported: "터미네이션
안될려고 무슨 플랭크같이 머리랑 발로 버티는 동작과 시작하자마자 땅이랑 겹치는지 튕겨나가고 행동이
고착됨" (policy holds a plank-like pose propped on head+feet to avoid termination; also appears
to clip into the ground at spawn and get flung, then locks up). Wrote `scripts/contact_diagnostic.py`
(now in the repo) to log real per-step trunk/head contact-sensor force + base height during a
rollout — found:
- Trunk/head contact force was **exactly 0.000N for all 200 sampled steps across 8 envs** —
  never once nonzero. Confirms `enabled_self_collisions=False` (in `robot_cfg.py`) means the
  head can visually overlap/rest against its own legs with **zero generated contact force** — no
  ground contact, so the new contact-termination literally cannot fire regardless of threshold.
  (Lowering the threshold, which the user also asked to try, was therefore a dead end — verified
  empirically before committing to it as the fix.)
- `base_z` (root height) settles into a rock-stable crouch at **0.133-0.134m** (HOME_BASE_HEIGHT
  is 0.193m, so ratio ≈0.69) for 100+ consecutive steps — just above the old
  `min_base_height_ratio=0.6` collapse threshold (0.1158m), so height-termination doesn't fire
  either. Combined with never being flipped, this crouch evades **all three** termination
  conditions (flip / height / contact) simultaneously — a new, more sophisticated reward-hack
  than v3's twitching.
Fix (both applied, per user's explicit choice of "do both"):
1. `min_base_height_ratio`: 0.6 → 0.75 in `joystick_env_cfg.py` (puts the measured 0.69-ratio
   crouch below the new threshold).
2. Contact-termination body list extended in `joystick_env.py` from
   [`trunk_assembly`, `head_pitch_assembly`] to also include all 4
   `knee_and_ankle_assembly*` bodies (both legs) — DeepMimic's own rule ("any bone other than
   feet touching ground → terminate", confirmed in the literature survey below) generalizes to
   "legs folding onto the ground" too, not just trunk/head.
Validated with a short run (num_envs=64, max_iterations=150) — no crash from the new body names,
episode length stayed in the normal 40-45 range (not collapsing to near-zero). Launched
`imitation_v5` (A30J25 + these two fixes, num_envs=4096, max_iterations=3000) — training now.
**Process note for future sessions**: this crouch-detection workflow (watch live via WebRTC →
if something looks like reward-hacking, write a targeted diagnostic script to log the actual
sensor values a rollout produces, rather than guessing from the training-log numbers alone) is
now the established playbook here — see also [[feedback_verify_training_visually]].

---

**UPDATE 2026-07-27 ~13:30 — literature survey (10 papers) done; v5(A30J25)/v6(imitation×2+noise×2)
direction is weakly-supported, better alternatives identified for v7 if both fail.**
Surveyed DeepMimic (Peng et al. 2018), Disney BD-X (Grandia et al. 2024), AMP (Escontrela/Peng
2021-22), ToddlerBot (Kwon et al. 2025, closest scale match at 3.4kg/0.56m), a 2026 paper
explicitly titled "Too Much of a Good Thing: When sim2real Efforts Impede Policy Learning"
(arXiv:2606.02636), a 2025 symmetry-loss humanoid paper (arXiv:2508.01247), MuJoCo Playground's
own GitHub Discussion #97 (other Open Duck Mini users hitting adjacent gait-quality issues, no
settled fix there either), and others. Key takeaways:
- User's current plan (raise alive_scale 20→30 AND imitation_scale 1.0→2.0 simultaneously,
  already coded as imitation_v5/A30J25 and imitation_v6/A30J25Im2+init_noise_std=2.0) is **weakly
  supported by the literature** — arXiv:2606.02636 explicitly warns that generous survival
  rewards produce "upright but not progressing" policies, the opposite direction from what we're
  trying. No source found that validates "raise both together" as an established fix for a
  joint-lock-and-pop local minimum.
- **Better-evidenced alternatives for v7 if v5/v6 both fail** (not yet implemented):
  1. **Reference State Initialization (RSI)** — DeepMimic's technique: reset each episode from a
     random point in the reference clip instead of always t=0/default pose. Directly targets
     "policy never practices the hard mid-gait phase" — plausible root cause of joints locking
     near a limit angle. Low risk, doesn't require re-litigating alive_scale/w_joint_pos.
  2. **Symmetry loss** — mirror left/right obs+action, penalize asymmetric policy output as an
     auxiliary PPO loss term. Orthogonal axis to reward-weight tuning.
  3. **Gait-phase/periodic reward terms** — explicit "this foot should be in contact now"
     structure rather than relying on raw imitation-error magnitude to imply stepping.
  4. AMP-style learned discriminator reward (Escontrela et al.) — a tier above manual per-joint
     weight tuning, worth considering if v7's targeted fixes also fail.
- Confirmed (reassuring, not new work needed): our newly-added contact-based termination is the
  same lineage as DeepMimic's (2018) "any bone but feet touching ground → terminate" — the
  correct choice, already implemented.
- User's call: decide after v5/v6 results land, not before. Do not preemptively switch to RSI/
  symmetry-loss without discussing first — this is a real direction change, not a parameter tweak.

---

**UPDATE 2026-07-27 07:00 — imitation_v3 (A20J5 full run) FAILED, worse than v2; added
Disney-paper contact-based termination; imitation_v4 now training (ETA ~3h).**
`imitation_v3` (3000 iter, num_envs=4096) completed but FAILED eval: foot-contact toggle 79.1
(vs v2's 43.4 — 1.8x WORSE), worst-upright -0.44 (more upright than v2's -0.0143, but far more
twitchy). So the sweep's short-run (800-iter) A20J5 win did NOT generalize to full-scale — the
`w_joint_pos` lever alone wasn't the answer. WebRTC live view of the checkpoint confirmed
visually: joint locks near a limit angle then "pops"/flings outward → terminates.
Separately, user shared an artifact analyzing the Disney BD-X paper (Grandia et al. 2024 — the
actual design reference Open Duck Mini reproduces) vs our reward/termination setup. Two
findings: (1) `alive_scale=20` is literally Disney's own number, not a Playground invention —
it's fine now that imitation (our Stage 2) is on. (2) Disney's termination condition is
**contact-based** ("head or torso touching ground") rather than height-ratio/flip heuristics —
more robust since no collapsed pose can evade it. Implemented this: `joystick_env.py` now
reads `ContactSensor` net force on `trunk_assembly`/`head_pitch_assembly` (NOT literally "head"
— that URDF link has no inertia and gets merged into its parent on import; this cost one
crash-and-fix cycle) and ORs it into `_get_dones()` alongside the existing flip/height checks
(additive, not a replacement). Validated with a short run (num_envs=64, 200 iter) — episode
length rose 13→50 then settled ~42-45, no instant-termination bug. Launched `imitation_v4`
(A20J5 config unchanged + contact termination, num_envs=4096, max_iterations=3000, run dir
`.../imitation_v4`) — training now, ETA ~3h from 2026-07-27 ~07:00.
Also found (not yet resolved): `_get_rewards()` clamps the summed reward to `[0, 10000]` —
any step where penalty terms push it negative gets floored to exactly 0, discarding the
magnitude. This plus the `rewbuffer` mechanics (tensorboard's "Mean reward" is a full-episode
cumulative sum, not a per-step average — confirmed by reading rsl_rl's on_policy_runner.py
source) explains why raw reward numbers look implausibly low; a `scripts/reward_breakdown.py`
diagnostic was written to quantify the clamp's actual impact but has crashed twice from GPU
contention with a concurrent Isaac Sim instance (once vs. training, once vs. the WebRTC
streaming server) — needs a run with the GPU fully idle to complete.
WebRTC streaming (`docs/webrtc_streaming.md` — signal port 8011, media port 49100,
`--/app/livestream/publicEndpointAddress=192.168.137.111` required) confirmed working for both
an empty-scene session and `play.py --livestream 2 --checkpoint ... --real-time` (loads a
trained policy directly into the streamed scene) — the latter is the way to actually watch a
policy live, not just an empty stage. Two Isaac Sim instances (training + anything else) on
this one GPU reliably crash one of them — never run a second one while training is active.

---

**UPDATE 2026-07-27 03:20 — sweep result: alive_scale=5 collapses training; A20J5 (alive=20,
w_joint_pos=5) is the clear standout. Awaiting user go-ahead for a full imitation_v3 run.**
Ran 4 concurrent short screening runs (num_envs=1024, max_iterations=800 — ~6.7% of a full
run's total steps) to disentangle the two reward-weight candidates from the previous note.
Result: `alive_scale=5` (A5J15, A5J5) makes training COLLAPSE — value function loss flatlines
to 0.0000, action noise std explodes to 4.35 (policy ≈ random noise), episode length stuck
~35 steps. `alive_scale=10-20` trains stably. Among the stable ones, **A20J5 (alive_scale=20
unchanged, w_joint_pos 15→5) is dramatically healthier** than A10J10 — reward 0.98 vs 0.02,
episode length 154 vs 67, and on eval_policy_stability.sh its foot-contact toggle (104.9) is
LOWER than A10J10's (124.9). None of the 4 pass STANDING/WALKING yet (expected given the tiny
training budget — this was a relative-comparison screen, not a convergence test). Conclusion:
the "w_joint_pos=15 too harsh" hypothesis now has real evidence favoring it over "alive_scale
too high" — in fact lowering alive_scale is actively harmful, the opposite of the original
suspicion. BAM/PD-gains ruled out separately (see [[project_openduck_bam_pd_gains]], user
verdict: current values are fine, no re-measurement needed).
Full comparison tables in docs/training_log.md ("alive_scale × w_joint_pos 스윕" section) and
report artifact rev7: https://claude.ai/code/artifact/332b789c-544e-432e-a702-e01cfc4bdf99
**Next**: proposed relaunching a full run (num_envs=4096, max_iterations=3000, config
JoystickEnvCfg_A20J5 already registered as gym task `Isaac-OpenDuckMini-Joystick-A20J5-v0`)
as `imitation_v3` — asked user for go-ahead, did not auto-launch (3h GPU commitment).

**(historical) UPDATE 2026-07-27 02:00 — imitation_v2 result: FAIL (twitching, worse than v1). Waiting on user.**
Trained to completion (iter 2999, ~295M steps — ~2x upstream's default 150M). eval_policy_stability.sh
verdict: STANDING=COLLAPSED/UNSTABLE, WALKING=REWARD-HACKING. worst-case upright -0.0143 (near
sideways), lin-vel tracking err 0.173 (just over 0.15 threshold), foot-contact toggles 43.4/foot/10s
— nearly 4x v1's 11.7, i.e. MORE violent trembling than v1, not less, despite the reference motion
itself being thoroughly fixed (knee ROM alive, symmetric, 0 limit violations, drift-corrected). This
rules out reference-data quality as the remaining cause. Three candidate root causes identified,
none clearly singled out (documented in training_log.md Run 7):
  1. alive_scale=20.0 still ~60% of the theoretical per-step reward ceiling (~33.5) — same value
     flagged as reward-hacking-prone in the earlier Stage-1 sweep, restored without re-testing once
     imitation was actually working.
  2. w_joint_pos=15.0 imitation penalty may be too harsh against the new reference's wider knee ROM
     (up to 120°), pushing the policy into high-frequency chatter near the target instead of smooth
     tracking (matches the toggle-count spike).
  3. Possible genuine non-convergence despite ~2x upstream's step count — reward stayed noisy
     (0.9-5.5) with no clear upward trend through the back half of training.
Per instructions, did NOT auto-retry (no single clear low-cost fix identified) — reported to user,
awaiting their call on which lever to pull for imitation_v3. Recap artifact rev6 has full writeup:
https://claude.ai/code/artifact/332b789c-544e-432e-a702-e01cfc4bdf99

**(historical) UPDATE 2026-07-27 late — AUTONOMOUS MODE: user left ("알아서 학습 돌려 검증하고 알아서하삼"). imitation_v2 launched (4096 envs, log ~/train_imitation2.log, ~3h ETA). When it finishes: run eval_policy_stability.sh on the final checkpoint (look for STANDING + WALKING verdict lines), record the result in docs/training_log.md Run 7 (both machines, commit), update this memory + the report artifact (개정 6) with the outcome, and if it FAILS analyze root cause before any retry. Recap artifact rev5 published. Head motions are runtime commands (7-dim command incl. 4 head targets) — reference's fixed head is by design.**

**UPDATE 2026-07-27 — knee saturation RESOLVED; data fully ready for imitation_v2.**
User widened both knee mates in OnShape to a symmetric ±120° (±2.0944 rad; first attempt
came out [−90°,+120°] both sides — wrong side for the left knee whose bend direction is
NEGATIVE — second import fixed it). `placo_walk_engine.py`'s per-side knee direction
enforcement now reads the magnitude from the URDF `<limit>` instead of hardcoding ±1.5708,
so future OnShape limit changes flow through. USD rebuilt, reference motions regenerated:
118 gaits, 0 limit violations, 0 direction violations, 100% in-band, and knee ROM is
ALIVE again (left mean 0.44 rad within [−120°,−69°], right mirrored — was frozen at
ROM≈0). pkl re-fit (drift-corrected velocity channels included) and synced/committed both
machines (Mac robot.urdf + viewer URDF copy also synced — remember the stale-viewer-URDF
lesson). Next step: relaunch `imitation_v2` training (4096 envs) when user says go; then
eval_policy_stability.sh. BAM re-measure of XM430 still pending (user's plan).

**(historical, resolved) UPDATE 2026-07-26 ~23:00 — day wrapped; ONE open issue blocks retraining: knee 90° saturation.**

Everything from the 14:30 note below got RESOLVED during the day (OnShape re-export fixed
mass/knee-axis/frames; native Fastened mates now provide trunk/left_foot/right_foot/head
frames with correct mirrored transforms; HOME_JOINT_POS is now all-zero and
HOME_BASE_HEIGHT=0.193m, verified standing PASS in Isaac). Additional fixes landed same
day: Placo `enable_joint_limits(True)` (was False — 100% of gaits violated limits),
medium.json walk_com_height 0.205→0.16 + feet_spacing→0.18 + neck/head preset angles→0,
knee bend direction enforced (left_knee≤0/right_knee≥0 — right knee had been bending
human-style forward), PolyReferenceMotion sparse-grid nearest fallback, fit_poly velocity
channels re-centered onto grid-key commands (planner has an inherent ~2°/s yaw bias,
reproduced on untouched upstream v1 assets — not our robot). Final pkl: 118 gaits, 0
limit violations, 0 knee-direction violations, 100% speed in-band. Local Mac replay works
via `reference_motion_generator/scripts/replay_motion_meshcat.py` (pure pinocchio,
multi-clip playlist; NOTE: keep the Mac copy of
reference_motion_generator/.../open_duck_mini_v2.urdf synced with robot/robot.urdf —
a stale copy caused hours of phantom "detached foot" viewer bugs).

**OPEN ISSUE (why training is paused)**: in the final set the knees are SATURATED at
their ±1.5708 URDF limits (left −1.571 const, right +1.571 const, ROM≈0) — geometry
says walk_com_height=0.16 needs ~105° knee bend but the OnShape mate limit is ±90°.
Gait is limit-clipped stiff-crouch style. Upstream v2 used knee range up to π.
**Tomorrow: user checks the real robot's actual mechanical knee range → update OnShape
mate limits → re-import → regenerate → verify knee ROM alive → relaunch imitation_v2.**
`imitation_v2` was briefly launched (run dir 2026-07-26_22-27-31_imitation_v2) then
stopped at user's request ("내일 할 거임") after a few iterations; GPU confirmed freed.
Also still pending: BAM re-measure of XM430 (stiffness/damping), user's own plan.
Full day story in docs/training_log.md (Run 1~7) and the report artifact
https://claude.ai/code/artifact/332b789c-544e-432e-a702-e01cfc4bdf99 (개정 4).

**(historical, resolved) UPDATE 2026-07-26 ~14:30 — BLOCKED on CAD-level left/right asymmetry, training paused pending user's OnShape fix.**

Stage 2 (`imitation_v1`, num_envs=4096, 3000 iter, 3h11m) ran to completion but FAILED
`eval_policy_stability.sh` (twitching/reward-hacking, not real walking — worst-case
upright -0.01, vel tracking err 0.21). Root-caused a real bug: `auto_waddle.py`'s
speed filter compared `preset_name=="medium"` against values that were actually
`"{index}_medium"` (never matched), so all 240 sweep combos (incl. wildly
out-of-band ones) went into the pkl unfiltered. Fixed (`preset_name.split("_",1)[1]`),
regenerated → 117/240 survived, 100% now within the intended 0.05-0.15 m/s band
(commits `ddbda76`/`b3efee6` fix, `07bd13f`/`929d763` regenerated pkl+plots).

**But then the user caught something bigger** while reviewing the updated verification
report: `left_knee` ROM sits well above its home pose, `right_knee` ROM sits well
below its home pose (not mirrored), and `right_knee`'s recorded max (+0.228) actually
*exceeds* its own URDF joint limit (upper≈0) by 13°. User checked OnShape directly and
found 3 CAD/export-level causes: (1) `left_roll_to_pitch_assembly` (105.16g) vs
`right_roll_to_pitch_assembly` (121.62g) — confirmed via URDF parsing, +16.47g/15.7%
mismatch, but user verified density/volume/surface-area are identical in OnShape, so
this is an **onshape-to-robot export bug** (stray fastener merged into one side's mass,
or a caching/material bug) — not a real design asymmetry, so no CAD redesign needed,
just find what's wrong in the export; (2) `right_knee`'s joint rotation direction is
CW/CCW-flipped vs `left_knee` in the OnShape mate definition itself (angle display same,
rotation sense opposite) — matches the observed joint-limit violation; (3) some assembly
frames were imported in a suppressed state, and the robot may not have been imported in
an upright reference pose, so `HOME_JOINT_POS` might not reflect true physical neutral.

**Decision (user, 2026-07-26)**: pause all Stage 2 retraining until the user fixes these
3 things in OnShape and re-exports the URDF. Retraining now (even with the speed-filter
fix) would likely reproduce a similar asymmetric-gait failure. Full writeup in
`docs/decisions.md` under "좌우 비대칭 발견 — OnShape CAD 레벨 이슈".

**How to apply when resuming**: once the user says OnShape is re-exported, the sequence
is: `scripts/patch_urdf_for_placo.py` (re-inject Placo frame aliases into the new URDF)
→ `scripts/generate_reference_motion.sh` (regenerate reference motion, should now be
symmetric) → re-run the 3 verification plots (`verify_gait.py` should show
left/right-knee ROMs that actually mirror each other and stay within joint limits) →
only then relaunch `imitation_v2` training. Don't skip straight to relaunching training
on the assumption the fix worked — verify the symmetry first, it's cheap (CPU-only,
minutes) compared to another 3-hour GPU training run that might fail the same way.

User went to sleep (2026-07-26 ~03:00) and explicitly asked to run autonomous training + self-verification on `~/Desktop/open_duck_mini_isaaclab` (lab PC copy: `/media/do/Extreme SSD/parksuho/open_duck_mini_isaaclab`, SSH `do@192.168.137.111` pw `433`), the same way the TD3/bipedalwalker project's autonomous loop worked ([[project_autonomous_experiment_branching]] is that other project's memory — different codebase, same "user is away, keep iterating and verifying yourself" pattern).

**State at handoff**: Just fixed a real bug — `_get_dones()` in `source/open_duck_mini_isaaclab/tasks/velocity/joystick_env.py` only terminated on being fully flipped over (`projected_gravity_b.z > 0`, ported faithfully from Playground's `upvector_z < 0`), never on collapsing-without-inverting. A full 3000-iteration Stage-1 run (`use_imitation=False`) converged on exactly that degenerate policy: reward 354, episode length 902/1000 looked great, but the robot was visibly in a collapsed heap over WebRTC — it had learned to farm `alive_scale=20.0` by never fully flipping, not to walk. Added `min_base_height_ratio=0.6` config + a base-height termination check (terminate if root height < 60% of `HOME_BASE_HEIGHT=0.15m`) to close that hole. Smoke-tested (3 iters, 16 envs) — episode length back to a sane ~20-24 (was previously inflated). Committed on both Mac (`d20cdba`) and lab PC (`75f2101`).

First retrain attempt at `--num_envs 256` (`~/train_full2.log`) was killed mid-run (~iter 1547/3000, reward 194 trending up normally) and **restarted at `--num_envs 2048`** (`~/train_full3.log`, started ~03:15) per the user's explicit "use max compute resources" request — checked `nvidia-smi` first: 256 envs only used 3.1GB/16.3GB VRAM and 53% GPU util, so there was large headroom. At 2048 envs: 4.5GB VRAM (27%), 73% GPU util, 23.7k steps/s (vs 5.1k at 256). Same `max_iterations=3000` was kept (not reduced), so total env-steps this run collects is ~147M vs the earlier ~18.4M-step runs — ETA is correspondingly longer, ~2 hours instead of ~1 hour, but the resulting policy sees far more experience per gradient update. **The current/authoritative run to check is `~/train_full2.log`'s successor `~/train_full3.log`, num_envs=2048** — ignore `train_full2.log`, it was intentionally superseded, not crashed.

**Why**: To verify the fix actually produces real standing/walking behavior instead of another reward-hacked collapse, and because the user explicitly wants this iterated on autonomously overnight rather than waiting for a single run.

**Backup point**: User asked for a safety net before further unattended changes. Tagged the current commit as `backup-2026-07-26-pre-autonomous` on BOTH the Mac repo (at `9894ad7`) and the lab-PC repo (at `2f35d66`) — `git checkout backup-2026-07-26-pre-autonomous` on either machine returns to this known-good point (termination fix in place, stability-check scripts committed) if anything done after this point turns out broken. User's plan: check back tomorrow; if the autonomous work went well, keep using it, otherwise revert to this tag first.

Also extended `scripts/eval_policy_stability.py` (DONE, committed Mac `288a61e` / lab-PC `2622dc0`) per user feedback: standing upright isn't proof of walking either — a policy can reward-hack by standing still. Added three more checks, all printed by the script plus a "WALKING RESULT" verdict line:
  - lin-vel tracking error (achieved `root_lin_vel_b.xy` vs commanded, only counting steps where |command|>0.02)
  - mean leg-joint range of motion across the 10 leg joints (near-zero = legs not swinging)
  - mean foot-contact toggle count per foot (0 = that foot never left the ground = no stepping)
  - verdict threshold used: `mean_leg_rom > 0.15 rad AND mean_vel_err < 0.15 m/s AND mean_toggles_per_foot > 4` over the eval window → "LIKELY ACTUALLY WALKING", else "LIKELY REWARD-HACKING". These thresholds are a first guess, not validated against a known-good gait yet — sanity-check them against the printed raw numbers, don't trust the verdict line blindly the first time it runs.
This script has STILL never been executed end-to-end (only `py_compile` syntax-checked) — the next wakeup after training finishes needs to actually run it and debug real runtime errors if any (e.g. `unwrapped._command` / `_contact_sensor` / `_feet_ids` attribute access, `ACT_LEG_JOINT_IDX` indexing into `joint_ids`).

Training status as of this note: iteration 183/3000 at 2048 envs, reward 37.67 climbing normally, ETA ~1h33m from a 03:50 start.

**UPDATE — Run 2 finished and also failed (2026-07-26 ~05:35)**: The num_envs=2048/alive=20 run completed 3000 iterations (reward ~320-340, episode length ~830-930/1000) but the user directly watched it over WebRTC (checkpoint model_2600) and it was STILL visibly collapsed/contorted, not standing — the height-termination fix alone wasn't sufficient. Root-caused against Playground's original config: `alive_scale=20.0` is copied verbatim from upstream, which calibrates it assuming `reward_imitation`'s steep per-step penalty (`w_joint_pos=15.0`) counterbalances it. Stage 1 (`use_imitation=False`) has no such counterweight — alive alone is 70% of the ~570 theoretical per-episode reward ceiling (worked out with the user: `tracking_lin_vel(2.5) + tracking_ang_vel(6.0) + alive(20.0) = 28.5/step ceiling * dt(0.02) * 1000 steps = 570`). Full narrative now lives in `docs/training_log.md` (both machines) — **check that file first**, it's more detailed than this memory note and will keep being updated per-run.

**Now running an alive_scale sweep** (user's request, "like the TD3 parallel-variant pattern"): three gym tasks registered in `source/open_duck_mini_isaaclab/__init__.py` — `Isaac-OpenDuckMini-Joystick-v0` (alive_scale=2.0, now the base class default in `joystick_env_cfg.py`), `-Alive5-v0` (5.0), `-Alive10-v0` (10.0) — via `JoystickEnvCfg_Alive5`/`JoystickEnvCfg_Alive10` subclasses. Launched all three CONCURRENTLY on the one lab-PC GPU at `--num_envs 512` each (`--run_name alive2/alive5/alive10`, logs `~/train_alive2.log`/`~/train_alive5.log`/`~/train_alive10.log`) — confirmed running clean, GPU at 9.4GB/16GB (58%) and 93% compute util for all three combined, so there was no need to run them sequentially. Committed as `853235e` (Mac) / `dc450c4` (lab PC).

**How to apply from here**:
1. Check all three `~/train_alive{2,5,10}.log` files — they'll finish at different times since ETAs diverged a bit at launch (alive2 ~1h10m, alive5 ~2h05m, alive10 was still starting). Don't assume they finish together.
2. For each as it finishes, run `eval_policy_stability.sh` against its final checkpoint (look under `logs/rsl_rl/open_duck_mini_v2_joystick/*_alive2` / `*_alive5` / `*_alive10` run dirs — the `--run_name` suffix is appended to the timestamped dir name) and read BOTH the STANDING and WALKING result lines.
3. Record every result (pass or fail) in `docs/training_log.md`'s Run 3-5 table with root-cause analysis if it fails — the user explicitly asked for failures to be logged with analysis, not just successes.
4. If one of the three clearly wins (standing + walking, good leg ROM, good foot-contact alternation, good vel tracking), that's the value to adopt as the new default and report to the user. If none pass, the next lever is probably building out Stage 2 (imitation/reference-motion pipeline) rather than further alive_scale tweaking — flag that clearly rather than guessing a 4th value blindly.
5. Backup tag `backup-2026-07-26-pre-autonomous` still exists on both repos if anything needs reverting.

**How to apply** (if this session gets cut off and a fresh one picks up):
1. Check `~/train_full3.log` on the lab PC (`sshpass -p '433' ssh do@192.168.137.111 "tail -40 ~/train_full3.log"`) for current iteration/reward/episode-length. This is `--num_envs 2048`, ETA ~2hr from ~03:15 start — `train_full2.log` (num_envs=256) was intentionally killed and superseded, not a crash.
2. Don't trust reward/episode-length numbers alone as "success" — that's exactly what broke last time. Verify with `scripts/check_joint_stability.sh` (zero-action PD sanity check, already built) AND ideally an actual trained-policy rollout check (base height should stay near 0.15m, `projected_gravity_b.z` near -1, not pinned near the 0.09m termination floor) before declaring it fixed.
3. WebRTC visual check remains the most reliable verification (`nohup ~/isaacsim/isaac-sim.streaming.sh --/app/livestream/publicEndpointAddress=192.168.137.111 &` for a generic view, or `play.sh --checkpoint <path> --livestream 1 --real-time` for the actual policy) — see `docs/webrtc_streaming.md`. Always kill any prior streaming/play session first (`pkill -9 -f play.py` or `pkill -f isaacsim.exp.full.streaming`) — leftover sessions have twice caused port 8011 conflicts this session.
4. If the fix still doesn't produce real standing, the next lever is probably reward shaping (reduce `alive_scale` relative to tracking terms, or add an explicit upright/height-tracking reward term) rather than another termination tweak — `use_imitation=True` (Stage 2) is the "real" fix but needs the reference-motion generator pipeline set up first, which hasn't been done yet.
5. Keep syncing every change to both the Mac and lab-PC copies (scp + separate git commit on each — no shared remote between them) per the established workflow this whole session.
