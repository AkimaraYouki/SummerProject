---
name: graph-based-verification
description: "User cannot watch WebRTC (away from lab / mobile) — verify OpenDuck policies with graphs, always against the same fixed baseline conditions"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-28T19:24:20.929Z
---

When the user is away from the lab, WebRTC is unavailable and **graphs are the only
verification channel**. They asked for this explicitly and reconfirmed it.

**Why:** reward curves cannot distinguish walking from trembling — that mistake cost this
project several runs ([[feedback_verify_training_visually]] required visual confirmation, and
when video is impossible, quantified plots are the substitute, not reward numbers).

**How to apply:** two committed tools, always run with identical settings so runs are
comparable:

- `scripts/joint_periodicity.py` + `scripts/analysis/plot_periodicity.py` — autocorrelation at
  the gait period + power spectrum. Separates periodic gait from broadband jitter.
- `scripts/gait_compare.py` + `scripts/analysis/plot_gait_compare.py` — five pinned commands
  (forward +0.15 / backward −0.15 / left +0.2 / right −0.2 / turn yaw +1.0, all inside the
  reference polynomial's fitted range) × joint position, joint velocity, base velocity vs
  command, and foot-contact pattern.

Both need the GPU, so pause training first (two Isaac Sim instances kill one another), wait for
the next 100-iteration checkpoint so little is lost, then resume with
`--resume --load_run <dir> --checkpoint model_N.pt`. Note `--resume True` fails — Hydra parses
the `True` as a malformed override; the flag takes no value.

**Baseline to compare against (imitation_v12 @ iter 400):** forward 0.097 m/s vs 0.15 command
(65%), backward −0.080, left 0.005 / right −0.034 against ±0.2 (lateral essentially absent),
joint RMS 7–10°, contact toggles 144–319/10s, knee autocorrelation 0.77 at the 27-step gait
period, dominant frequency 1.80 Hz vs the gait's 1.85 Hz.

Send figures with SendUserFile — the user reads them on mobile.

**Visualization is forward-command-only, always.** The user asked for this explicitly
("시각화는 무조건 전진으로만"). `play.py` drives the env's own random command sampler, so the
robot keeps switching between forward/backward/lateral/turn and you cannot tell which target it
is reacting to. Use `scripts/play_fixed_cmd.py --cmd_x 0.15 --cmd_y 0 --cmd_yaw 0 --livestream 2`
instead; it pins the command and prints achieved `vx`/`vy` every 250 steps, which is what
exposed v16 drifting sideways at ±0.3 m/s under a pure forward command.
