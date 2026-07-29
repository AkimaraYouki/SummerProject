---
name: no-local-training-runs
description: "In the BipedalWalker RL project, don't launch training runs on this local machine — user runs all training themselves via SSH on their lab computer"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fa1e1079-ee0d-4474-bed8-1f5019bb6dfd
---

Don't launch (nohup/background) TD3/SAC training processes locally in this project.
Only give direction: hyperparameter/config recommendations, exact commands to run,
rationale for changes, and analysis of results the user brings back.

**Why:** User said "이제부터 여긴 학습 방향성만 제시해줘 모든 학습은 내가 연구실 ssh 로
돌릴거임" — from now on all actual training happens on their lab machine over SSH;
this local session is for guidance, code changes, and analysis only.

**How to apply:**
- Still fine to: edit `orinaltonew/Bipedalwalker_TD3.py`, propose/explain hyperparameter
  changes, write out the exact `--set ...` command for them to run, analyze logs/checkpoints/
  reward histories they bring back, build report artifacts.
- Don't: run `python3 ... Bipedalwalker_TD3.py` (train mode) via Bash in the background
  on this local Mac anymore. `--play` (quick local viewing) wasn't explicitly banned but lean
  toward asking first now that the workflow has shifted to remote training.
- If unsure whether a request means "run it here" vs "give me the command," default to
  giving the command only.

**Update:** user later gave SSH access to their lab PC (`do@192.168.137.111`) and
explicitly directed launching/managing training runs there ("너가 다해 ㄱㄱ"). That's a
distinct, separately-authorized context — SSH-driven execution on the lab PC is fine
when asked. This memory is specifically about not training on the local Mac session.
See [[labpc-no-deletion]] for the safety rule that applies on that lab PC.
