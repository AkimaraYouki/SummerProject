---
name: feedback-verify-training-visually
description: "Never call an RL training run 'healthy'/'successful' from reward/episode-length/eval-script numbers alone — visually confirm via WebRTC/play.py first"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-27T03:50:01.331Z
---

Don't characterize an RL policy as "healthy," "improved," or "successful" based on aggregate
numeric proxies alone (TensorBoard reward, episode length, action-noise-std, or even
`eval_policy_stability.sh`'s STANDING/WALKING verdict) — visually confirm the actual rollout
(WebRTC `play.py --livestream 2 --checkpoint ... --real-time`, or MeshCat/replay tooling) before
declaring success.

**Why:** In [[project_openduck_autonomous_training]], the `alive_scale`×`w_joint_pos` sweep
called `A20J5` "dramatically healthier" than the other candidates purely from reward/episode
length/toggle-count numbers. The user then watched the actual policy over WebRTC and reported
"관절이 특정각도에 고착되면서 팡 하고 튕겨나감" (joint locks near a limit angle then pops/flings
outward, terminates) — a local-minimum failure mode the numeric proxies completely missed. The
user's direct correction: "건강? 실제로 걷는 모션도 없고 정책이 로컬 미니멈에 빠지는걸 그 건강한다고
하던 정책을 시각화 해서 확인했는데" — i.e. the numeric "healthy" framing was actively misleading
once actually watched.

**How to apply:** Before reporting any training run as PASS/successful/healthy to the user:
1. If `eval_policy_stability.sh` (or any numeric proxy) says FAIL, that's sufficient to report
   FAIL — no need to visualize a known failure.
2. If numeric proxies say PASS/healthy, do NOT report success yet. Load the checkpoint into a
   live view (`play.py --livestream 2 --checkpoint <ckpt> --num_envs 1 --real-time`, kill any
   other Isaac Sim instance on the GPU first — they crash each other) and either watch it
   yourself or ask the user to confirm what they see before using words like "성공"/"walks well."
3. When relaying results mid-training or in scheduled autonomous checks, hedge explicitly:
   "수치는 좋지만 육안 확인 필요" rather than declaring victory from numbers alone.
