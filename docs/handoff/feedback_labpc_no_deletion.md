---
name: labpc-no-deletion
description: "The lab PC reached via SSH (do@192.168.137.111) is a shared machine, not the user's own — never delete/cleanup anything there, only add files within the Claude project folder"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fa1e1079-ee0d-4474-bed8-1f5019bb6dfd
---

On the lab PC (SSH `do@192.168.137.111`, project dir `/home/do/Pictures/Claude/`),
never run deletion/cleanup commands (`rm`, `rmdir`, clearing old checkpoints/logs,
"tidying up"), even when asked to "정리해줘" (clean up) in a broader instruction.
Only create/add new files, and only inside `/home/do/Pictures/Claude/`.

**Why:** User stopped me mid-command when I was about to `rmdir` an empty stray
checkpoint folder there: "이거 내컴 아냐 위험한거 하지마 제발... Claude 폴더 내에서만 놀아"
— it's not their personal machine (shared lab resource), and even a seemingly-safe
deletion (empty dir) triggered real alarm. The instinct here should be much more
conservative than on their own local Mac.

**How to apply:**
- If a request says "필요없는 파일 정리하고" or similar, do NOT delete anything —
  either skip that part silently-but-flag it, or explicitly ask before any rm/rmdir
  on this host.
- Stay scoped to `/home/do/Pictures/Claude/` for all writes (new checkpoints,
  new script copies, new log files) — don't touch paths outside it.
- Disk space concerns on this host (was 93% full, 62GB free) should be reported
  to the user, not resolved unilaterally by deleting their files.
- This caution is specific to the lab PC / remote SSH context — distinct from
  [[no-local-training-runs]] which is about not training on the local Mac.
