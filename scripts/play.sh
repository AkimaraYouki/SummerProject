#!/usr/bin/env bash
# Ubuntu-only: thin wrapper around IsaacLab's own rsl_rl play script.
#
# Running this ALSO exports the trained policy to ONNX automatically —
# IsaacLab's play.py calls `runner.export_policy_to_onnx(...)` internally
# on every run, writing <checkpoint_dir>/exported/policy.onnx. There is no
# separate export_onnx.py in this repo for that reason (see docs/decisions.md).
#
# Usage:
#   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/play.sh --checkpoint /path/to/model.pt
set -euo pipefail

if [ -z "${ISAACLAB_PATH:-}" ]; then
    echo "ISAACLAB_PATH is not set. Point it at your IsaacLab checkout." >&2
    exit 1
fi

"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/play.py" \
    --task Isaac-OpenDuckMini-Joystick-v0 \
    "$@"

echo "If successful, look for exported/policy.onnx next to your checkpoint."
echo "Verify it on this Mac with: ./scripts/mujoco_infer.py --onnx <path-to-policy.onnx>"
