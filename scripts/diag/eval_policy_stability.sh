#!/usr/bin/env bash
# Ubuntu-only. Rolls out a trained checkpoint with its real actions and
# reports base height/upright — see eval_policy_stability.py's module
# docstring for why (reward/episode-length alone were misleading once).
#
# Usage:
#   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/eval_policy_stability.sh --checkpoint /path/to/model_XXXX.pt [--headless] [--num_envs 8] [--num_steps 500]
set -euo pipefail

if [ -z "${ISAACLAB_PATH:-}" ]; then
    echo "ISAACLAB_PATH is not set. Point it at your IsaacLab checkout." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$ISAACLAB_PATH/isaaclab.sh" -p -c "import open_duck_mini_isaaclab" 2>/dev/null \
    || (cd "$REPO_ROOT" && "$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e .)

"$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/_isaaclab_launch.py" \
    "$REPO_ROOT/scripts/diag/eval_policy_stability.py" \
    "$@"
