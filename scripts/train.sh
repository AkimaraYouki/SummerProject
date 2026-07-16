#!/usr/bin/env bash
# Ubuntu-only: thin wrapper around IsaacLab's own (maintained) rsl_rl train
# script — deliberately NOT reimplemented in this repo (see docs/decisions.md:
# duplicating IsaacLab's AppLauncher/Hydra boilerplate would just be a worse,
# unmaintained copy of a script IsaacLab already ships and updates).
#
# Usage:
#   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/train.sh [--num_envs 256] [extra rsl_rl/train.py args...]
set -euo pipefail

if [ -z "${ISAACLAB_PATH:-}" ]; then
    echo "ISAACLAB_PATH is not set. Point it at your IsaacLab checkout." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Make sure this repo's extension package is importable (pip install -e .
# should already have been run once; this is a safety net for fresh shells).
python3 -c "import open_duck_mini_isaaclab" 2>/dev/null || (cd "$REPO_ROOT" && pip install -e .)

"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task Isaac-OpenDuckMini-Joystick-v0 \
    "$@"
