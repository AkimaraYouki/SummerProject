#!/usr/bin/env bash
# Ubuntu-only. Zero-action PD-gain sanity check — see
# check_joint_stability.py's module docstring for what this actually tests
# and why (not an RL script; run after any stiffness/damping/armature change
# to robot_cfg.py before trusting a full training run).
#
# Usage:
#   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/check_joint_stability.sh [--headless] [--num_envs N] [--num_steps N]
set -euo pipefail

if [ -z "${ISAACLAB_PATH:-}" ]; then
    echo "ISAACLAB_PATH is not set. Point it at your IsaacLab checkout." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ISAACLAB_PATH/isaaclab.sh" -p -c "import open_duck_mini_isaaclab" 2>/dev/null \
    || (cd "$REPO_ROOT" && "$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e .)

"$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/_isaaclab_launch.py" \
    "$REPO_ROOT/scripts/check_joint_stability.py" \
    "$@"
