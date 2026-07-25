#!/usr/bin/env bash
# Ubuntu-only: requires an Isaac Sim / Isaac Lab install.
#
# Converts robot/robot.urdf (generated via onshape-to-robot — see
# docs/onshape_import.md) into assets/usd/open_duck_mini_v2.usd using
# IsaacLab's UrdfConverter, with joint drive settings matching the actuator
# gains documented in docs/decisions.md (must stay in sync with
# source/open_duck_mini_isaaclab/robot_cfg.py).
#
# Usage:
#   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/convert_urdf.sh
set -euo pipefail

if [ -z "${ISAACLAB_PATH:-}" ]; then
    echo "ISAACLAB_PATH is not set. Point it at your IsaacLab checkout, e.g.:" >&2
    echo "  ISAACLAB_PATH=/home/\$USER/IsaacLab ./scripts/convert_urdf.sh" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URDF_PATH="$REPO_ROOT/robot/robot.urdf"
OUTPUT_USD="$REPO_ROOT/assets/usd/open_duck_mini_v2.usd"

mkdir -p "$(dirname "$OUTPUT_USD")"

"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/tools/convert_urdf.py" \
    "$URDF_PATH" \
    "$OUTPUT_USD" \
    --merge-joints \
    --joint-stiffness 13.37 \
    --joint-damping 0.56 \
    --joint-target-type position \
    "$@"
# extra args (e.g. --headless if running without a display) are forwarded via "$@"

echo "Wrote $OUTPUT_USD"
echo "NOTE: --fix-base is intentionally omitted (defaults to false / floating base for this biped)."
echo "Verify articulation root + joint/body names in Isaac Sim before proceeding to Stage 1's robot_cfg.py."
