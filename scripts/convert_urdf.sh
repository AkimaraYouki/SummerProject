#!/usr/bin/env bash
# Ubuntu-only: requires an Isaac Sim / Isaac Lab install.
#
# Converts robot/robot.urdf (generated via onshape-to-robot — see
# docs/onshape_import.md) into assets/usd/open_duck_mini_v2.usd using
# IsaacLab's UrdfConverter.
#
# --joint-damping is the Dynamixel XM430's measured viscous friction
# (friction_viscous, "m4" model) from the BAM actuator characterization at
# ~/Desktop/robot make/bam_xm430_params/m4.json (0.8470782260272692, rounded
# below). armature is also directly measured there (0.014317492733137276)
# but NOT wired in yet — only damping was requested so far.
#
# --joint-stiffness is still the old STS3215 placeholder (13.37) — BAM's
# friction model doesn't produce a position-control stiffness/Kp value (that's
# a control-loop tuning choice, not a measurable actuator property), so this
# still needs to be set deliberately, not sourced from BAM. Must stay in sync
# with source/open_duck_mini_isaaclab/robot_cfg.py — see docs/decisions.md.
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
    --joint-damping 0.847 \
    --joint-target-type position \
    "$@"
# extra args (e.g. --headless if running without a display) are forwarded via "$@"

echo "Wrote $OUTPUT_USD"
echo "NOTE: --fix-base is intentionally omitted (defaults to false / floating base for this biped)."
echo "Verify articulation root + joint/body names in Isaac Sim before proceeding to Stage 1's robot_cfg.py."
