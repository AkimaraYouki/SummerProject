#!/usr/bin/env bash
# Ubuntu-only: requires an Isaac Sim / Isaac Lab install.
#
# Converts robot/robot.urdf (generated via onshape-to-robot — see
# docs/onshape_import.md) into robot/usd/open_duck_mini_v2.usd using
# IsaacLab's UrdfConverter. Output lives under robot/ alongside the URDF and
# meshes so everything from a given OnShape import stays in one place.
#
# --joint-stiffness / --joint-damping are both derived from the "m4"-model
# BAM actuator characterization at
# ~/Desktop/robot make/bam_xm430_params/m4.json, via BAM's own
# VoltageControlledActuator.to_mujoco() conversion (bam/bam/actuator.py) —
# NOT a raw BAM field, a formula that turns BAM's fitted kt/R/friction_viscous
# plus the servo's real Position P Gain register setting into physical PD
# gains:
#   stiffness = (1/128) * 800.0(Position P Gain) * 12.0V * kt/R  = 37.65
#   damping   = friction_viscous + kt**2/R                       = 1.352
# (rounded; see source/open_duck_mini_isaaclab/robot_cfg.py's module
# docstring for the full unrounded derivation and each term's source).
# armature/friction/effort_limit_sim/velocity_limit_sim are ALSO set (BAM +
# XM430-W350 datasheet @ confirmed 12.0V) but live in robot_cfg.py, not as
# CLI flags here — this converter only exposes stiffness/damping. Must stay
# in sync with robot_cfg.py — see docs/decisions.md.
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
OUTPUT_USD="$REPO_ROOT/robot/usd/open_duck_mini_v2.usd"

mkdir -p "$(dirname "$OUTPUT_USD")"

"$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/scripts/tools/convert_urdf.py" \
    "$URDF_PATH" \
    "$OUTPUT_USD" \
    --merge-joints \
    --joint-stiffness 37.65 \
    --joint-damping 1.352 \
    --joint-target-type position \
    "$@"
# extra args (e.g. --headless if running without a display) are forwarded via "$@"

echo "Wrote $OUTPUT_USD"
echo "NOTE: --fix-base is intentionally omitted (defaults to false / floating base for this biped)."
echo "Joint/body names cross-checked against joint_order.py and confirmed via a working"
echo "train.sh smoke test (2026-07-25) — re-verify only if robot/robot.urdf changes."
echo "STILL OPEN: HOME_JOINT_POS has not been visually confirmed as a symmetric standing"
echo "pose in Isaac Sim — check this next if training behaves oddly."
