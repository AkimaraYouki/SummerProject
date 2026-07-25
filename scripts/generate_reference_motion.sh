#!/usr/bin/env bash
# Ubuntu-only in practice (confirmed 2026-07-26): the required `placo==0.6.3`
# pin (see pyproject.toml's reference-motion extra) has no macOS wheel on
# PyPI, only manylinux. Pure CPU kinematics though — no Isaac Sim/GPU needed,
# so this can run on the lab PC's CPU alongside a GPU training job with no
# contention. Also needs a URDF with Placo's expected frame names
# (trunk/left_foot/right_foot/head) — run scripts/patch_urdf_for_placo.py
# first if robot/robot.urdf changed since the last patch.
#
# Sweeps (dx, dy, dtheta) gait presets for open_duck_mini_v2 via Placo, then
# fits degree-15 polynomials per dimension, producing
# source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_DIR="$REPO_ROOT/reference_motion_generator"
RECORDINGS_DIR="$GEN_DIR/recordings"
OUTPUT_PKL="$REPO_ROOT/source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl"

python3 -c "import placo" 2>/dev/null || {
    echo "placo is not importable in this Python environment." >&2
    echo "Install it (pip install -e '.[reference-motion]') or run this script on Ubuntu instead." >&2
    exit 1
}

echo "=== Sweeping gait presets (this can take a while; use -j<N> inside auto_waddle.py for parallelism) ==="
python3 "$GEN_DIR/scripts/auto_waddle.py" \
    -j4 \
    --duck open_duck_mini_v2 \
    --sweep \
    --output_dir "$RECORDINGS_DIR"

echo "=== Fitting polynomials ==="
mkdir -p "$(dirname "$OUTPUT_PKL")"
# fit_poly.py writes polynomial_coefficients.pkl to the current working
# directory (see reference_motion_generator/scripts/fit_poly.py), so cd into
# a known directory before running it, then move the output into place.
(cd "$REPO_ROOT" && python3 "$GEN_DIR/scripts/fit_poly.py" --ref_motion "$RECORDINGS_DIR")
mv "$REPO_ROOT/polynomial_coefficients.pkl" "$OUTPUT_PKL"

echo "Wrote $OUTPUT_PKL"
