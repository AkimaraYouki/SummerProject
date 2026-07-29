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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Make sure this repo's extension package is importable IN THE SAME PYTHON
# THAT WILL ACTUALLY RUN TRAINING BELOW. Must go through isaaclab.sh -p for
# both the check and the install — a plain `python3`/`pip` here silently
# checks/installs into a *different* interpreter (system Python, or whatever
# conda env is active) than the IsaacLab-bundled one `isaaclab.sh -p` uses to
# launch train.py, so the check could pass while training still fails with
# ModuleNotFoundError. (Confirmed on the lab PC 2026-07-25: package was
# missing from both system python3 AND the bundled interpreter.)
"$ISAACLAB_PATH/isaaclab.sh" -p -c "import open_duck_mini_isaaclab" 2>/dev/null \
    || (cd "$REPO_ROOT" && "$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e .)

# Delegate through _isaaclab_launch.py rather than calling train.py directly
# — IsaacLab's train.py never imports this repo's package, so
# --task Isaac-OpenDuckMini-Joystick-v0 would fail with
# gymnasium.error.NameNotFound otherwise (confirmed 2026-07-25 smoke test;
# see _isaaclab_launch.py's docstring for why the -m pip check above doesn't
# already cover this).
"$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/scripts/_isaaclab_launch.py" \
    "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task Isaac-OpenDuckMini-Joystick-v0 \
    "$@"
