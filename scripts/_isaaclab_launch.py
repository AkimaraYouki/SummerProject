#!/usr/bin/env python3
"""Shim: import this package (so its gym.register() runs) before delegating
to an IsaacLab script.

IsaacLab's own train.py/play.py only `import isaaclab_tasks` — they have no
mechanism for discovering external task packages like
open_duck_mini_isaaclab, so `--task Isaac-OpenDuckMini-Joystick-v0` fails
with `gymnasium.error.NameNotFound` unless something imports this package
first, in the SAME process, before gym.spec() is called. A separate
`isaaclab.sh -p -c "import open_duck_mini_isaaclab"` pre-check does NOT
work for this — each `isaaclab.sh -p` invocation is its own process, and
gym's registry is in-memory per-process (confirmed by hitting exactly this
failure during the 2026-07-25 smoke test).

Also inserts the target script's own directory onto sys.path — IsaacLab's
train.py does `import cli_args` (a sibling file), which only resolves if
that directory is on sys.path the way it would be when run directly as
`python train.py` (confirmed needed: without this, the delegated script
fails with `ModuleNotFoundError: No module named 'cli_args'`).

Usage: _isaaclab_launch.py <isaaclab_script.py> [args-for-that-script...]
"""

import os
import runpy
import sys

import open_duck_mini_isaaclab.tasks  # noqa: F401 - side effect: gym.register()

target = sys.argv.pop(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
