# Open Duck Mini V2 on Isaac Lab

A port of the Open Duck Mini V2 stack to **Isaac Lab** (PyTorch + PhysX) instead of
MJX/Brax. It pulls together what lives in three separate upstream repos:

- [Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) for the robot itself
- [Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) for the RL setup and rewards
- [Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) for the placo reference gait

Everything from CAD to the real robot is in here, so it's a working repo. Some of it
is tidy, some of it isn't.

```
 Onshape CAD ──▶ URDF + meshes ──▶ USD ──▶ RL ──▶ policy ──▶ real robot
                      │                    ▲
                      └──▶ placo gait ─────┘
                           (imitation target)
```

The robot is a 2.74 kg biped with 14 Dynamixel XM430-W350 servos, driven by a Jetson
Orin Nano over a single U2D2 bus.

## Where this is at (August 2026)

It walks. Has done since 2026-08-13, when it turned out that the thing making it fall
forward for days was not the policy, not the gains, not the left knee. It was the
power supply. A USB-PD adapter was capping the leg bus at 5.22 A, the rail sagged to
8.6 V, and XM430s stop pulling current below 9.5 V. Swapping the supply and changing
nothing else took the minimum bus voltage to 11.2 V, peak leg current to 10.12 A, and
left-knee tracking error from +22.9° to +2.7°.

> The 5.22 A was the adapter's ceiling, not the robot's demand. Measuring at the load
> and concluding "supply is fine" is how I lost those days.

So the question now isn't whether it walks, it's *how* it walks. Two things are wrong:
peak torque is pinned against the actuator limit most of the time, and the torso rocks
side to side.

### What I'm optimising for now

Up until last week I ranked six-direction tracking error first and stability second.
That's changed. The order now is:

1. Peak torque
2. Walking efficiency
3. Torso motion, specifically the side-to-side rocking
4. Stability and robustness
5. Command tracking, which now only has to be *directionally correct*. Slow is fine.

This matters more than it sounds. Policies I threw away for bad tracking are the best
ones under the new ranking, so I've been re-reading old runs rather than training new
ones.

### Scoring

`scripts/diag/scoreboard.py` grades every measured policy. I added three columns when
the goals changed, because you can't chase something you don't measure:

- **sat%** is the fraction of time `|τ|` sits at the effective limit of 3.16 N·m
- **W** is mean `|τ·ω|` summed over the ten leg joints
- **roll rate** is the RMS of roll angular velocity, in deg/s. This is the rocking.

Peak torque on its own is useless as a metric: all seven policies I checked have a p99
of exactly 3.16, because they all bottom out against the clamp. How *often* they hit it
is the real difference.

| ver | score | sat% | W | roll rate | roll RMS | falls% | what it is |
|---|---|---|---|---|---|---|---|
| v61 | 0.0221 | 1.21 | 9.6 | 75.8 | 4.97 | 0.7 | best one on hardware |
| v65 | 0.0138 | 1.05 | 9.9 | 87.5 | 4.89 | 1.9 | best sim score, July |
| v73 | 0.0269 | 1.18 | 10.2 | 79.6 | 4.71 | 1.2 | big foot + CoM −10 mm |
| v74 | 0.0140 | 0.31 | 7.3 | 80.0 | 5.05 | 1.1 | v73 + torque −0.2, fall term 0.85, σ 1.3 |
| v75 @2800 | 0.0275 | 0.20 | 6.6 | 46.2 | 2.85 | 0.2 | v74 + `torso_ang_vel -0.7` |
| **v75 @11800** | **0.0170** | **0.20** | **6.4** | **31.0** | **2.25** | **0.0** | same thing, trained 4× longer |
| v77 @3000 | 0.0265 | 0.29 | 7.1 | 53.0 | 3.11 | 0.3 | v75 + wider friction/stiffness |

Two things did the work, and they don't overlap. The torque penalty bundle (v74) took
saturation from 1.18% to 0.31% and power from 10.2 W to 7.3 W without touching the
rocking. The torso angular velocity penalty (v75) took the rocking from 80.0 to 46.2
and falls from 1.1% to 0.2%.

### Training longer mattered, and the reward curve hid it

Then I ran v75 out to 11,800 iterations instead of 3,000. Reward had been flat since
about 3,700 (+1 to +7 per thousand iterations, down from +44 early), so I called it
converged and was wrong:

    roll rate   46.2 → 31.0    (−33%)
    roll RMS    2.85 → 2.25    (−21%)
    falls       0.2% → 0.0%
    forward     0.0228 → 0.0092  (−60%)
    turning     0.0320 → 0.0148  (−54%)
    sideways    0.0370 → 0.0429  (+16%, the only regression)

Reward missed all of it. The two biggest terms (`imitation` at 23%, `path_tracking` at
32%) had already saturated, and the remaining gains were coming from small penalty
terms worth 0.01–0.03 each against a total of 450. **Judge convergence on the target
metrics, not on reward.** Torque and power don't improve with longer training; those
came entirely from the reward design in v74/v75.

### Low-pass filtering the actions: useful, until it isn't

`ODM_LPF` filters the action with `filt = α·filt + (1−α)·new`, so higher α is heavier
smoothing. On the 2,800-iteration policy, α=0.3 looked free — saturation, power, roll
rate and falls all improved at once, costing only a little tracking. α=0.5 and 0.7 made
things worse, and the boundary lines up with physics: the gait runs at 1.85 Hz, and
α=0.3 cuts at 9.6 Hz (shaves jitter) while α=0.7 cuts at 2.8 Hz (shaves the gait
itself). The torque savings at 0.7 came from the robot moving less, which is also why
its tracking collapsed and its falls quadrupled.

But re-measuring on the long-trained policy, the filter is now a net loss:

| v75 @11800 | no filter | α=0.3 |
|---|---|---|
| roll rate | **31.0** | 35.7 |
| turning | **0.0148** | 0.0352 |
| saturation | 0.20% | 0.14% |
| power | 6.4 W | 5.9 W |

The undertrained policy was shaking and the filter cleaned it up. The trained one
isn't, so the filter only removes signal. Hardware default is no filter; α=0.3 stays
in the pocket in case the real robot shakes in ways the sim doesn't.

### Sim rank is not hardware rank

Worth stating plainly because it burned me. v59 has the better sim score (0.0165 vs
0.0221) and v61 walks better on the real robot. The two are basically tied on forward
and turning and only differ in sideways tracking, but v61 shakes 25% less. On hardware
a fall ends the run, so 25% less shake beats 0.003 of tracking error every time.

Use the sim score to narrow the field. Pick the winner on hardware.

### Things I was wrong about

Every one of these came from estimating a number instead of measuring it.

**The foot.** I spent six training runs trying to fix roll instability with rewards,
then measured the foot and found the contact patch is 16.3 mm wide. Great, mechanical
limit, case closed. Except that's the TPU sole measured on its own, and in simulation
the lowest collision surface on the foot link isn't the sole at all, it's
`l_foot_side`, sitting 2.70 mm below it. Sim has been standing on a 99 × 39 mm plate
the whole time. The knife edge I was chasing never existed in the simulator, which
also means v73 (the big-foot export) only widened sim support by 11% and couldn't
really test the hypothesis.

**Foot indices.** `_feet_ids` indexes the contact sensor's body list. I used it on
`_robot.data.*`, which is the articulation, and the two orders differ. Index 5 is
`foot_assembly` on the sensor and `head_pitch_assembly` on the articulation. Five foot
rewards were reading the head. Invalidated five runs. `tests/test_index_domains.py`
now blocks it statically.

**"The feet are chattering at 7 Hz."** They weren't. With 40 ms debouncing every
policy sits at a 540 ms gait period. What I was looking at was landing bounce. I had
already told myself not to deploy a policy over this.

**Foot slip.** Measured μ > 0.38. Friction was never the problem.

**A 90 g / 26 mm discrepancy between the USD and the docs.** There isn't one. The USD
is 2.7430 kg with CoM y at +0.2 mm, matching the URDF exactly. I had quoted a stale
docstring instead of opening the file.

**Gyro bias.** Estimating it at startup, during ramp-in, gave −0.0203 rad/s, which
integrated into 41.7° of phantom yaw over a 35.9 s run. Removed the estimator; the
measured bias is ~0 anyway.

Out of that came one rule that has paid for itself: **when you add a reward term,
measure its raw magnitude before training, and check it's zero on the stop command.**
`scripts/diag/probe_terms.py` does this. A term that's nonzero at standstill is a
permanent tax on the reward budget, and I lost a run (v53, reward 318.8 → 100.4)
finding that out.

### Still open

Falls have never gone below 0.5%. v55 gets to 0.0 but pays for it in tracking.

The `path_error` observation was constant on hardware while making up 32% of the
reward, so the policy was being paid for something it couldn't see. Parking that until
I have VLA or optical flow, or until I just mix heading into the stick.

Whether the real foot contacts through the TPU pad or through `l_foot_side` decides
the next experiment. If the pad is proud of the shell, hardware support is 16 mm while
sim thinks it's 38.7 mm, and a 2.4× mismatch in lateral support would explain a lot
about why sim policies look steadier than they are. This is a five-second look at the
underside of the foot and I haven't done it yet.

Also unmeasured: real robot mass and CoM. And there's a bus timeout at 13.9 s that
isn't voltage (11.0 V minimum), so probably a connector.

## Driving it remotely

```bash
# on the workstation: video out over WebRTC, commands in over UDP
ODM_HOST_IP=<workstation ip> ./scripts/odm play v74 --stream --key
#   http://<ip>:8011/streaming/webrtc-client
#   signalling 49100, media 47998, page 8011

# on the laptop: curses dashboard, WASD/QE to steer, Z/X throttle, space to stop
python3 scripts/console.py --host <workstation ip>
```

Tested over an external network and both the steering and the training monitor hold up.

> `--stream` never actually worked before 2026-08-20. Isaac Lab takes `--kit_args` as
> a single string, and if you pass it with a space the argparse call chokes on a value
> starting with `--`. It has to be `--kit_args=--/app/...`.

## Getting started

Nearly everything goes through `scripts/odm`. Symlink it, don't copy it, or your copy
drifts from the repo:

```bash
ln -s "$PWD/scripts/odm" ~/bin/odm
```

| command | what it does |
|---|---|
| `odm train [ver] [iters] [envs]` | start training, defaults 3000 iters / 4096 envs |
| `odm watch [ver]` | one-line progress |
| `odm tb` | tensorboard over all runs |
| `odm play [ver]` | native window. `--joystick`, `--key`, `--stream` |
| `odm record [ver] [secs]` | six-direction cycle to mp4. seconds are *sim* seconds |
| `odm measure [ver] [iter]` | six-direction tracking, gait period, reward terms |
| `odm test` | test suite, no Isaac Sim needed |
| `odm import` | Onshape to URDF to USD to colours, in one go |
| `odm stop` / `odm list` | clean up / list runs and checkpoints |
| `scripts/diag/scoreboard.py [ver…]` | the scoring table above |
| `scripts/console.py --host <ip>` | keyboard dashboard |

If you can't find something, [`docs/map.md`](docs/map.md) is the index. Most of the
docs are in Korean.

## Requirements

Ubuntu with an NVIDIA GPU, plus Isaac Sim and Isaac Lab. Verified against Isaac Lab
0.54.3, Isaac Sim 5.1.0, rsl-rl 5.0.1, torch 2.7.0+cu128. macOS can run the pure
Python tests and nothing else.

```bash
pip install -e .
export ISAACLAB_PATH=~/Desktop/IsaacLab   # not vendored here
```

placo, pinocchio and meshcat conflict with Isaac's Python, so they get their own venv
at `~/.odm-tools`. `./demo.sh setup` builds it.

## Robot facts you need before touching anything

| | value | source of truth |
|---|---|---|
| mass | 2.7430 kg | `robot/robot.urdf` |
| mass, big-foot variant | 2.7518 kg | `big_foot/robot.urdf`, +10.4 g per foot, −12.0 g trunk |
| links / actuated joints | 20 / 14 | same |
| actuators | Dynamixel XM430-W350 ×14 | `source/.../hardware_map.py` |
| body frame | +x forward, +y left, +z up | `source/.../imu_map.py` |
| joint order | `ACTUATOR_JOINT_NAMES` | `source/.../joint_order.py` |
| IMU | BNO055, i2c-7 `0x28`, identity axis map | `source/.../imu_map.py` |

Don't reorder the joints. That order *is* the policy's action vector layout, and
changing it breaks the ONNX export and the hardware port without any error message.

The `axis` field in `robot/robot.urdf` is `0 0 1` on all 14 joints and carries no
directional information. `onshape-to-robot` pins the rotation axis to local Z and puts
the real direction in `<origin rpy>`. Read the notes in `imu_map.py` if you need signs.

## The pipeline

### CAD to sim robot

```bash
odm import      # backup, import, tally warnings, convert to USD, inject colours
```

Point `robot/config.json` at your Onshape URL. Check the workspace ID (`w/`) carefully:
if the document and element IDs match and only `w/` differs, the URL looks unchanged
but you pull geometry from months ago. `odm import` prints which workspace it used.

A clean import ends with ERROR 0, no-mass 0, multiple-base 0. `Multiple base links`
means some instance in the assembly isn't mated, and that part gets silently dropped.
Boolean in the Part Studio won't fix it. See [`docs/onshape_import.md`](docs/onshape_import.md).

### Reference gait

placo sweeps the command grid and the result gets fitted to polynomials.

```bash
python3 scripts/setup/patch_urdf_for_placo.py
./scripts/setup/gen_reference_remote.sh --height 0.193 --yaw-sweep 0.28 --out ref_h193
```

`--height` is CoM height, not body height, so the same number gives a different pose
once the mass distribution changes. Use `--yaw-sweep 0.28`; the default grid misses
zero-turn commands entirely.

The reference only produces **half** the commanded speed (0.15 m/s command gives
0.075 m/s of leg motion). The policy makes up the difference through
`tracking_lin_vel`. Replaying the reference with `scripts/viz_ref_pkl.py` looks like
the feet are sliding, which they are, because it's pure kinematics with no contact.

### Training

```bash
odm train v74 3000 4096
odm watch v74
odm tb
```

The version name resolves to a task automatically. Two Isaac Sim instances on one GPU
means one of them dies quietly, so `odm` refuses to start a second.

### Judging

```bash
odm measure v74
python3 scripts/diag/scoreboard.py v61 v68 v73 v74
scripts/train_health.py --run <run dir>      # lr floor, std collapse, reward plateau
```

### Hardware

Hang the robot up before you start. Tools are in `scripts/hw/`.

| tool | what it does |
|---|---|
| `joint_cal.py` | joint sign check, interactive, needs a TTY |
| `goto_ready.py` | move slowly to the READY pose |
| `imu_check.py` | wake the BNO055, work out the axes, verify |
| `rl_walk.py` | run a policy. `--joy`, `--path-imu`, `--current`, `--gyro-bias` |
| `export_ref_gait.py` + `play_ref_gait.py` | open-loop reference gait replay |
| `dxl_bridge.py` + `cad_teleop.py` | browser slider teleop, command vs measured |

The U2D2 is on the Jetson at `/dev/ttyUSB0`, not the desktop.

Two things about gains that cost me time. The position PID lives in **RAM**, so it
resets on power cycle, on reboot, and on mode change. 800/0/4700 isn't a value anyone
chose, it's what the firmware writes when you switch to mode 5. Set gains *after*
`arm()` does the mode switch or they vanish. Homing Offset (address 20) is EEPROM and
only writable with torque off.

## Rules I don't break

- Self-collision destroys actuators. Keep 5 mm between legs and body. Runtime guard is
  the CBF filter in `source/.../safety_filter.py`.
- Don't reorder joints.
- One Isaac Sim at a time.
- Never delete existing runs or checkpoints. They're the baselines.
- Jog the real robot inside URDF limits, one joint at a time.
- The lab PC (`do@192.168.137.111`) is shared. Nothing gets deleted there.
- No force pushes. Single `main` branch.
- Measure a new reward term's raw value before training with it.

## Docs

| | |
|---|---|
| [`docs/map.md`](docs/map.md) | where everything is. Start here |
| [`docs/decisions.md`](docs/decisions.md) | design decisions and why, actuator gains, env API |
| [`docs/training_log.md`](docs/training_log.md) | what changed each run and what happened |
| [`docs/versions.md`](docs/versions.md) | version index, generated by `scripts/diag/versions.py` |
| [`docs/onshape_import.md`](docs/onshape_import.md) | CAD import, and the traps in it |
| [`docs/isaaclab_setup.md`](docs/isaaclab_setup.md) | Isaac Lab install |
| [`docs/webrtc_streaming.md`](docs/webrtc_streaming.md) | remote streaming and control |
| [`docs/graph_conventions.md`](docs/graph_conventions.md) | plotting conventions |
| [`docs/handoff/`](docs/handoff/) | hardware bringup, joint and IMU calibration |
| [`docs/reports/`](docs/reports/) | one-off investigations |
| [`docs/hw_logs/`](docs/hw_logs/) | raw hardware run logs, read with `scripts/hw/analyze_hw_log.py` |

Read `docs/training_log.md` before designing an experiment. A fair number of hypotheses
in there have already failed, and that record is usually the best argument for what to
try next.
