"""Replays a recorded reference-motion .json with the actual robot meshes in
MeshCat (browser-based) — pure pinocchio, NO placo.

Why not placo: placo.HumanoidRobot (0.9.23 on this Mac) silently mangles the
displayed pose — at a frame where the recorded data has both feet on the
ground, it reported the left_foot frame at z=+0.173 (hip height!) while
right_foot sat near the ground. Its internal support-foot re-anchoring
and/or joint→q mapping does not respect a plain
set_T_world_fbase/set_joint/update_kinematics sequence, so every prior
MeshCat replay rendered contorted poses that do NOT exist in the data
(verified: recorded world toe heights stay within [-0.01, +0.03] m for both
feet, and independent numpy FK through the URDF shows perfectly symmetric,
ground-level foot frames). Building q explicitly by joint name through raw
pinocchio removes all of that magic.

Usage:
  python3 scripts/replay_motion_meshcat.py -f path/to/recording.json
  (open the printed URL in a browser)
"""

import argparse
import json
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, nargs="+", required=True,
                    help="one or more recording .json files; multiple files play back-to-back in a loop")
parser.add_argument(
    "--urdf",
    type=str,
    default="reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/open_duck_mini_v2.urdf",
)
parser.add_argument("--freeze", type=int, default=None, help="display only this frame index and hold")
args = parser.parse_args()

episodes = []
for path in args.file:
    ep = json.load(open(path))
    episodes.append((path.split("/")[-1], ep))

mesh_dir = "/".join(args.urdf.split("/")[:-1])
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    args.urdf, mesh_dir, pin.JointModelFreeFlyer()
)

# Map recording joint order -> pinocchio q indices, explicitly by name.
joint_names = episodes[0][1]["Joints"]
qidx = {}
for name in joint_names:
    jid = model.getJointId(name)
    assert jid > 0, f"joint {name} not in pinocchio model"
    qidx[name] = model.joints[jid].idx_q

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=False)
viz.loadViewerModel()
print(f"Viewer URL: {viz.viewer.url()}")
print(f"Playlist ({len(episodes)} clips): " + ", ".join(n for n, _ in episodes))

q = pin.neutral(model)


def show(off, frame):
    # freeflyer q[0:7] = [x, y, z, qx, qy, qz, qw] (pinocchio quaternion order
    # matches scipy's/our recording's xyzw)
    q[0:3] = frame[off["root_pos"] : off["root_pos"] + 3]
    q[3:7] = frame[off["root_quat"] : off["root_quat"] + 4]
    for name, angle in zip(joint_names, frame[off["joints_pos"] : off["joints_pos"] + len(joint_names)]):
        q[qidx[name]] = angle
    viz.display(q)


if args.freeze is not None:
    name, ep = episodes[0]
    show(ep["Frame_offset"][0], ep["Frames"][args.freeze])
    print(f"FROZEN on frame {args.freeze} of {name} — Ctrl+C to exit.")
    while True:
        time.sleep(5)

while True:
    for name, ep in episodes:
        off = ep["Frame_offset"][0]
        frame_duration = 1.0 / ep["FPS"]
        print(f"▶ {name}")
        for frame in ep["Frames"]:
            show(off, frame)
            time.sleep(frame_duration)
