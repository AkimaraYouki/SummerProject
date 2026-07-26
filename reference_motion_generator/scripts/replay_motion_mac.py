"""macOS-compatible version of replay_motion.py.

Why this exists: FramesViewer.Viewer.start() runs the GLUT window/render
loop (glutMainLoop()) in a background thread. macOS's Cocoa windowing
requires all GUI/GL calls to happen on the main thread — on macOS that
background thread silently produces no visible window (no error raised).
See docs/webrtc_streaming.md's "OnShape importer" section for the same
underlying GLFW/GLUT-on-macOS constraint hit elsewhere in this project.

Fix part 1: don't call Viewer.start() (which backgrounds FramesViewer's
private __initGL/glutMainLoop). Instead run that same GL setup directly on
the main thread (blocking, as GLUT requires), and push frames from a
background thread instead — the inverse of upstream's threading model,
but functionally identical: push_frame() just updates a dict that the
GLUT display callback reads every redraw.

Fix part 2 (found by testing on this machine): FramesViewer.Camera.__init__
calls self.update(0), which issues raw OpenGL calls (glMatrixMode etc.)
immediately — but Viewer() constructs its Camera *before* any GLUT window/
GL context exists (that only happens later, inside __initGL). On this
PyOpenGL/macOS setup that segfaults immediately (SIGSEGV), every time,
independent of the threading fix above — reproduced by constructing
FramesViewer.camera.Camera() in isolation. Worked around by monkeypatching
Camera.update to a no-op for the duration of Viewer()'s constructor only,
then restoring the real update() before the GL context exists — by the
time it's called for real (every frame, from Viewer.__run after
__initGL has created the window), a valid context is in place.

Usage (identical to upstream replay_motion.py):
  python3 scripts/replay_motion_mac.py -f path/to/recording.json
"""

import argparse
import json
import threading
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.camera import Camera
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

_real_camera_update = Camera.update
Camera.update = lambda self, dt: None

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
args = parser.parse_args()

fv = Viewer()
Camera.update = _real_camera_update  # restore now that construction is done

episode = json.load(open(args.file))
frame_duration = episode["FrameDuration"]
frames = episode["Frames"]
frame_offsets = episode["Frame_offset"][0]

root_pos_slice = slice(frame_offsets["root_pos"], frame_offsets["root_quat"])
root_quat_slice = slice(frame_offsets["root_quat"], frame_offsets["joints_pos"])
left_toe_pos_slice = slice(frame_offsets["left_toe_pos"], frame_offsets["right_toe_pos"])
right_toe_pos_slice = slice(frame_offsets["right_toe_pos"], frame_offsets["world_linear_vel"])


def feed_frames():
    # Runs on a background thread — the GL window itself lives on the main
    # thread (see bottom of file). push_frame() only writes into a dict
    # that the display callback reads on its next redraw, so this is safe
    # without extra locking (same as upstream's single-threaded version,
    # just with the GL loop and the data loop on swapped threads).
    pose = np.eye(4)
    while True:
        for frame in frames:
            root_position = frame[root_pos_slice]
            root_orientation_quat = frame[root_quat_slice]
            root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()

            pose[:3, 3] = root_position
            pose[:3, :3] = root_orientation_mat

            fv.push_frame(pose, "trunk")

            # left_toe_pos/right_toe_pos are T_body_leftFoot/T_body_rightFoot
            # from gait_generator.py — i.e. TRUNK-LOCAL coordinates, not
            # world. Upstream replay_motion.py plots them raw (its
            # "+ np.array(root_position)" is commented out), which is why
            # that tool's view looks like disconnected floating axes too.
            # Transform to world frame with the trunk's full pose (rotation
            # + translation) so the feet actually render under the trunk.
            left_toe_pos_body = np.array(frame[left_toe_pos_slice])
            right_toe_pos_body = np.array(frame[right_toe_pos_slice])
            left_toe_pos = root_orientation_mat @ left_toe_pos_body + np.array(root_position)
            right_toe_pos = root_orientation_mat @ right_toe_pos_body + np.array(root_position)

            fv.push_frame(fv_utils.make_pose(left_toe_pos, [0, 0, 0]), "left_toe")
            fv.push_frame(fv_utils.make_pose(right_toe_pos, [0, 0, 0]), "right_toe")

            time.sleep(frame_duration)


threading.Thread(target=feed_frames, daemon=True).start()

print(f"Loaded {len(frames)} frames from {args.file} (frame_duration={frame_duration}s).")
print("Opening viewer window on the main thread (macOS requires this) — close the window to exit.")

# Deliberately NOT fv.start() — see module docstring. This calls the same
# private GL setup directly, blocking on the main thread as GLUT requires.
fv._Viewer__initGL()
