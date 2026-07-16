"""Static verification that REF_LEG_JOINT_IDX / ACT_LEG_JOINT_IDX (used by
rewards.py::reward_imitation to compare live joint state against the
16-joint reference frame) actually select the same 10 physical leg joints
in the same order. See docs/decisions.md for why this needed a dedicated
check rather than trusting the slices by inspection — the original
Playground source itself carries a `# TODO double check if the slices are
correct` comment on this exact code.

Mac-runnable, no torch/Isaac Sim dependency.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.joint_order import (  # noqa: E402
    ACT_LEG_JOINT_IDX,
    ACTUATOR_JOINT_NAMES,
    REF_JOINT_NAMES,
    REF_LEG_JOINT_IDX,
)

EXPECTED_LEG_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def test_both_index_lists_select_the_same_10_leg_joints_in_the_same_order():
    from_actuator = [ACTUATOR_JOINT_NAMES[i] for i in ACT_LEG_JOINT_IDX]
    from_reference = [REF_JOINT_NAMES[i] for i in REF_LEG_JOINT_IDX]
    assert from_actuator == from_reference == EXPECTED_LEG_ORDER


def test_index_lists_have_length_10():
    assert len(ACT_LEG_JOINT_IDX) == 10
    assert len(REF_LEG_JOINT_IDX) == 10


def test_excluded_indices_are_head_and_antennas_only():
    excluded_ref = set(range(16)) - set(REF_LEG_JOINT_IDX)
    excluded_names = {REF_JOINT_NAMES[i] for i in excluded_ref}
    assert excluded_names == {"neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna", "right_antenna"}

    excluded_act = set(range(14)) - set(ACT_LEG_JOINT_IDX)
    excluded_act_names = {ACTUATOR_JOINT_NAMES[i] for i in excluded_act}
    assert excluded_act_names == {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}


if __name__ == "__main__":
    test_both_index_lists_select_the_same_10_leg_joints_in_the_same_order()
    test_index_lists_have_length_10()
    test_excluded_indices_are_head_and_antennas_only()
    print("All reward leg-index alignment tests passed.")
