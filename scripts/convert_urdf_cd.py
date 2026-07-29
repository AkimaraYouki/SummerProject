"""URDF -> USD 변환, 충돌 형상을 convex decomposition으로.

IsaacLab의 scripts/tools/convert_urdf.py는 UrdfConverterCfg의 collider_type을
설정하지 않아 기본값 "convex_hull"이 쓰인다. 이 로봇에서는 그것이 문제가 된다:
자기충돌을 켜면 첫 스텝부터 수천 N의 접촉력이 뜨는데(neck_pitch<->neck_yaw
4340 N, trunk<->roll_to_pitch ~2000 N), pinocchio로 정확한 메시를 재보면 같은
쌍이 10~14 mm 떨어져 있고 관통은 0이다. 즉 물리 문제가 아니라 충돌 형상
근사의 문제다.

joint/body 이름이 바뀌면 joint_order.py와 학습된 정책이 전부 깨지므로,
scripts/convert_urdf.sh와 동일한 인자(merge-joints, 게인, target type)를 쓰고
collider_type만 바꾼다. 출력도 기존 USD를 덮지 않고 별도 파일로 쓴다.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)
parser.add_argument("--merge-joints", action="store_true", default=False)
parser.add_argument("--fix-base", action="store_true", default=False)
parser.add_argument("--joint-stiffness", type=float, default=37.65)
parser.add_argument("--joint-damping", type=float, default=1.352)
parser.add_argument("--joint-target-type", type=str, default="position")
parser.add_argument("--collider-type", type=str, default="convex_decomposition",
                    choices=["convex_hull", "convex_decomposition"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

cfg = UrdfConverterCfg(
    asset_path=os.path.abspath(args_cli.input),
    usd_dir=os.path.dirname(os.path.abspath(args_cli.output)),
    usd_file_name=os.path.basename(args_cli.output),
    fix_base=args_cli.fix_base,
    merge_fixed_joints=args_cli.merge_joints,
    force_usd_conversion=True,
    collider_type=args_cli.collider_type,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=args_cli.joint_stiffness,
            damping=args_cli.joint_damping,
        ),
        target_type=args_cli.joint_target_type,
    ),
)
print(f"[convert] collider_type={cfg.collider_type}", flush=True)
conv = UrdfConverter(cfg)
print(f"[convert] wrote {conv.usd_path}", flush=True)
simulation_app.close()
