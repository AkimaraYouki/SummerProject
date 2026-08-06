"""URDF 의 색을 변환된 USD 에 입힌다.

IsaacLab 의 URDF 변환기는 시각 머티리얼을 통째로 버린다 -- URDF 에 material 이
267 개 정의돼 있는데 변환된 USD 에는 `def Material` 이 0 개다. 그래서 Isaac Sim
에서는 로봇이 전부 회색으로 보이고, URDF 를 직접 읽는 meshcat 에서는 색이 나온다.

짝은 **부모 프림 이름**으로 맞춘다. 메시 프림은 전부 `mesh` 로 같은 이름이지만
경로가 `/visuals/<링크>/<부품명>/mesh` 이고 그 부품명이 STL 파일 stem 과 같다
(`batt`, `body_bottom`, ...). URDF 의 <visual> 은 <mesh filename="...stl"> 과
<material><color rgba=...> 을 같이 들고 있으므로 stem 을 열쇠로 쓰면 된다.

`displayColor` 로 넣는다. 프리뷰 서피스 머티리얼을 만들어 바인딩하는 편이
정석이지만, 프림 수백 개에 대해 머티리얼 프림을 만들면 USD 가 무거워지고
렌더러 설정을 타는데, displayColor 는 Isaac Sim 뷰포트가 바로 읽는다.

    isaaclab.sh -p scripts/setup/inject_materials.py          # 기본 USD
    isaaclab.sh -p scripts/setup/inject_materials.py --dry-run
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

from pxr import Gf, Usd, UsdGeom, Vt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def urdf_colors(urdf):
    """STL 파일 stem -> (r, g, b). 같은 메시가 여러 번 쓰이면 첫 색을 쓴다."""
    out = {}
    for vis in ET.parse(urdf).getroot().iter("visual"):
        mesh = vis.find("geometry/mesh")
        color = vis.find("material/color")
        if mesh is None or color is None:
            continue
        stem = os.path.splitext(os.path.basename(mesh.get("filename", "")))[0]
        rgba = [float(x) for x in color.get("rgba", "").split()]
        if stem and len(rgba) >= 3:
            out.setdefault(stem, tuple(rgba[:3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(ROOT, "robot", "robot.urdf"))
    ap.add_argument("--usd", nargs="*", default=[
        os.path.join(ROOT, "robot", "usd", "configuration", "open_duck_mini_v2_base.usd"),
        os.path.join(ROOT, "robot", "usd", "configuration", "open_duck_mini_v2_physics.usd"),
    ], help="메시는 최상위가 아니라 configuration/ 레이어에 있다")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    colors = urdf_colors(a.urdf)
    print(f"[info] URDF 색 {len(colors)}종")

    for usd in a.usd:
        stage = Usd.Stage.Open(usd, Usd.Stage.LoadAll)
        hit = miss = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            # 메시 프림은 모두 이름이 `mesh` 다. 부품명은 **조상 어딘가**가 들고 있다.
            #
            # 부모 하나만 보면 안 된다 (2026-08-07). 임포터 버전에 따라 부품명과
            # 메시 사이에 중간 노드가 낀다:
            #   예전  /visuals/<링크>/<부품>/mesh
            #   지금  /visuals/<링크>/<부품>/node_STL_BINARY_/mesh
            # 부모만 보면 `node_STL_BINARY_` 를 열쇠로 삼아 642개 전부 놓친다.
            # 그래서 조상을 거슬러 올라가며 색 표에 있는 이름을 찾는다.
            key = None
            p = prim.GetParent()
            while p and p.GetPath().pathString != "/":
                for cand in (p.GetName(), re.sub(r"_\d+$", "", p.GetName())):
                    if cand in colors:
                        key = cand
                        break
                if key:
                    break
                p = p.GetParent()
            if key is None:
                miss += 1
                continue
            hit += 1
            if not a.dry_run:
                g = UsdGeom.Mesh(prim)
                g.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colors[key])]))
                g.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        name = os.path.basename(usd)
        print(f"[info] {name}: 색 입힘 {hit} · 못 찾음 {miss}")
        if not a.dry_run:
            stage.GetRootLayer().Save()
            print(f"[ok] 저장: {name}")
    if a.dry_run:
        print("[dry-run] 저장하지 않음")


if __name__ == "__main__":
    main()
