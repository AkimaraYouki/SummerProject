#!/usr/bin/env python3
"""OnShape 임포트가 덮어써 버리는 로컬 수정을 `robot/robot.urdf` 에 다시 입힌다.

왜 필요한가: `odm import` 는 URDF 를 통째로 새로 쓴다. 그래서 CAD 에는 없고
여기서만 손으로 고쳐 둔 값은 재임포트 때마다 조용히 사라진다. 2026-08-07 에
head_pitch 한계를 ±50도로 넓혀 두었는데 2026-08-08 재임포트가 그걸 ±45도로
되돌렸고, USD 까지 그 상태로 변환됐다 (시뮬은 URDF 가 아니라 USD 에서 관절
한계를 읽으므로 재변환 전에는 티도 안 난다).

`joystick_env_cfg.py` 에 "재임포트하면 되돌아간다" 는 경고가 적혀 있었지만
경고는 실행되지 않는다. 그래서 표로 만들어 파이프라인에 묶었다.

멱등이다 — 이미 값이 맞으면 아무것도 안 하고 그렇게 보고한다.

    python3 scripts/setup/apply_local_urdf_overrides.py
"""

import math
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF = os.path.join(_REPO_ROOT, "robot", "robot.urdf")

# (관절 이름, lower, upper, 근거)
#
# head_pitch: CAD 의 ±45도는 기구 한계가 아니라 그냥 클램프다. 실제 설계
# 가동범위는 ~±60도 (2026-08-07 사용자 확인). Z 자 목 자세가 30도라서 45도면
# 여유가 15도뿐이고, v33 은 실제로 리밋에 얹힌 채 663 iter 를 돌았다.
# hardware_map.py 의 ("head_pitch": (-0.8727, 0.8727)) 와 같은 값이어야 한다.
JOINT_LIMIT_OVERRIDES = [
    ("head_pitch", -0.872665, 0.872665, "CAD 45도 클램프는 기구 한계가 아님 (실제 ~±60도)"),
]


def main():
    if not os.path.exists(URDF):
        sys.exit(f"URDF 가 없다: {URDF}")
    src = open(URDF).read()
    out = src
    changed, already = [], []

    for name, lo, hi, why in JOINT_LIMIT_OVERRIDES:
        # 해당 <joint name="..."> 블록 안의 <limit .../> 만 건드린다.
        pat = re.compile(
            r'(<joint name="%s" type="[^"]*">.*?<limit\b[^/]*?)'
            r'lower="([^"]*)"(\s*)upper="([^"]*)"' % re.escape(name),
            re.S,
        )
        m = pat.search(out)
        if not m:
            sys.exit(f"관절 {name} 의 <limit> 을 못 찾았다 — URDF 구조가 바뀌었는지 확인할 것")
        cur_lo, cur_hi = float(m.group(2)), float(m.group(4))
        if abs(cur_lo - lo) < 1e-9 and abs(cur_hi - hi) < 1e-9:
            already.append(name)
            continue
        out = pat.sub(lambda mm: f'{mm.group(1)}lower="{lo:g}"{mm.group(3)}upper="{hi:g}"', out, count=1)
        changed.append((name, cur_lo, cur_hi, lo, hi, why))

    if out != src:
        open(URDF, "w").write(out)

    for name, clo, chi, lo, hi, why in changed:
        print(f"[override] {name}  ±{math.degrees(chi):.1f}도 -> ±{math.degrees(hi):.1f}도   ({why})")
    for name in already:
        print(f"[override] {name}  이미 적용돼 있음")
    if changed:
        print("⚠️  URDF 를 고쳤으므로 USD 를 반드시 재변환할 것 — 시뮬은 USD 에서 한계를 읽는다.")


if __name__ == "__main__":
    main()
