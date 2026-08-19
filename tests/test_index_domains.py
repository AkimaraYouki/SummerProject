"""인덱스를 **엉뚱한 목록에 쓰는 것**을 정적으로 막는다.

## 왜 이 테스트가 있나

2026-08-14, `_feet_ids` 는 **접촉센서**의 body 목록에 대한 인덱스인데 그것을
`_robot.data.*`(아티큘레이션)에 그대로 썼다. 두 목록의 순서가 달라서 전혀
다른 링크가 나온다 — 이 로봇에서는:

    접촉센서    [5] foot_assembly       [14] foot_assembly_2
    아티큘레이션 [5] head_pitch_assembly  [14] foot_assembly_2

즉 "왼발" 자리에 **머리**가 들어갔다. `body_pos_w` 의 z 가 [0.329, 0.016] 로
나와서 겨우 잡았다 (몸통이 0.19 m 인데 발이 0.33 m 에 있을 수 없다).

그 사이 만든 발 리워드 다섯 개(foot_lift/lateral/clearance/slip/impact)가
전부 머리를 보고 있었고, v53·v55 의 결과 해석이 통째로 뒤집혔다. 학습 다섯
판이 무효가 됐다.

숫자가 우연히 겹칠 수 있어서 실행 중에는 잘 안 드러난다 — 이 로봇도
오른발만 우연히 14 로 일치했다. 그래서 **정적으로** 막는다.

Isaac 없이 0.01 초에 돈다.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = ROOT / "source/open_duck_mini_isaaclab/tasks/velocity/joystick_env.py"

#: 접촉센서에서 온 인덱스. `_robot.data.*` 에 쓰면 안 된다.
SENSOR_IDX = "_feet_ids"
#: 아티큘레이션에서 뽑은 인덱스. 이쪽을 써야 한다.
ARTIC_IDX = "_foot_body_ids"


def test_sensor_index_not_used_on_articulation():
    src = ENV.read_text()
    # 주석은 건너뛴다 — 이 버그를 **설명하는** 문장이 모듈 독스트링과 주석에
    # 남아 있고, 그게 오탐으로 잡히면 기록을 지우게 된다.
    bad = [
        (i, line.strip())
        for i, line in enumerate(src.splitlines(), 1)
        if not line.strip().startswith("#")
        and re.search(rf"_robot\.data\.\w+\[\s*:\s*,\s*self\.{SENSOR_IDX}", line)
    ]
    assert not bad, (
        f"접촉센서 인덱스({SENSOR_IDX})를 아티큘레이션 데이터에 썼다. "
        f"{ARTIC_IDX} 를 쓸 것 — 모듈 독스트링 참고:\n"
        + "\n".join(f"  L{i}: {t}" for i, t in bad)
    )


def test_articulation_index_is_derived_from_articulation():
    """`_foot_body_ids` 는 반드시 `_robot.find_bodies` 로 만들어야 한다."""
    src = ENV.read_text()
    m = re.search(rf"self\.{ARTIC_IDX}\s*,\s*\w+\s*=\s*self\.(\w+)\.find_bodies", src)
    assert m, f"{ARTIC_IDX} 를 find_bodies 로 만드는 곳을 못 찾았다"
    assert m.group(1) == "_robot", (
        f"{ARTIC_IDX} 를 self.{m.group(1)}.find_bodies 로 만들었다 — "
        "아티큘레이션(self._robot)에서 뽑아야 한다"
    )


def test_articulation_index_order_is_asserted():
    """이름 순서를 assert 로 확인하는 코드가 남아 있어야 한다.

    숫자만 보고 안심하면 안 된다 — 이 로봇은 오른발만 우연히 일치했다.
    """
    src = ENV.read_text()
    assert re.search(r"assert\s+_fb_names\s*==", src), (
        "_foot_body_ids 의 이름 순서를 확인하는 assert 가 사라졌다"
    )
