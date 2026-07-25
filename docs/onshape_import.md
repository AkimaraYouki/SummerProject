# OnShape → URDF 임포트 절차

Isaac Sim의 OnShape 임포터는 `docs/webrtc_streaming.md`에 기록된 대로 헤드리스 스트리밍
모드와 구조적으로 안 맞아서(GLFW 창을 못 띄움) 쓰지 않는다. 대신 `onshape-to-robot`
(OnShape API 키 기반 헤드리스 CLI)으로 URDF를 직접 뽑는다.

## 캐노니컬 위치

**앞으로 모든 OnShape 임포트 결과물은 이 프로젝트의 `robot/` 디렉토리에 둔다:**

```
robot/
├── config.json   # onshape-to-robot 설정 (어셈블리 URL 등)
├── robot.urdf    # 생성된 URDF
└── assets/       # 메시(STL) 파일들
```

`scripts/convert_urdf.sh`가 이 위치(`robot/robot.urdf`)를 그대로 읽어서 USD로 변환한다.
과거 원본 Open Duck Mini 클론에서 가져온 `assets/robot/open_duck_mini_v2/`는 이제 레거시
참고용이고, 새 임포트는 전부 `robot/`으로 간다.

## 사전 준비 (랩 PC, 한 번만)

- `onshape-to-robot`은 `pip install --user onshape-to-robot`로 이미 설치돼 있음
  (`~/.local/bin`, `~/.bashrc`에 PATH 등록됨)
- OnShape API 키는 `~/.onshape_env`에 저장돼 있음 (⚠️ `.bashrc`에 넣으면 비대화형 SSH
  세션에서는 상단의 "비대화형이면 종료" 가드 때문에 안 읽힌다 — 그래서 별도 파일로 분리함).
  `source ~/.onshape_env`로 명시적으로 불러와야 함.

## 새 임포트 실행 절차

1. `robot/config.json`의 `url`을 원하는 OnShape 어셈블리 URL로 교체
   (OnShape에서 **어셈블리 탭**을 연 상태의 브라우저 주소창 URL을 그대로 복사)
   ```json
   {
     "url": "https://cad.onshape.com/documents/.../w/.../e/...",
     "output_format": "urdf"
   }
   ```
2. 랩 PC에서 실행:
   ```bash
   ssh do@192.168.137.111
   source ~/.onshape_env
   cd "/media/do/Extreme SSD/parksuho/open_duck_mini_isaaclab/robot"
   onshape-to-robot .
   ```
3. `robot.urdf`와 `assets/*.stl`이 갱신된다. 관절 이름은 아래로 확인:
   ```bash
   grep -oP '(?<=<joint name=")[^"]+' robot.urdf
   ```
   `source/open_duck_mini_isaaclab/joint_order.py`의 `ACTUATOR_JOINT_NAMES` 14개와
   이름 집합이 일치하는지 반드시 대조할 것 — 다르면 `joint_order.py`를 갱신해야 함.
4. USD로 변환 (기존 검증된 경로 그대로):
   ```bash
   ISAACLAB_PATH=~/IsaacLab ./scripts/convert_urdf.sh --headless
   ```

## 알려진 경고 (2026-07-25 첫 임포트, 522줄 전체 로그 확인 — 매 실행마다 재현됨, ERROR는 0건)

중복 제거하면 고유 경고는 21종:

- **"part ... has no mass, maybe you should assign a material to it?"** (20개 파츠) — 이름
  없는 범용 파츠(`Part 1`~`Part 16`, 아마 나사류)와 전자보드(`Main_Board_Simplified`,
  `MODULE_BOARD_ME`)에 OnShape에서 재질이 할당 안 돼서 그 파츠 개별로는 질량 0.
  **실제 `robot.urdf`를 파싱해서 확인한 결과, 15개 링크 중 질량 0인 링크는 0개** —
  재질 없는 파츠들이 재질 있는 다른 파츠와 같은 URDF 링크로 묶이면서 링크 단위 합산
  질량은 0이 되지 않았음(전체 로봇 질량 합계 1.98kg). 물리적으로 당장 문제는 아님.
  다만 이 파츠들 자체의 무게(나사 등, 미미할 것으로 추정)는 합산 질량에서 빠져있으니,
  무게중심 정밀도가 중요해지면 그때 재질을 지정해줄 것.
- **"Parts with same name ..., incrementing STL name to ..."** (2건) — 같은 파츠가 여러
  인스턴스로 재사용될 때 자동으로 `_2` 접미사를 붙여 처리. 정상 동작, 무시해도 됨.
- **"Multiple base links detected, which is not supported by URDF. Only the first base
  link will be considered."** — **매 실행마다 재현됨** (이전 기록에 "재현 안 됨"이라고
  적었던 건 grep이 ANSI 색상 코드 때문에 놓친 오탐이었음, 정정함). OnShape 어셈블리 안에
  메이트로 메인 트리에 연결 안 된 "떠있는" 파츠가 있어서 URDF가 지원 못 하는 두 번째
  루트가 생기고, 그중 하나만 채택된다는 뜻. 관절 14개는 매번 정상적으로 다 나오지만,
  **USD 변환 후 Isaac Sim에서 뭔가 시각적으로 빠진 게 없는지 반드시 눈으로 확인할 것.**

## 참고 — 첫 임포트에서 확인된 사실

첫 임포트(2026-07-25)에서 나온 14개 관절 이름이 `joint_order.py`의 기존 `ACTUATOR_JOINT_NAMES`와
**완전히 일치했다** — 로봇이 기구학적으로 원본 Open Duck Mini v2와 동일하다는 걸(액추에이터·
컴퓨터만 교체) 실제 데이터로 재확인. `xm430_어셈`이라는 base link도 있어 XM430 액추에이터가
어셈블리에 포함된 것도 확인됨.
