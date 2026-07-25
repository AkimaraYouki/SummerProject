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

## 알려진 경고 (2026-07-25 첫 임포트 기준, 치명적이지 않음)

- **"part ... has no mass, maybe you should assign a material to it?"** — OnShape에서
  이름 없는(`Part 1`, `Part 2` ...) 일부 파츠에 재질이 할당 안 돼서 질량 0으로 잡힘.
  이름 붙은 주요 부품(발, 다리 프레임 등)엔 안 뜬 걸로 봐서 작은 고정 부품(나사 등)일
  가능성이 높지만, USD 변환 후 Isaac Sim에서 전체 질량이 말이 되는지 한 번은 확인할 것.
- **"Multiple base links detected... Only the first base link will be considered."**
  — 첫 실행에서 1회 뜨고 재실행 시엔 안 떴음(캐시 영향으로 추정). 관절 14개가 전부
  예상한 이름으로 나온 걸로 봐서 치명적 손실은 아닌 듯하나, 확신은 못 함 — 링크 개수가
  기대와 다르면 이게 원인일 수 있음.

## 참고 — 첫 임포트에서 확인된 사실

첫 임포트(2026-07-25)에서 나온 14개 관절 이름이 `joint_order.py`의 기존 `ACTUATOR_JOINT_NAMES`와
**완전히 일치했다** — 로봇이 기구학적으로 원본 Open Duck Mini v2와 동일하다는 걸(액추에이터·
컴퓨터만 교체) 실제 데이터로 재확인. `xm430_어셈`이라는 base link도 있어 XM430 액추에이터가
어셈블리에 포함된 것도 확인됨.
