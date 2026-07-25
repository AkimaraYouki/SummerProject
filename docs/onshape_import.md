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
├── assets/       # 메시(STL) 파일들
└── usd/          # convert_urdf.sh가 생성하는 USD (open_duck_mini_v2.usd) — .gitignore 대상
```

`scripts/convert_urdf.sh`가 이 위치(`robot/robot.urdf`)를 그대로 읽어서 `robot/usd/`에 USD로
변환한다 — **입력(URDF/메시)과 출력(USD) 전부 `robot/` 안에 모여있다.** 과거 원본 Open Duck
Mini 클론에서 가져온 `assets/robot/open_duck_mini_v2/`, 그리고 한때 있었던 최상위 `assets/usd/`는
이제 안 쓴다.

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

## ⚠️ 임포트 결과가 실행마다 다르다 — 근본 원인 (2026-07-25, 여러 차례 재현 후 확정)

같은 `config.json`으로 여러 번 재임포트해본 결과, **매번 파츠 구성이 달라졌다**:

| 실행 | 로그 줄 수 | 파츠 인스턴스 | 전자보드 포함 여부 | "no mass" 경고 | "multiple base link" 경고 |
|---|---|---|---|---|---|
| 최초 임포트 | 522줄 | (보드 포함) | 포함됨 | 20개 파츠 | 있음 |
| 재실행(캐시 있음) | 288줄 | 263개 | **누락됨** | 0개 | 없음 |
| 캐시 완전 삭제 후 재실행 | 288줄 | 263개 | **누락됨** | 0개 | 없음 |

**근본 원인**: `Main_Board_Simplified`/`MODULE_BOARD_ME`(전자보드류)가 OnShape 어셈블리에서
메인 로봇 구조에 **메이트로 연결돼 있지 않다** — "떠있는" 별도 서브트리로 존재한다. 이게
바로 `Multiple base links detected, which is not supported by URDF. Only the first base link
will be considered.` 경고의 정체이고, 어느 쪽을 "첫 번째"로 잡을지가 실행마다 안정적이지
않아서 보드가 포함되기도, 빠지기도 한다. **캐시 문제가 아니라 OnShape 어셈블리 구조 자체의
문제**(`onshape-to-robot-clear-cache`로 캐시를 완전히 지워도 재현됨을 확인).

**질량에는 실질적 영향 없음**: 두 경우 다 전체 로봇 질량은 **1.9809kg으로 동일**했다 — 그
전자보드 부품들은 OnShape에서 애초에 재질(밀도)이 할당돼 있지 않아서, 포함되든 안 되든
질량 기여가 0이었기 때문. `robot.urdf`를 직접 파싱해 확인한 결과 **15개 링크 중 질량 0인
링크는 0개, 263개 파츠 인스턴스 전체 중 질량 없는 파츠도 0개**(캐시 삭제 후 클린 임포트 기준).

**해결됨 (2026-07-25, 같은 날 후속 조치)**: 이후 OnShape에서 `Main_Board_Simplified`를
**블룬(불리언) 처리로 `Part 1`에 합치고 원래 이름은 삭제**했다. 그래서 이후 재임포트
로그에 `Main_Board_Simplified`라는 이름이 안 보이는 게 당연한 것이었다 — 사라진 게
아니라 이름이 없어진 것. 메이트 재작업 후 "multiple base link" 경고도 사라졌고,
`Part 1`은 모든 클린 재임포트에서 안정적으로 포함되며 질량도 정상(no-mass 경고 0건)임을
확인했다. **결론: 보드 지오메트리·질량 다 정상적으로 들어가 있고, 더 이상 조치 불필요.**

**최종 검증 (2026-07-25, 최신 재임포트 전체 로그 기준)**: 남은 경고는 단 2건, 전부 무해함:
```
WARNING: Parts with same name "part_1", incrementing STL name to "part_1__2"
WARNING: Parts with same name "part_1", incrementing STL name to "part_1__3"
```
같은 파츠가 여러 인스턴스로 재사용될 때 자동으로 이름을 구분해주는 정상 동작 — 무시해도 됨.
**"no mass" 경고, "multiple base link" 경고, ERROR 전부 0건.** 링크 15개/관절 14개/전체
질량 1.9809kg. **이 리포트를 기준으로 `robot/robot.urdf`는 USD 변환 다음 단계로 넘어가도
안전한 상태.**

(참고: 이 과정에서 `onshape-to-robot`이 OnShape 서버로의 TCP 연결(SYN-SENT)에서 두 차례
멈췄다 — 캐시를 완전히 지운 직후의 콜드 스타트에서만 발생, 캐시를 유지한 채로 재실행하면
발생 안 함. 짧은 시간에 반복적으로 캐시를 지우고 전체 재임포트한 것과 관련 있을 가능성.
멈추면 프로세스를 kill하고 캐시를 지우지 않은 상태로 재시도할 것.)

## 참고 — 첫 임포트에서 확인된 사실

첫 임포트(2026-07-25)에서 나온 14개 관절 이름이 `joint_order.py`의 기존 `ACTUATOR_JOINT_NAMES`와
**완전히 일치했다** — 로봇이 기구학적으로 원본 Open Duck Mini v2와 동일하다는 걸(액추에이터·
컴퓨터만 교체) 실제 데이터로 재확인. `xm430_어셈`이라는 base link도 있어 XM430 액추에이터가
어셈블리에 포함된 것도 확인됨.
