# Open Duck Mini V2 — Isaac Lab

이족 보행 로봇 **Open Duck Mini V2** 의 전체 파이프라인. CAD 에서 로봇을 뽑아
시뮬레이터에 세우고, 모방할 보행 궤적을 만들고, 강화학습으로 걷게 한 뒤,
실제 하드웨어에 옮긴다.

---

## 📍 지금 상황 (2026-08-20)

실기는 자기 발로 걷는다 (2026-08-13 부터). 지금 남은 문제는 "걷느냐" 가 아니라
**어떻게 걷느냐** 다 — 피크 토크가 크고, 걸을 때 몸통이 좌우로 휘청인다.

### 목표가 바뀌었다

2026-08-20 부터 우선순위가 이렇다:

1. **피크 토크 최소화** — 액추에이터 한계에 상시 물려 있다
2. **보행 효율 최대화**
3. **토르소 움직임 최소화** — 걸을 때 좌우로 휘청거리지 않게
4. **주행 안정성 · robustness**
5. **명령 추종** — *천천히라도* 명령대로 가면 된다

**추종 정확도는 더 이상 1 순위가 아니다.** 이전에는 6 방향 추종 오차가 1 순위,
안정성이 2 순위였다. 그 기준으로 버렸던 정책들이 새 기준으로는 최고일 수 있어서
과거 결과를 다시 읽는 중이다 (아래 v55 참조).

### 판정 지표를 다시 세웠다

재지 않으면 달성했는지 알 수 없다. `scripts/diag/scoreboard.py` 에 세 열을 더했다:

| 열 | 정의 |
|---|---|
| **포화%** | `\|τ\|` 가 실효 한계 **3.16 N·m** 에 붙어 있는 시간 비율 |
| **일률W** | `\|τ·ω\|` 합의 평균 [W] — 낮을수록 효율적 |
| **roll속** | roll 각속도 RMS [deg/s] — 좌우 휘청거림 |

피크 토크는 **값 자체로는 지표가 안 된다.** 7 개 정책 전부 p99 가 3.16 으로
같았다 — 한계에 상시 물려 있다는 뜻이라, 얼마나 자주 물리는지로 바꿨다.

새 지표로 보니 답이 이미 데이터 안에 있었다:

| 버전 | 점수 | 포화% | 일률W | roll속 | rollRMS | 낙상% | |
|---|---|---|---|---|---|---|---|
| v55 | 0.0463 | 1.46 | 10.8 | **53.4** | **2.64** | **0.0** | `torso_ang_vel -0.7` |
| v60 | 0.0304 | 0.93 | 10.0 | 55.4 | 3.73 | 0.6 | `torso_ang_vel -1.2` |
| v68 | 0.0204 | **0.36** | **7.7** | 64.3 | 4.05 | 1.1 | 토크 −0.2 + 낙상종료 0.85 + σ1.3 |
| **v61** | 0.0221 | 1.21 | 9.6 | 75.8 | 4.97 | 0.7 | **실기에서 가장 잘 걷는 것** |
| v65 | **0.0138** | 1.05 | 9.9 | 87.5 | 4.89 | 1.9 | 심 점수 1 등 |
| v73 | 0.0269 | 1.18 | 10.2 | 79.6 | 4.71 | 1.2 | 큰 발 + CoM −10 mm |

**v55 는 낙상 0.0 % 로 역대 유일**인데 추종이 나빠서 버렸던 판이다. 목표가
바뀌니 같은 데이터가 정반대로 읽힌다. v68 은 토크 포화가 3.3 배, 일률이 24 %
낫다.

### 진행 중인 학습

| 버전 | 내용 | 상태 |
|---|---|---|
| v73 | big_foot USD + CoM 뒤로 10 mm | ✅ 완료 — 흔들림 개선 실패 (아래) |
| **v74** | v73 + 토크 −0.2 · 낙상종료 0.85 · 탐색 σ1.3 | 🔄 학습 중 |
| v75 | v74 + `torso_ang_vel -0.7` | ⏳ 대기 (v74 끝나면 자동) |

기반인 **big_foot USD + CoM −10 mm 는 계속 유지**한다.

### ⚠️ 심 점수 1 등이 실기 1 등이 아니다

| | 심 점수 | roll RMS | 실기 |
|---|---|---|---|
| v59 | 0.0165 (1 등) | 6.66 | 잘 안 됨 |
| **v61** | 0.0221 | 4.97 | **가장 잘 걷는다** |

실기에서는 **넘어지면 그 주행이 끝난다.** 추종 0.003 차이보다 흔들림 25 % 가
성패를 가른다. **심 점수로 후보를 좁히되 순위는 실기로 정한다.**

### 측정이 뒤집은 것들

이 프로젝트에서 되풀이된 실패 양식은 하나다 — **재지 않고 추정했다.**

| 믿었던 것 | 실제 | 대가 |
|---|---|---|
| 심의 발 접지폭이 16 mm 라 칼날 위에 서 있다 | 심이 딛는 면은 TPU 밑창이 **아니라** `l_foot_side` 다. TPU 는 2.70 mm 위라 안 닿는다. 심의 좌우 지지는 이미 **38.7 mm** | v73 이 겨냥한 효과가 애초에 심에 없었다 |
| 발 리워드가 발을 보고 있다 | `_feet_ids` 는 **접촉센서** 인덱스인데 아티큘레이션에 썼다. `[5]` 는 머리였다 | 학습 5 판 무효 |
| 발을 7 Hz 로 떤다 | 40 ms 디바운싱하니 전 정책이 540 ms 로 같다. **착지 바운스**였다 | 배포 판단이 뒤집혔다 |
| 발이 미끄러진다 | 실측 μ > 0.38. 마찰은 원인이 아니다 | — |
| 심 USD 가 문서보다 90 g·26 mm 어긋난다 | USD 2.7430 kg, CoM y +0.2 mm — URDF 와 정확히 일치 | docstring 을 재지 않고 인용했다 |
| 자이로 바이어스를 부팅 때 재면 된다 | 램프인 중에 재서 −0.0203 rad/s 가 나왔고 35.9 초에 **+41.7° 허깨비 yaw** 를 만들었다 | 추정을 통째로 제거 |

그래서 규칙이 생겼다: **리워드 항을 새로 넣으면 학습 전에 원시값을 재고, 특히
정지 명령에서 0 인지 본다** (`scripts/diag/probe_terms.py`).

### 남은 갭

| | 상태 |
|---|---|
| **낙상률** | ❌ 합격선 0.5 % 를 넘은 적이 없다. v55 만 0.0 % 인데 추종을 크게 내줬다 |
| **토르소 흔들림** | 🔄 v75 로 직접 겨냥 중. `torso_ang_vel` 이 유일하게 효과가 확인된 레버다 |
| **피크 토크** | 🔄 v74 로 겨냥 중. 실효 한계 3.16 N·m 에 1.2 % 의 시간 동안 물려 있다 |
| **심-실기 발 접지 불일치** | ❓ CAD 대로면 실기도 `l_foot_side` 로 딛는다. 만약 실물에서 TPU 가 튀어나와 있다면 실기 16 mm vs 심 38.7 mm — **좌우 지지가 2.4 배 어긋난다.** 실물 발바닥 육안 확인이 다음 실험의 방향을 정한다 |
| **`path_error`** | ⏸ 관측이 실기에서 상수였다. 리워드의 32 % 를 차지하는데 정보가 없었다. VLA / optical flow 로 나중에 잡기로 미뤘다 |
| **yaw 드리프트** | 🔄 실기는 자이로 z 적분으로 보정 중 (`rl_walk --path-imu`). 정지 시 적분을 0 으로 되돌린다 |
| **실기 질량·CoM** | ❌ 안 쟀다. 질량 랜덤화 폭을 좁히려면 필요하다 |
| **실기 버스 타임아웃** | ❓ 13.9 초에 끊긴 적 있다. 전압은 11.0 V 로 정상이었어서 커넥터 접촉을 의심 중 |
| **레퍼런스 보행** | ✅ 격자 충전율 100 %, 좌우 대칭 |
| **실기 통신** | ✅ 47 Hz. ONNX 온보드 추론 0.25 ms |
| **원격 조종·관제** | ✅ WebRTC 스트림 + 키보드 대시보드. **외부 망에서도 조종·학습 확인됨** |

### 원격으로 보고 몰기

```bash
# PC: 화면은 WebRTC 로 내보내고 명령은 UDP 로 받는다
ODM_HOST_IP=<이 PC IP> ./scripts/odm play v74 --stream --key
#   http://<IP>:8011/streaming/webrtc-client
#   포트: 시그널링 49100 · 미디어 47998 · 페이지 8011

# 노트북: 터미널 대시보드로 조향 (WASD/QE, Z/X 스로틀, Space 정지)
python3 scripts/console.py --host <PC IP>
```

`--stream` 은 2026-08-20 이전에 **한 번도 뜬 적이 없었다** — `--kit_args <값>` 을
띄어 쓰면 argparse 가 `--` 로 시작하는 값을 옵션으로 오인해 죽는다. `--kit_args=`
로 붙여야 한다.

### 다음에 할 일

1. **실물 발바닥 확인** — TPU 패드가 주변 쉘보다 튀어나왔는지 들어갔는지. 5 초짜리
   육안 확인인데 다음 실험 방향을 통째로 정한다 (위 "심-실기 발 접지 불일치").
2. **v74 / v75 판정** — 포화율 ≤ 0.5 %, 일률 ≤ 8.5 W, roll속 ≤ 60 이 목표.
   **6 방향 속도 부호가 유지되는지 반드시 볼 것** — 토크 벌점과 토르소 벌점은
   둘 다 "덜 움직이는 쪽" 으로 밀어서 가만히 서 있으려 할 수 있다.
3. **실기 질량·CoM 실측** — 랜덤화 폭을 좁히는 근거.
4. **실기 버스 타임아웃** — 주행 중 로봇을 흔들며 `gain_report.py` 로 재현.
5. **`path_error` 복원** — VLA / optical flow 또는 전진·회전을 스틱으로 섞는 방식.

**알아둘 것**: 레퍼런스 궤적은 명령 속도의 **절반만** 만든다 (명령 0.15 m/s →
다리가 만드는 전진 0.075 m/s). 정책은 `tracking_lin_vel` 로 명령을 쫓으므로 그
차이를 스스로 메운다. `scripts/viz_ref_pkl.py` 로 재생하면 그만큼 발이
미끄러져 보이는데, 접촉이 없는 순수 기구학 재생이라 그렇다.

---

원본 세 리포(`Open_Duck_Mini` 로봇 설명 / `Open_Duck_Playground` MJX·Brax 학습 /
`Open_Duck_reference_motion_generator` 궤적 생성)의 로직을 하나로 합치고,
MJX 대신 **Isaac Lab(PyTorch + PhysX)** 으로 다시 구현했다.

```
 Onshape CAD ──▶ URDF + 메시 ──▶ USD ──▶ 강화학습 ──▶ 정책 ──▶ 실기
                      │                    ▲
                      └──▶ placo 보행 궤적 ─┘
                           (모방 목표)
```

---

## 5분 안에 시작하기

거의 모든 일은 **`odm` 하나**로 한다. `scripts/odm` 을 PATH 에 심볼릭 링크로 걸어두면 편하다:

```bash
ln -s "$PWD/scripts/odm" ~/bin/odm     # 복사하지 말 것 — 복사본은 리포와 어긋난다
```

| 명령 | 하는 일 |
|---|---|
| `odm train [ver] [iters] [envs]` | 학습 시작 (기본 3000 iter, 4096 env) |
| `odm watch [ver]` | 진행 상황 한 줄 |
| `odm tb` | 텐서보드 (전체 런 겹쳐보기) |
| `odm play [ver]` | 재생 — 네이티브 창. `--joystick` Xbox 패드 · `--key` 키보드 · `--stream` WebRTC |
| `odm record [ver] [초]` | 6방향 순환을 mp4 로 녹화 (`--seconds` 는 **시뮬 시간**) |
| `scripts/console.py --host <IP>` | 터미널 대시보드 — WASD/QE 조향 + 상태 계측 |
| `odm measure [ver] [iter]` | 6방향 추종·주기성·리워드 성분 측정 |
| `scripts/diag/scoreboard.py [ver…]` | 합격선 채점 — 점수·roll·낙상·**포화%·일률W·roll속** |
| `odm test` | 테스트 전부 (Isaac Sim 불필요) |
| `odm import` | Onshape → URDF → USD → 색상, 한 번에 |
| `odm teleop [--no-robot]` | 웹 슬라이더로 CAD 를 움직이고 실기와 대조 |
| `odm refgait [--vx 0.15]` | 레퍼런스 보행을 뽑아 Jetson 으로 전송 |
| `odm stop` / `odm list` | 정리 / 런·체크포인트 목록 |

> **뭐가 어디 있는지 모르겠으면 [`docs/map.md`](docs/map.md) 부터.**

---

## 요구사항

| | |
|---|---|
| **필수** | Ubuntu + NVIDIA GPU + Isaac Sim / Isaac Lab |
| 검증 환경 | Isaac Lab 0.54.3 · Isaac Sim 5.1.0 · rsl-rl 5.0.1 · torch 2.7.0+cu128 |
| macOS | **시뮬레이션·학습 불가.** 코드 작성/리뷰와 순수 Python 테스트만 |

```bash
pip install -e .
# Isaac Lab 은 이 리포에 포함되지 않는다. 별도로 설치하고 경로를 알려준다:
export ISAACLAB_PATH=~/Desktop/IsaacLab
```

일부 도구(placo/pinocchio/meshcat)는 Isaac 파이썬과 의존성이 충돌해서
`~/.odm-tools` 라는 별도 venv 를 쓴다. `./demo.sh setup` 이 만들어 준다.

---

## 로봇 사양 — 코드를 건드리기 전에 알아야 할 것

| | 값 | 진실원천 |
|---|---|---|
| 총질량 | **2.7430 kg** | `robot/robot.urdf` |
| 총질량 (big_foot) | **2.7518 kg** | `big_foot/robot.urdf` — 발 편측 +10.4 g, 몸통 −12.0 g |
| 링크 / 가동관절 | 20 / **14** | 〃 |
| 액추에이터 | Dynamixel XM430-W350 ×14 | `source/.../hardware_map.py` |
| **몸통 프레임** | **+x 앞 / +y 좌 / +z 위** | `source/.../imu_map.py` |
| 관절 순서 | `ACTUATOR_JOINT_NAMES` (14개) | `source/.../joint_order.py` |
| IMU | BNO055, i2c-7 `0x28`, 축맵 항등 | `source/.../imu_map.py` |

**관절 순서를 재정렬하지 말 것.** 이 순서가 곧 정책의 액션 벡터 레이아웃이고,
바꾸면 ONNX 내보내기와 실기 이식이 조용히 깨진다.

**`robot/robot.urdf` 의 `axis` 는 14개가 전부 `0 0 1` 이라 방향 정보가 아니다.**
`onshape-to-robot` 이 회전축을 로컬 Z 로 고정하고 실제 방향을 `<origin rpy>` 에
넣기 때문이다. 부호를 알아야 하면 `imu_map.py` 의 설명을 읽을 것.

---

## 파이프라인

### 1. CAD → 시뮬 로봇

```bash
odm import          # 백업 → 임포트 → 경고 집계 → USD 변환 → 색상 주입
```

`robot/config.json` 의 Onshape URL 만 바꿔 두면 된다.

> ⚠️ **워크스페이스(`w/`) 를 반드시 확인할 것.** 문서·엘리먼트 ID 가 같고 `w/` 만
> 다르면 URL 이 안 바뀐 것처럼 보이는데 몇 달 전 형상을 받아온다. 실제로 겪었다.
> `odm import` 는 어떤 워크스페이스를 쓰는지 먼저 찍는다.

임포트가 끝나면 로그에 **ERROR 0 · no-mass 0 · multiple-base 0** 이어야 한다.
`Multiple base links` 는 어셈블리에 **메이트 안 된 인스턴스**가 있다는 뜻이고
그 부품은 통째로 버려진다 — Part Studio 의 Boolean 으로는 안 고쳐진다.
자세한 것은 [`docs/onshape_import.md`](docs/onshape_import.md).

### 2. 레퍼런스 보행 궤적 (모방 목표)

placo 로 명령 격자 전체를 스윕해 걷는 궤적을 만들고 다항식으로 피팅한다.

```bash
python3 scripts/setup/patch_urdf_for_placo.py               # 생성기에 최신 URDF·메시 복사
./scripts/setup/gen_reference_remote.sh --height 0.193 --yaw-sweep 0.28 --out ref_h193
```

- `--height` 는 **CoM 높이**다(몸통 높이가 아니다). 질량 분포가 바뀌면 같은 값이
  다른 자세를 만든다.
- `--yaw-sweep 0.28` 을 쓸 것. 기본 격자는 0 회전 명령을 비껴간다.

### 3. 학습

```bash
odm train v33 3000 4096
odm watch v33            # 진행
odm tb                   # 텐서보드
```

버전 이름을 주면 태스크가 자동으로 붙는다. **GPU 에 Isaac Sim 을 두 개 띄우면
한쪽이 조용히 죽으므로 `odm` 이 자동으로 막는다.**

### 4. 판정

```bash
odm measure v33 1500                       # 6방향 추종
scripts/train_health.py --run <런 디렉터리>  # lr 바닥 / std 붕괴 / 리워드 정체
~/.odm-tools/bin/python scripts/diag/leg_trunk_clearance.py \
  --npz ~/odm_out/gait_v33.npz --urdf robot/robot.urdf --mesh-dir robot
```

### 5. 실기

로봇을 **매달아 놓고** 시작한다. 도구는 전부 `scripts/hw/` 에 있다.

| 도구 | 하는 일 |
|---|---|
| `joint_cal.py` | 관절 부호 확인 (대화형, TTY 필요) |
| `goto_ready.py` | READY 자세로 천천히 이동 |
| `imu_check.py` | BNO055 깨우기 / 축 판정 / 검증 |
| `export_ref_gait.py` + `play_ref_gait.py` | 레퍼런스 보행 개루프 재생 |
| `dxl_bridge.py` + `cad_teleop.py` | 웹 슬라이더 텔레옵 (명령 vs 실측 비교) |

U2D2 는 **데스크탑이 아니라 Jetson**(`parksuho@192.168.137.7`) 의 `/dev/ttyUSB0`
에 있다. 배경과 절차는 [`docs/handoff/project_hardware_bringup_2026-08-06.md`](docs/handoff/project_hardware_bringup_2026-08-06.md).

---

## 절대 어기지 말 것

- **자가충돌이 액추에이터를 부순다.** 다리↔몸통 **5 mm** 를 지킨다.
  런타임 보호는 `source/.../safety_filter.py` (CBF 필터).
- **관절 순서 재정렬 금지** (위 참조).
- **Isaac Sim 두 개 동시 실행 금지** — `odm` 이 막지만 직접 띄울 때는 주의.
- **기존 런·체크포인트 삭제 금지.** 비교 기준선이 사라진다.
- 실기 조그는 URDF 한계 안에서, **한 번에 한 관절만.**
- **랩 PC(`do@192.168.137.111`) 는 공용 장비다. 아무것도 지우지 않는다.**
- **강제 푸시 금지.** 브랜치는 `main` 하나만 쓴다.
- **리워드 항을 새로 넣으면 학습 전에 원시값을 잰다** — 특히 정지 명령에서
  0 인지. 상시로 깎이는 세금이면 리워드 예산을 조용히 먹는다
  (`scripts/diag/probe_terms.py`).

---

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/map.md`](docs/map.md) | **어디에 뭐가 있나** — 여기부터 |
| [`docs/decisions.md`](docs/decisions.md) | 설계 결정과 근거 (액추에이터 게인, env API 등) |
| [`docs/training_log.md`](docs/training_log.md) | 실험 이력 — 무엇을 왜 바꿨고 어떻게 됐는지 |
| [`docs/onshape_import.md`](docs/onshape_import.md) | CAD 임포트 절차와 함정 |
| [`docs/isaaclab_setup.md`](docs/isaaclab_setup.md) | Isaac Lab 설치 |
| [`docs/graph_conventions.md`](docs/graph_conventions.md) | 그래프 규약 |
| [`docs/versions.md`](docs/versions.md) | **버전 색인** — 각 판이 무엇을 바꿨는지 (`scripts/diag/versions.py` 가 생성) |
| [`docs/webrtc_streaming.md`](docs/webrtc_streaming.md) | 원격 스트리밍·조종 |
| [`docs/handoff/`](docs/handoff/) | 실기 브링업, 관절·IMU 캘리브레이션 |
| [`docs/reports/`](docs/reports/) | 단발 조사 보고서 (아래) |
| [`docs/hw_logs/`](docs/hw_logs/) | 실기 주행 로그 원본 (`scripts/hw/analyze_hw_log.py` 로 분석) |

### 2026-08-10~11 조사 보고서

| 문서 | 내용 |
|---|---|
| [`reference_gaps_2026-08-11.md`](docs/reports/reference_gaps_2026-08-11.md) | **레퍼런스 격자 구멍 3 가지** — 충전율 55 %, 직진 슬라이스 공백, `dy=0` 부재 |
| [`hw_v40_collapse_2026-08-10.md`](docs/reports/hw_v40_collapse_2026-08-10.md) | v40 이 0.5 초 만에 주저앉은 원인 (전류 상한 + 중력 관측) |
| [`hw_v40_followup_2026-08-10.md`](docs/reports/hw_v40_followup_2026-08-10.md) | 셋 고친 뒤에도 남은 쏠림 → **토크-속도 결합** 발견 |
| [`jetson_code_verify_2026-08-10.md`](docs/reports/jetson_code_verify_2026-08-10.md) | 배포 브리지 검증 — 13 블록 중 12 일치, 액션 이력 1 칸 불일치 |
| [`jetson_workorder_stiffness_2026-08-10.md`](docs/reports/jetson_workorder_stiffness_2026-08-10.md) | `stiffness` 37.65 vs 26.1 을 가르는 정적 부하 시험 |
| [`joint_saturation_2026-08-09.md`](docs/reports/joint_saturation_2026-08-09.md) | 관절 한계 포화 조사 |
| [`lowpass_2026-08-09.md`](docs/reports/lowpass_2026-08-09.md) | 저역통과 필터 실험 (교수님 보고) |

실험을 시작하기 전에 `docs/training_log.md` 를 훑어볼 것 — 이미 실패한 가설이
꽤 있고, 그 기록이 다음 실험의 설계 근거다.
