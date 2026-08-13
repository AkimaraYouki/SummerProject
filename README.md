# Open Duck Mini V2 — Isaac Lab

이족 보행 로봇 **Open Duck Mini V2** 의 전체 파이프라인. CAD 에서 로봇을 뽑아
시뮬레이터에 세우고, 모방할 보행 궤적을 만들고, 강화학습으로 걷게 한 뒤,
실제 하드웨어에 옮긴다.

---

## 📍 지금 상황 (2026-08-13)

### 🎉 실기가 자기 발로 걷는다

2026-08-13, 처음으로. 며칠 동안 "앞으로 꼬꾸라진다" 던 것의 원인은 정책도
게인도 무릎도 아니라 **전원**이었다.

전원은 USB-PD 어댑터의 12 V 프로파일(보통 3 A / 36 W)이었는데, 보행 중 다리
10 축 합계 전류가 그걸 넘는다. 버스 전압이 8.6 V 까지 무너졌고 — XM430 의 최소
동작전압은 9.5 V 다 — 그 아래에서는 듀티를 100 % 로 밀어도 전류가 안 흐른다.
전원을 바꾸고 **다른 것은 하나도 안 바꾸고** 다시 돌린 결과:

| | PD 어댑터 | 전원 교체 후 |
|---|---|---|
| 버스 전압 최저 | 8.6 V | **11.2 V** |
| 다리 합계 전류 최대 | 5.22 A | **10.12 A** |
| 왼무릎 평균 추종오차 | +22.9° | **+2.7°** |
| 접지 전환 | 1.56 /s | **8.26 /s** |
| 몸통 pitch 범위 | −33 ~ +34° | −0.4 ~ +16.9° |

**5.22 A 는 로봇의 수요가 아니라 어댑터의 상한이었다.** 부하 쪽 계측만 보고
"이 정도면 공급이 충분하다" 고 판단하면 안 된다는 사례로 남긴다.

그리고 며칠 쫓던 **"왼무릎 하드웨어 고장" 가설은 기각**됐다. 전원을 고치자
좌우 추종 RMS 가 9.3° vs 8.4° 로 대칭이 됐다. 주파수 응답과 `full_check` 가
두 번 무죄 판정했던 것이 옳았고, 그걸 뒤집으려 한 쪽이 틀렸다.

### 심과 실기를 같은 자로 잰다

`scripts/diag/track_stats.py` 가 심(`play_fixed_cmd --track-csv`)과
실기(`rl_walk.py` 의 `~/rl_walk_log.csv`)를 **같은 열 이름·같은 정의**로 읽어
관절추종을 나란히 놓는다. 정의가 조금이라도 다르면 갭인지 측정 방법 차이인지
못 가린다 — 이 프로젝트에서 이미 여러 번 그걸로 오진했다.

vx = +0.15 구간, v46 기준:

| 관절 | 심 이득 | 실기 이득 | 심 지연 | 실기 지연 | 심 오프셋 | 실기 오프셋 |
|---|---|---|---|---|---|---|
| L.knee | 0.69 | **0.76** | 40 ms | 64 ms | −2.48° | −3.06° |
| R.knee | 0.71 | **0.83** | 20 ms | 64 ms | +1.19° | +1.15° |
| L.hip_roll | 0.70 | 0.86 | 20 ms | 43 ms | +1.21° | +1.15° |

**실기가 심보다 잘 따라간다.** 무릎이 명령보다 25 % 덜 움직이는 것은 하드웨어
부족이 아니라 심에서도 똑같이 일어나는 물리(체중)다. 중력에 눌려 처지는
오프셋까지 재현된다 — 오늘 맞춘 액추에이터 모델이 실제로 들어맞았다는 뜻이다.
**그러므로 P 게인을 더 올리면 안 된다.** 실기가 심보다 뻣뻣해져 반대 방향
갭이 생긴다.

### 오늘 맞춘 배포 계약

| 항목 | 전 | 후 | 근거 |
|---|---|---|---|
| `rl_walk --current` | 350 틱 (1.32 N·m) | **700 틱 (3.16 N·m)** | 심 `effort_limit` 과 같은 값 |
| 다리 Position P Gain | 800 (21.5 N·m/rad) | **1402 (37.65)** | 심 `stiffness` 와 같은 값 |
| `goto_ready` `TICK_RAD` | `4.0π/4096` (2 배 틀림) | **`2.0π/4096`** | XM430 은 4096 틱이 1 회전 |
| READY 자세 출처 | 하드코딩 (08-08 자) | **`policy.meta.json`** | v46 과 최대 25.6° 어긋나 있었다 |

P/D 게인은 **RAM** 이라 전원 재인가·reboot **그리고 모드 전환** 때마다 되돌아
간다. 800/0/4700 은 아무도 쓴 값이 아니라 모드 5(전류기반 위치제어)로 바꿀 때
펌웨어가 넣는 값이다 (모드 3/4 는 D=0). 그래서 게인은 `arm()` 이 모드를 쓴
**뒤에** 걸어야 하고, 순서를 뒤집으면 조용히 무효가 된다.

### 남은 갭

| | 상태 |
|---|---|
| **제어 지연** | ⚠️ 실기가 심보다 정확히 **한 스텝(21 ms)** 느리다 (43 vs 20 ms). 전 축 일관 → **v47** 이 `action_delay` 를 2~3 스텝으로 맞춤 |
| **덜컹거림** | ⚠️ roll 진폭이 심 20.98° / 실기 20.39° 로 **같다**. 배포 문제가 아니라 학습된 걸음걸이다. 정책이 `max_motor_velocity`(4.82 rad/s)를 상시 물어 착지 피크가 중력의 2.4 배 → **v48/v49** |
| **yaw 드리프트** | ❌ 명령 없이 8 초에 191° 왼쪽으로 돈다. `path_error` 가 `[0,1,0]` 으로 고정돼 정책이 방향 오차를 **관측하지 못한다**. 단발 지지 구간에서 심의 10 배로 도는 것으로 보아 발 미끄러짐도 섞여 있다 |
| **오도메트리** | ❌ 없다. 위 두 문제의 공통 뿌리 |
| **레퍼런스 보행** | ✅ 격자 충전율 100 %, 좌우 대칭 (hip_pitch +41 % → −2.9 %) |
| **실기 통신** | ✅ 47 Hz. ONNX 온보드 추론 0.25 ms |
| **패드 조종** | ✅ 젯슨 직결 (`bt_pad` / `joy_local` / `pad_ctl`, `rl_walk --joy`) |

### 다음에 할 일

1. **`path_error` 복원** — 자이로 z 를 적분해 실제 방향 오차를 넣는다. yaw
   드리프트의 유일한 근본 해결이고, 30 초 런에는 적분 드리프트가 문제 안 된다.
2. **v48 / v49 판정** — `action_rate` 를 v40 이전(−0.5)으로 되돌린 것만으로
   목표각 변화 속도 p95 가 클램프(4.82)에서 내려오는지. 안 내려오면 v49 처럼
   `max_motor_velocity` 를 직접 낮춘다.
3. **지면 마찰 확인** — 고무 매트 위에서 같은 런을 돌려 yaw 드리프트가 줄면
   마찰이 원인이다. 2 분이면 갈린다.
4. **`odm gap`** — 심과 실기에 `--cmd-udp` 로 같은 명령을 동시에 주고 끝나면
   `track_stats --vs` 리포트까지. 배관은 이미 다 있고 오케스트레이션만 남았다.
5. **실기 질량·CoM 실측** — 심 USD 는 2.6404 kg, 문서는 2.7430 kg. CoM 은 y 로
   +26.1 mm 치우쳐 있는데 어떤 랜덤화도 이걸 안 덮는다.

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
| `odm play [ver]` | 재생 — 네이티브 창. `--joystick` 으로 Xbox 패드 조종 |
| `odm measure [ver] [iter]` | 6방향 추종·주기성·리워드 성분 측정 |
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
| 총질량 | **2.7140 kg** | `robot/robot.urdf` |
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
