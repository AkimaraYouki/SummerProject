# Open Duck Mini V2 — Isaac Lab

이족 보행 로봇 **Open Duck Mini V2** 의 전체 파이프라인. CAD 에서 로봇을 뽑아
시뮬레이터에 세우고, 모방할 보행 궤적을 만들고, 강화학습으로 걷게 한 뒤,
실제 하드웨어에 옮긴다.

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

실험을 시작하기 전에 `docs/training_log.md` 를 훑어볼 것 — 이미 실패한 가설이
꽤 있고, 그 기록이 다음 실험의 설계 근거다.
