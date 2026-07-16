# Open Duck Mini V2 — Isaac Lab

Open Duck Mini V2 이족 보행 로봇을 위한 NVIDIA Isaac Lab 학습 파이프라인. `Open_Duck_Mini`(로봇 설명), `Open_Duck_Playground`(기존 MJX/Brax 학습 파이프라인), `Open_Duck_reference_motion_generator`(참조 동작 생성) 세 리포의 로직을 하나로 통합하고, MJX 대신 Isaac Lab(PyTorch/PhysX)으로 다시 구현한 것.

## 중요: 이 리포는 macOS에서 실행할 수 없습니다

Isaac Sim/Isaac Lab은 NVIDIA GPU + Linux(또는 Windows) 전용입니다. macOS에서는 코드 작성/리뷰/일부 순수-Python 단위테스트만 가능하고, 실제 시뮬레이션·학습은 **우분투(NVIDIA GPU) 머신에서만** 가능합니다. 아래 "Mac에서 가능한 것 / 우분투 전용"을 참고하세요.

## 설치 (우분투, Isaac Lab이 이미 설치되어 있다고 가정)

```bash
# IsaacLab은 이 리포의 의존성일 뿐, 이 리포 안에 포함되지 않습니다.
# 별도로 클론/설치된 IsaacLab 경로를 IsaacLab 공식 문서대로 준비한 뒤:
cd open_duck_mini_isaaclab
pip install -e .
```

## 사용 순서

1. **URDF → USD 변환** (우분투 전용, Isaac Sim 필요)
   ```bash
   ./scripts/convert_urdf.sh
   ```
   `assets/robot/open_duck_mini_v2/robot.urdf`를 읽어 `assets/usd/open_duck_mini_v2.usd`를 생성합니다.

2. **참조 동작(모방학습용) 생성** (Mac에서 시도 가능, 안 되면 우분투에서)
   ```bash
   ./scripts/generate_reference_motion.sh
   ```
   `reference_motion_generator/`의 Placo 기반 스윕 생성 + 다항식 피팅을 실행해 `source/open_duck_mini_isaaclab/reference_motion/data/polynomial_coefficients.pkl`을 만듭니다. 물리 시뮬레이터와 무관한 순수 기구학 계산이라 Isaac Sim이 필요 없습니다 (단, `placo` 패키지가 해당 플랫폼에 설치되어야 함).

3. **학습** (우분투 전용 — IsaacLab 자체의 rsl_rl train.py를 그대로 감싼 래퍼. 별도의 train.py를 이 리포에서 재구현하지 않음 — docs/decisions.md 참고)
   ```bash
   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/train.sh --num_envs 256
   ```

4. **재생/평가 + ONNX 자동 내보내기** (우분투 전용)
   ```bash
   ISAACLAB_PATH=/path/to/IsaacLab ./scripts/play.sh --checkpoint <path/to/model.pt>
   ```
   IsaacLab의 `play.py`가 실행될 때마다 `<checkpoint_dir>/exported/policy.onnx`를 자동으로 함께 내보냅니다 — 별도의 export 스크립트가 필요 없습니다.

5. **ONNX 정책 검증** (Mac에서 실행 가능 — 순수 MuJoCo, Isaac Sim 불필요)
   ```bash
   ./scripts/mujoco_infer.py --onnx <path/to/exported/policy.onnx>
   ```
   `assets/robot/open_duck_mini_v2/playground_mjcf/scene_flat_terrain.xml`(Playground의 실제 학습용 MJCF를 그대로 복사해온 것)을 CPU MuJoCo로 로드해서 101차원 관측 조립 → ONNX 추론 → 14차원 액션 적용을 실제로 몇 스텝 돌려보고 NaN/비정상 액션이 없는지 확인합니다.

## Mac에서 가능한 것 / 우분투 전용

**Mac에서 완결 가능**
- 이 리포의 모든 Python 소스 작성/리뷰 (`source/open_duck_mini_isaaclab/**`)
- `python3 tests/test_poly_reference_motion_cpu.py`, `python3 tests/test_reward_leg_index_alignment.py` — 둘 다 순수 torch/Python이라 CPU에서 직접 검증 가능 (Isaac Sim 불필요)
- `scripts/generate_reference_motion.sh` 시도 (`placo`가 macOS에 설치되면)
- `scripts/mujoco_infer.py` — ONNX 정책이 나온 뒤 순수 MuJoCo로 재생 검증 (`pip install -e '.[sim2real-check]'`)

**우분투(Isaac Sim) 전용**
- `scripts/convert_urdf.sh`
- 로봇 스폰/물리 스모크테스트, 모든 env 동작 확인
- `scripts/train.sh` / `scripts/play.sh` (IsaacLab 자체 스크립트 래퍼)

## 설계 결정 근거

`docs/decisions.md` 참고 — 액추에이터 게인, 관절 순서, env API 선택(DirectRLEnv) 등 이유가 정리되어 있습니다.
