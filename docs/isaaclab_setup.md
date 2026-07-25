# IsaacLab 최초 설치 (랩 PC, 한 번만)

`~/IsaacLab`는 클론만 돼 있고 확장 패키지가 설치 안 된 상태였다(2026-07-25 확인).
`convert_urdf.sh`/`train.sh`/`play.sh`를 처음 쓰기 전에 이 설치를 한 번 해야 한다.

## 1. conda 비활성화부터

인터랙티브 쉘에서 `.bashrc`의 conda init 블록 때문에 `base` 환경이 자동 활성화돼 있다.
`isaaclab.sh`는 `$CONDA_PREFIX`가 설정돼 있으면 **번들 파이썬 대신 conda 파이썬을 씀**
(이 랩 PC의 Isaac Sim은 conda 환경이 아니라 독립 번들 설치본이라 conda 쪽엔 `isaacsim`
모듈이 없음 → `ModuleNotFoundError: No module named 'isaacsim'`로 실패):

```bash
conda deactivate
```

## 2. 설치 실행

```bash
cd ~/IsaacLab
./isaaclab.sh --install
```

PyTorch(1.1GB) 등 큰 다운로드가 있어서 시간이 걸린다.

## ⚠️ 알려진 실패 지점: flatdict 빌드 에러

```
Getting requirements to build wheel: finished with status 'error'
ModuleNotFoundError: No module named 'pkg_resources'
```

`flatdict==4.0.1`(오래된 패키지, `setup.py`가 구식 `pkg_resources` 임포트에 의존)이 pip의
격리된 빌드 환경에서 실패한다. 격리된 빌드 환경은 최신 setuptools를 새로 받는데, 거기엔
`pkg_resources`가 없을 수 있음. **메인 환경의 setuptools는 이미 갖고 있으므로**, 격리 없이
직접 설치해서 우회:

```bash
~/IsaacLab/_isaac_sim/python.sh -m pip install flatdict==4.0.1 --no-build-isolation
```

성공하면(`Successfully installed flatdict-4.0.1`) `./isaaclab.sh --install`을 다시 실행 —
`flatdict`는 "already satisfied"로 건너뛰고 나머지 확장(`isaaclab_mimic`, `isaaclab_rl` 등)을
이어서 설치한다.

## 정상적으로 오래 걸리는 구간 (멈춘 게 아님)

- `pytransform3d`/`eigenpy` 버전 호환성 때문에 pip이 여러 버전을 내려받으며 되짚는
  (backtracking) 구간이 있다 — 수 분 걸릴 수 있음, 로그가 안 바뀌어 보여도 정상 진행 중.
- `isaaclab_rl` 설치 과정에서 torch가 2.7.0 → 2.13.0으로 재설치될 수 있음(다른 확장의
  버전 요구사항 때문) — 정상 동작.

## 설치 확인

```bash
~/IsaacLab/_isaac_sim/python.sh -m pip list | grep isaaclab
```

`isaaclab`, `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks` 5개가 다 떠야
한다.

## 검증된 결과 (2026-07-25)

설치 후 `ISAACLAB_PATH=~/IsaacLab ./scripts/convert_urdf.sh --headless`로 `robot/robot.urdf`를
`robot/usd/open_duck_mini_v2.usd`로 성공적으로 변환함. (출력 경로는 원래 최상위 `assets/usd/`였다가,
모든 임포트 결과물을 한곳에 모으기 위해 `robot/usd/`로 옮김 — `robot/`가 이제 URDF·메시·USD 전부의
캐노니컬 위치.)

**한 가지 주의(해결됨, 2026-07-25 같은 날 후속조치)**: 첫 변환 로그엔
```
The path xm430_어셈 is not a valid usd path, modifying to xm430_______
```
경고가 있었다 — USD는 경로 이름에 비-ASCII 문자(한글)를 허용 안 해서 자동으로 밑줄로
치환됐던 것. OnShape에서 `xm430_어셈` → `xm430_assem`으로 영문 리네임한 뒤 재임포트 +
재변환하니 이 경고가 완전히 사라짐(`grep "not a valid usd path"` 결과 0건). 확인된
교훈: **OnShape 서브어셈블리·파츠 이름은 반드시 영문으로 지을 것** — 한글 이름은 USD
변환 시 자동으로 밑줄 치환되어 원래 이름을 알아보기 어려워진다.

## 첫 스모크 테스트 성공 (2026-07-25)

USD 변환 후 `scripts/train.sh --num_envs 4 --max_iterations 2 --headless`로 실제 학습
파이프라인 전체를 검증했다. 이 과정에서 실행해보지 않고는 못 잡았을 버그를 5개 더
발견하고 고쳤다 — 전부 실제 실행 로그의 에러 메시지를 보고 원인을 추적한 것:

1. **body 이름 불일치** (`joint_order.py`) — `BASE_BODY_NAME="base"`가 실제 URDF엔
   없는 링크였음. IMU가 그 이름을 하드코딩 참조하고 있어서 안 고쳤으면 env 초기화에서
   바로 죽었을 것. `ROOT_BODY_NAME`("trunk_assembly")을 쓰도록 수정.
2. **태스크 미등록** (`gymnasium.error.NameNotFound`) — IsaacLab의 `train.py`/`play.py`는
   외부 태스크 패키지를 전혀 모른다. `scripts/_isaaclab_launch.py` 셔틀을 만들어서
   `open_duck_mini_isaaclab`을 먼저 임포트(gym.register 부작용)한 뒤 실제 스크립트에
   위임하도록 `train.sh`/`play.sh`를 고침.
3. **rsl_rl API 불일치** (`ImportError: cannot import name 'RslRlMLPModelCfg'`) —
   이전 세션이 "최신 API"라고 가정했던 것과 실제 설치된 `isaaclab_rl==0.2.0`이 달랐음.
   설치된 `rl_cfg.py`를 직접 읽어서 실제 API(`RslRlPpoActorCriticCfg`)에 맞게
   `agents/rsl_rl_ppo_cfg.py` 재작성.
4. **torch/CUDA 버전 불일치** (`RuntimeError: driver too old`) — `isaaclab_rl` 설치 중
   torch가 2.13.0+cu130으로 자동 업그레이드됐는데 드라이버는 CUDA 12.8까지만 지원.
   cu128과 호환되는 최신 버전(2.11.0+cu128)으로 재설치, torchvision도 같이 맞춤
   (ABI 불일치로 `RuntimeError: operator torchvision::nms does not exist` 추가 발생 →
   torchvision 재설치로 해결).
5. **HOME_JOINT_POS 관절 한계 초과** — `right_hip_pitch`/`right_knee`가 이 URDF의
   실제 `<limit>` 범위를 벗어나 있었음(좌우 관절 축 부호 관례가 원본 Playground와
   다름). 두 값 부호 반전으로 해결 — Isaac Sim에서 자세가 실제로 대칭적인지는
   아직 육안 확인 안 함.

**결과**: 에러 0건, `model_0.pt`/`model_1.pt` 체크포인트 저장 확인
(`logs/rsl_rl/open_duck_mini_v2_joystick/2026-07-25_17-41-30/`). 파이프라인 전체
(OnShape 임포트 → URDF → USD → Isaac Lab env → RSL-RL 학습)가 처음으로 끝까지
돌아간 것을 확인했다.
