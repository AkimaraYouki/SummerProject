# 설계 결정 기록

## 왜 IsaacLab을 포크하지 않고 외부 확장으로 만들었나

이전 시도(`~/Desktop/robot make/IsaacLab`)는 IsaacLab 전체를 클론해서 그 소스트리 안에 커스텀 파일을 얹었다. 이러면 IsaacLab을 업데이트할 때마다 충돌 위험이 있고, 리포를 하나로 관리하려는 목적과도 맞지 않는다. 이 리포는 `pip install -e .`로 설치되는 독립 패키지이고, IsaacLab은 별도로 존재하는 의존성으로만 참조한다. IsaacLab 자체는 절대 수정하지 않는다.

같은 이유로 `scripts/train.sh`/`scripts/play.sh`도 IsaacLab 자체의 `scripts/reinforcement_learning/rsl_rl/{train,play}.py`를 직접 재구현하지 않고 그대로 감싸는 얇은 wrapper로만 작성했다. 그 스크립트들은 AppLauncher/Hydra 설정, rsl_rl 버전별 호환 처리(`handle_deprecated_rsl_rl_cfg`), 체크포인트 탐색, 비디오 녹화 등 상당한 분량의 로직을 이미 잘 유지보수된 형태로 제공하고 있어서, 이걸 이 리포 안에 복사해서 들고 있으면 IsaacLab이 업데이트될 때마다 뒤처진 사본이 될 뿐이다. `play.py`는 실행할 때마다 `runner.export_policy_to_onnx(...)`를 자동으로 호출해서 체크포인트 옆에 `exported/policy.onnx`를 만들어주므로, 별도의 `export_onnx.py`도 이 리포에 없다.

## 왜 ManagerBasedRLEnv가 아니라 DirectRLEnv인가

포팅 대상인 Open_Duck_Playground의 `Joystick(mjx_env.MjxEnv)`은 이미 `reset()`/`step()`/`_get_obs()`/`_get_reward()` 한 덩어리로 짜여 있고, 액션/IMU 딜레이 히스토리, push 스케줄링, imitation phase 카운터를 인스턴스 상태로 들고 다닌다. `reward_imitation`처럼 기준동작·관절 서브셋·접촉·명령을 한꺼번에 다루는 보상은 매니저 기반의 독립 reward term으로 쪼개면 슬라이스 로직이 여러 곳에 중복되고 어긋나기 쉽다. `DirectRLEnv`는 이 구조를 거의 그대로 옮길 수 있다. 단, 도메인 랜덤화는 `DirectRLEnv`에도 붙는 IsaacLab의 `EventManager`/`EventTermCfg`를 그대로 재사용한다 — 코어 로직은 충실도, 랜덤화는 재사용성 위주로 나눠 가져간 것.

## 액추에이터 게인 — 어떤 수치를 썼고 왜

Open_Duck_Playground의 `xmls/open_duck_mini_v2.xml`에 있는 **활성(activated) sts3215 class 값**을 사용한다. 이게 현재 실제로 학습에 쓰이는 정책이 튜닝된 기준이기 때문이다. 후보로 있었던 다른 값들(모두 사용 안 함):
- `~/Desktop/robot make/IsaacLab`의 XM430 값 (armature=0.01432 등) — 실제 로봇 하드웨어(Feetech STS3215)가 아니라 랩에서 실험 중이던 다른 모터(Dynamixel XM430) 기준이라 부적합.
- `mini_bdx/robots/open_duck_mini_v2/robot.xml`의 대안 값 (kp=9.5, damping=1.0 등) — Playground가 실제로 학습에 쓰는 값이 아님.

사용한 값:
```
stiffness (kp)   = 13.37
damping          = 0.56
armature         = 0.027
friction         = 0.068   # frictionloss
effort_limit     = 3.23    # forcerange
velocity_limit   = 5.24    # max_motor_velocity (joystick.py default_config)
```

## 관절 순서 (14개 구동 관절)

`left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle`

Playground `xmls/open_duck_mini_v2.xml`의 `<actuator>` 블록 순서를 그대로 따름. 이 순서가 obs/action/reward 전체에서 기준이 된다 — 절대 재배열하지 말 것 (재배열하면 이미 존재하는 mujoco_infer.py 기반 sim2real 검증 및 향후 실기 배포 스크립트와 어긋남).

## reward_imitation의 관절 서브셋 인덱스

원본 Playground 코드 자체에 `# TODO double check if the slices are correct`라는 주석이 남아있을 정도로 불안정한 부분. 참조 프레임(16관절, 안테나 2개 포함)과 실제 액추에이터 배열(14관절, 안테나 없음)의 길이가 다른데도 우연히 "왼다리 먼저, 오른다리 마지막" 구조가 맞아떨어져서 동작한다.

- `REF_LEG_JOINT_IDX = [0,1,2,3,4, 11,12,13,14,15]` — 16차원 참조 프레임에서 다리 10개만 추출 (5~10번은 머리+안테나, 제외).
- `ACT_LEG_JOINT_IDX = [0,1,2,3,4, 9,10,11,12,13]` — 14차원 실제 액추에이터 배열에서 다리 10개만 추출 (5~8번은 머리, 제외. 안테나는 애초에 액추에이터가 아니라서 배열에 없음).

두 리스트는 반드시 "왼쪽 hip_yaw~ankle 5개 + 오른쪽 hip_yaw~ankle 5개, 같은 순서"를 가리켜야 한다. `tests/test_reward_leg_index_alignment.py`가 이걸 정적으로 검증한다.
