# 설계 결정 기록

## 왜 IsaacLab을 포크하지 않고 외부 확장으로 만들었나

이전 시도(`~/Desktop/robot make/IsaacLab`)는 IsaacLab 전체를 클론해서 그 소스트리 안에 커스텀 파일을 얹었다. 이러면 IsaacLab을 업데이트할 때마다 충돌 위험이 있고, 리포를 하나로 관리하려는 목적과도 맞지 않는다. 이 리포는 `pip install -e .`로 설치되는 독립 패키지이고, IsaacLab은 별도로 존재하는 의존성으로만 참조한다. IsaacLab 자체는 절대 수정하지 않는다.

같은 이유로 `scripts/train.sh`/`scripts/play.sh`도 IsaacLab 자체의 `scripts/reinforcement_learning/rsl_rl/{train,play}.py`를 직접 재구현하지 않고 그대로 감싸는 얇은 wrapper로만 작성했다. 그 스크립트들은 AppLauncher/Hydra 설정, rsl_rl 버전별 호환 처리(`handle_deprecated_rsl_rl_cfg`), 체크포인트 탐색, 비디오 녹화 등 상당한 분량의 로직을 이미 잘 유지보수된 형태로 제공하고 있어서, 이걸 이 리포 안에 복사해서 들고 있으면 IsaacLab이 업데이트될 때마다 뒤처진 사본이 될 뿐이다. `play.py`는 실행할 때마다 `runner.export_policy_to_onnx(...)`를 자동으로 호출해서 체크포인트 옆에 `exported/policy.onnx`를 만들어주므로, 별도의 `export_onnx.py`도 이 리포에 없다.

## 왜 ManagerBasedRLEnv가 아니라 DirectRLEnv인가

포팅 대상인 Open_Duck_Playground의 `Joystick(mjx_env.MjxEnv)`은 이미 `reset()`/`step()`/`_get_obs()`/`_get_reward()` 한 덩어리로 짜여 있고, 액션/IMU 딜레이 히스토리, push 스케줄링, imitation phase 카운터를 인스턴스 상태로 들고 다닌다. `reward_imitation`처럼 기준동작·관절 서브셋·접촉·명령을 한꺼번에 다루는 보상은 매니저 기반의 독립 reward term으로 쪼개면 슬라이스 로직이 여러 곳에 중복되고 어긋나기 쉽다. `DirectRLEnv`는 이 구조를 거의 그대로 옮길 수 있다. 단, 도메인 랜덤화는 `DirectRLEnv`에도 붙는 IsaacLab의 `EventManager`/`EventTermCfg`를 그대로 재사용한다 — 코어 로직은 충실도, 랜덤화는 재사용성 위주로 나눠 가져간 것.

## 액추에이터 게인 — 어떤 수치를 썼고 왜

**(2026-07-25 정정: 이 항목은 원래 Playground의 STS3215 값을 쓴다고 적혀 있었으나, 이후 실제
로봇 재설계가 확정되면서 완전히 뒤집혔다. 아래는 현재 유효한 버전.)**

이 로봇의 실제 액추에이터는 **Dynamixel XM430**으로 확정됐다(OnShape CAD에 `xm430_assem`
서브어셈블리로 포함, 12.0V 확정 — `docs/onshape_import.md` 참고). 즉 원래 Open_Duck_Playground가
튜닝 기준으로 삼았던 Feetech STS3215도, 한때 "랩에서 실험 중이던 다른 모터라 부적합"이라고
배제했던 `robot make/IsaacLab`의 XM430 값도 — 이제는 XM430 쪽이 맞고 STS3215 쪽이 이 로봇에는
안 맞는 값이다.

**출처가 두 갈래로 나뉜다:**

1. **BAM 실측** (`~/Desktop/robot make/bam_xm430_params/m4.json`, "m4" 모델) — 실제 이 XM430
   개체를 특성화(characterization)해서 얻은 값. 모터 개체차·기어박스 마찰까지 반영돼 있어서
   데이터시트 이상적 스펙보다 정확:
   ```
   armature = 0.01432   # BAM "armature" 직접
   friction = 0.0761     # BAM "friction_base" (Stribeck 모델의 base항만 — Isaac Lab의
                          # 단일 스칼라 friction엔 stribeck/load 항이 대응할 자리가 없음)
   ```
2. **로보티즈 데이터시트** (XM430-W350, 12.0V 확정 — BAM엔 토크/속도 한계값이 없어서):
   ```
   effort_limit_sim   = 4.1    # N*m, stall torque @ 12.0V/2.3A
   velocity_limit_sim = 4.82   # rad/s, no-load speed @ 12.0V(46rpm)
   ```

**stiffness/damping (2026-07-26 확정)** — 처음엔 "BAM은 위치제어 게인을 측정하지 않으니
결정 불가"라고 잘못 결론 냈었다(실제로는 BAM 자체 소스코드 `bam/bam/actuator.py`의
`VoltageControlledActuator.to_mujoco()`에 정확히 이 변환 공식이 이미 있었음 — JSON 출력
필드만 보고 판단해서 놓친 것, 이후로는 도구의 실제 소스코드를 먼저 확인할 것). BAM이 실측한
kt/R/friction_viscous와 서보의 실제 Position P Gain 레지스터값을 결합해서 물리적 PD 게인으로
변환하는 공식:
```
stiffness = error_gain * kp_register * vin * max_pwm * kt / R
          = (1/128)    * 800.0       * 12.0 * 1.0     * 1.0057156607538362 / 2.0032635699956667
          = 37.6528958476996  ->  37.65

damping   = friction_viscous + kt**2 / R
          = 0.8470782260272692 + 1.0057156607538362**2 / 2.0032635699956667
          = 1.3519863197174637  ->  1.352
          # friction_viscous 단독(이전 damping=0.847)은 back-EMF 항(kt**2/R)이 빠진 것 —
          # damping도 같이 틀렸던 것이었음
```
- `error_gain=1/128`: XM430 Position P Gain 레지스터 변환 공식(로보티즈 데이터시트와 동일),
  `bam/bam/dynamixel/actuator.py`의 `XM430Actuator`에 하드코딩됨
- `kp_register=800.0`: 공장 기본 Position P Gain(`XM430Actuator` 생성자 기본값과 동일) —
  **2026-07-26 실기 확인**: Dynamixel Wizard로 실제 하드웨어 컨트롤 테이블을 직접 읽어
  `Position P Gain=800(0x0320)` 그대로 확인됨(`Velocity P/I Gain=100/1920`도 데이터시트
  기본값과 일치, `Voltage=12.20V`도 가정한 12.0V와 거의 일치) — 더 이상 가정이 아니라
  실측 확인된 값.
- `vin=12.0`: 확정 동작전압, `max_pwm=1.0`: BAM 기본값(듀티 상한 미설정)

교차검증: IsaacLab GitHub Discussion #2627(Robotis OP3용 XM430-W350 PD 튜닝, 미해결
스레드)에서 커뮤니티가 시도한 값은 stiffness=45.0/damping=1.5 — BAM 유도값과 같은
자릿수라 서로 뒷받침되지만, 그 스레드 값은 검증된 게 아니고 다른 로봇 사례이므로 참고만.
우리 값은 실제 우리 개체를 실측한 BAM 피팅에서 나온 것이라 더 신뢰도가 높음.

추가 교차검증(2026-07-26): Disney Research의 BD-X 로봇 논문(`docs/`에서 다룬 BD-X
비교 아티클 참고)에 실린 Dynamixel XH540-V150(다른 모델, 우리 XM430-W350과는 다름)의
실측 kP=5.0은 우리 유도값(37.65)보다 약 7.5배 작다 — kp_register=800 자체는 이제
실기로 확인됐으니, 남은 차이는 kt/R(BAM 피팅) 쪽에 있을 가능성이 높음. 실기 스텝응답
검증 전까지는 미해결로 남겨둠.

**TODO (사용자, 2026-07-26 결정)**: XH540 값(5.0)을 빌려쓰지 않고 지금 값(37.65) 그대로
유지하기로 함 — 다른 액추에이터(기어비 150:1 vs XM430의 353.5:1) 값을 갖다 붙이는 것도
검증된 게 아니라서 실익이 없다고 판단. **사용자가 내일(2026-07-27 예상) XM430 실물을
BAM으로 직접 재측정할 예정** — kt/R 재피팅해서 stiffness/damping 값을 갱신할 것.

## 관절 순서 (14개 구동 관절)

`left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle`

Playground `xmls/open_duck_mini_v2.xml`의 `<actuator>` 블록 순서를 그대로 따름. 이 순서가 obs/action/reward 전체에서 기준이 된다 — 절대 재배열하지 말 것 (재배열하면 이미 존재하는 mujoco_infer.py 기반 sim2real 검증 및 향후 실기 배포 스크립트와 어긋남).

## reward_imitation의 관절 서브셋 인덱스

원본 Playground 코드 자체에 `# TODO double check if the slices are correct`라는 주석이 남아있을 정도로 불안정한 부분이었다. 원래는 참조 프레임(16관절, 안테나 2개 포함)과 실제 액추에이터 배열(14관절, 안테나 없음)의 길이가 달랐는데, **2026-07-26 안테나 참조를 파이프라인 전체에서 제거**하면서(이 로봇엔 안테나 자체가 없음 — `robot/robot.urdf`에 "antenna" 문자열이 0번 등장) 참조 프레임도 14관절로 줄어 **두 배열이 이제 완전히 동일**해졌다.

- `REF_LEG_JOINT_IDX = [0,1,2,3,4, 9,10,11,12,13]` — 14차원 참조 프레임에서 다리 10개만 추출 (5~8번은 머리, 제외).
- `ACT_LEG_JOINT_IDX = [0,1,2,3,4, 9,10,11,12,13]` — 14차원 실제 액추에이터 배열에서 다리 10개만 추출 (5~8번은 머리, 제외).

두 리스트가 이제 값 자체가 같지만(우연이 아니라 REF_JOINT_NAMES==ACTUATOR_JOINT_NAMES가 됐기 때문), 개념적으로는 여전히 별개("참조 pkl 레이아웃" vs "액추에이터/액션벡터 레이아웃")라 상수 자체는 분리 유지. `tests/test_reward_leg_index_alignment.py`가 이걸 정적으로 검증한다.

**연쇄 변경**: `poly_reference_motion.py`의 `REF_FRAME_DIM`도 40→36(14+14+2+3+3)으로, `rewards.py::reward_imitation`의 하드코딩된 슬라이스 인덱스(`[0:16]`→`[0:14]` 등)도 같이 바뀜. `reference_motion_generator`의 `placo_defaults.json`/`medium.json`/`fast.json`도 이미 안테나 참조 제거됨(Placo 크래시 수정과 같은 작업, `docs/training_log.md` 참고) — 이 변경들은 서로 맞물려있어서 하나만 바꾸면 인덱스가 어긋난다.
