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
실기로 확인됐으니, 남은 차이는 kt/R(BAM 피팅) 쪽에 있을 가능성이 높음.

추가 교차검증(2026-07-27, 웹서치): IsaacLab GitHub Discussion #2627(같은 XM430-W350으로
Robotis OP3 튜닝한 사례) — stiffness=45.0/damping=1.5. 우리 damping(1.352)과 거의 일치,
stiffness만 ~20% 차이. BAM 원본 저장소(Rhoban/bam)엔 애초에 XM430 피팅 데이터가 없음
(MX-64/MX-106/eRob80만 있음) — `m4.json`은 팀 자체 커스텀 측정치.

**TODO — 완료/불필요로 판정 (사용자, 2026-07-27)**: 위 교차검증들을 근거로 지금 값
(stiffness=37.65, damping=1.352)을 "쓸만함"으로 최종 판정, **BAM 재측정 안 하기로 결정**.
XH540 값(5.0)을 빌려쓰지 않고 유지한다는 2026-07-26 결정과 함께 이 항목은 종결.
`imitation_v2` twitching 실패의 원인 조사는 리워드 가중치(`alive_scale`/`w_joint_pos`)
스윕 쪽으로 집중.

## 좌우 비대칭 발견 — OnShape CAD 레벨 이슈 (2026-07-26, 학습 재개 전 BLOCKER)

재생성된 117개 레퍼런스 모션의 단일 걸음 검증(verify_gait.py)에서 사용자가 지적:
`left_knee` ROM [1.497, 2.692]는 홈포즈(1.368)보다 더 굽혀진 쪽에, `right_knee` ROM
[-0.913, +0.228]는 홈포즈(-1.379)보다 훨씬 덜 굽혀진 쪽에 있어 좌우가 전혀 미러링되지
않음. 게다가 `right_knee` 실측 최대값(+0.228)이 URDF에 정의된 관절 상한(`upper≈0`)을
0.228rad(13°) 초과 — 물리적으로 불가능한 값이 레퍼런스 궤적에 포함돼 있었음.

사용자가 OnShape을 직접 확인해 3가지 CAD 레벨 원인을 지목:
1. **질량 비대칭** — `left_roll_to_pitch_assembly`(105.16g) vs
   `right_roll_to_pitch_assembly`(121.62g), diff +16.47g(~15.7%). URDF 파싱으로 직접
   확인됨. Placo는 ZMP 기반으로 걸음을 생성하므로 무게중심 계산에 이 비대칭이 그대로
   들어가 좌우 다리가 다르게 걷도록 유도했을 가능성.
   **사용자 재확인**: OnShape 상에서 두 부품의 밀도/부피/겉넓이가 전부 동일 — 즉
   CAD 형상 자체는 대칭인데 질량만 다름. 따라서 설계를 다시 할 필요는 없고,
   **onshape-to-robot 익스포트 과정에서 한쪽에만 숨겨진 파스너가 질량 계산에
   포함됐거나, 재질/캐싱 오류로 질량이 잘못 산출됐을 가능성이 유력** — 재익스포트
   시 이 부분부터 확인.
2. **`right_knee` 관절 회전 방향(CW/CCW) 반전** — OnShape 메이트 정의에서 각도 표시는
   같지만 시계/반시계 방향이 `left_knee`와 반대. URDF 레벨에서는 검증 불가(OnShape
   접근 필요), 위 관절 한계 초과 현상과 정합적.
3. **억제된(suppressed) 프레임 상태로 어셈블리 임포트 + 로봇이 비직립 상태로 임포트됨**
   — `HOME_JOINT_POS`가 실제 물리적 중립 자세를 정확히 반영하지 못했을 가능성. 이것도
   OnShape 쪽에서만 확인 가능.

**결정 (사용자, 2026-07-26)**: 위 3가지를 OnShape에서 고치고 URDF를 재익스포트할
때까지 Stage 2 재학습을 보류. 지금 상태(속도필터는 고쳤지만 좌우 비대칭은 남아있음)로
학습을 돌려도 비슷한 실패(twitching/reward-hacking)가 재현될 가능성이 높다고 판단.
재익스포트되면 `scripts/patch_urdf_for_placo.py`부터 다시 돌려 궤적 재생성 → 검증 →
재학습 순서로 진행.

**재임포트 결과 (2026-07-26, 랩PC `onshape-to-robot .` 재실행)**: 사용자가 프레임
억제 해제, 파스너 추가(openduck 참고), 직립 자세로 수정, `right_knee` 관절 방향 수정을
OnShape에서 완료 후 재익스포트.
- ✅ **무릎 관절 방향 — 수정 확인됨**: 재임포트 후 `left_knee`/`right_knee` 둘 다
  `lower=-1.5708, upper=1.5708`로 완전히 동일 (이전엔 left `[0,π]` vs right `[-π,0]`로
  비대칭이었음). 이게 관절 한계 초과(+0.228rad) 현상의 직접 원인이었을 가능성이 큼.
- ✅ **프레임 억제 해제/파스너/직립 — 정상 반영됨**: `Multiple base link` 경고 없음,
  관절 14개 그대로(`ACTUATOR_JOINT_NAMES`와 이름 일치), 새 `imu_frame` 고정관절 추가됨.
  총질량 1.9809kg → 2.1219kg로 증가(파스너 추가분 반영으로 추정, 정상).
  새 "Frame" 파츠에 대해 "no mass" 경고 4건 발생 — 재질 미할당으로 추정, 무해할
  가능성 높지만 추후 확인 필요.
- ⏳ **질량 비대칭 — 미해결**: `left_roll_to_pitch_assembly`(105.16g) vs
  `right_roll_to_pitch_assembly`(121.62g), diff +16.47g — 재임포트 후에도 완전히
  동일한 수치로 그대로 남아있음(이번 OnShape 수정 범위에 포함 안 됐음, 예상된 결과).
  사용자가 CAD에서 밀도를 직접 맞추기로 함(2026-07-26) — 완료 후 다시 재임포트해서
  확인할 것. URDF를 직접 임시 패치하는 방안(어셈블리 총질량을 121.62g로 맞추고
  관성텐서 비례 스케일)도 검토했으나, CAD에서 근본 수정하는 쪽으로 결정.

**최종 해결 (2026-07-26, 커밋 `0ffa3d5`/`abc61fd`)**: 질량 CAD 수정 후에도 순수 직진
걸음에서 여전히 좌우 비대칭(`left_knee`/`right_knee`가 아예 겹치지 않는 범위)이 남아있어서
추가로 파고든 결과, **`patch_urdf_for_placo.py`가 주입하던 `left_foot_frame`/
`right_foot_frame` 오프셋이 좌우 완전히 동일한 값(미러링 안 됨)이었던 게 진짜 원인**이었음.
사용자가 OnShape에 `trunk_frame`/`left_foot_frame`/`right_foot_frame`/`head_frame`이라는
이름의 Fastened 메이트를 직접 만들어서(업스트림 GitHub 구조 참고), `imu_frame`이 그랬듯
onshape-to-robot이 이 4개를 **네이티브로, 실제로 미러링된 xyz/rpy**로 뽑아내도록 만듦.
재검증 결과 `left_hip_pitch`/`right_hip_pitch` ROM 0.642/0.627, `left_knee`/`right_knee`
ROM 1.092/1.117(부호만 반대 — 정상 축 컨벤션), 발 접촉 토글 38/37 — 완전히 대칭인 보행 확인.

부수적으로 두 가지 더 정리:
- `patch_urdf_for_placo.py`는 이제 4개 프레임이 네이티브로 있는지 확인만 하는
  검증 스크립트로 축소 (하드코딩 주입은 비상 폴백으로만 남김).
- 재구조화 과정에서 발 링크 이름이 `foot_assembly`/`foot_assembly_2`로 원복돼서
  `joint_order.py`의 `LEFT_FOOT_BODY_NAME`/`RIGHT_FOOT_BODY_NAME`도 같이 갱신.
- `HOME_JOINT_POS`를 전부 0으로, `HOME_BASE_HEIGHT`를 0.15→0.193으로 갱신 (Isaac Sim에서
  zero-action PD hold로 직접 검증: steady-state 높이 0.1937~0.1938m, upright -0.9960,
  soft-limit 위반 0건, PASS).
- 속도필터로 걸러진 스파스 그리드(6×4×10=240 중 120개 생존) 때문에 `PolyReferenceMotion`이
  `KeyError`로 크래시하던 버그도 발견/수정 — 빠진 grid cell은 최근접 실제 기록으로 채우도록
  변경 (`poly_reference_motion.py`).

**다음 스텝**: Stage 2 재학습(`imitation_v2`) 시작 준비 완료.

## 관절 순서 (14개 구동 관절)

`left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle`

Playground `xmls/open_duck_mini_v2.xml`의 `<actuator>` 블록 순서를 그대로 따름. 이 순서가 obs/action/reward 전체에서 기준이 된다 — 절대 재배열하지 말 것 (재배열하면 이미 존재하는 mujoco_infer.py 기반 sim2real 검증 및 향후 실기 배포 스크립트와 어긋남).

## reward_imitation의 관절 서브셋 인덱스

원본 Playground 코드 자체에 `# TODO double check if the slices are correct`라는 주석이 남아있을 정도로 불안정한 부분이었다. 원래는 참조 프레임(16관절, 안테나 2개 포함)과 실제 액추에이터 배열(14관절, 안테나 없음)의 길이가 달랐는데, **2026-07-26 안테나 참조를 파이프라인 전체에서 제거**하면서(이 로봇엔 안테나 자체가 없음 — `robot/robot.urdf`에 "antenna" 문자열이 0번 등장) 참조 프레임도 14관절로 줄어 **두 배열이 이제 완전히 동일**해졌다.

- `REF_LEG_JOINT_IDX = [0,1,2,3,4, 9,10,11,12,13]` — 14차원 참조 프레임에서 다리 10개만 추출 (5~8번은 머리, 제외).
- `ACT_LEG_JOINT_IDX = [0,1,2,3,4, 9,10,11,12,13]` — 14차원 실제 액추에이터 배열에서 다리 10개만 추출 (5~8번은 머리, 제외).

두 리스트가 이제 값 자체가 같지만(우연이 아니라 REF_JOINT_NAMES==ACTUATOR_JOINT_NAMES가 됐기 때문), 개념적으로는 여전히 별개("참조 pkl 레이아웃" vs "액추에이터/액션벡터 레이아웃")라 상수 자체는 분리 유지. `tests/test_reward_leg_index_alignment.py`가 이걸 정적으로 검증한다.

**연쇄 변경**: `poly_reference_motion.py`의 `REF_FRAME_DIM`도 40→36(14+14+2+3+3)으로, `rewards.py::reward_imitation`의 하드코딩된 슬라이스 인덱스(`[0:16]`→`[0:14]` 등)도 같이 바뀜. `reference_motion_generator`의 `placo_defaults.json`/`medium.json`/`fast.json`도 이미 안테나 참조 제거됨(Placo 크래시 수정과 같은 작업, `docs/training_log.md` 참고) — 이 변경들은 서로 맞물려있어서 하나만 바꾸면 인덱스가 어긋난다.
