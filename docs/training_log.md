# 학습 시도 기록 (성공/실패 전부)

Stage 1(순수 RL, `use_imitation=False`) 학습에서 실제로 시도한 모든 런과 그 결과, 실패한 경우 원인 분석까지 기록한다. 성공한 런만 남기면 왜 지금 설정에 도달했는지 다음에 알 수 없어서, 실패도 전부 남긴다.

---

## Run 1 — 2026-07-26 01:50, num_envs=256, alive_scale=20.0(원본)

- **설정**: `min_base_height_ratio` 없음(아직 종료조건 버그 수정 전), `alive_scale=20.0`(Playground 원본값 그대로)
- **결과**: 3000 iteration 완료. `Mean reward: 354.29`, `Mean episode length: 902.55/1000` — 학습 로그만 보면 매우 성공적으로 보임.
- **실제 확인**: WebRTC로 직접 보니 로봇이 완전히 주저앉은 채(다리가 뒤엉킨 자세) 안 움직이고 있었음. 넘어진 게 맞는데 "안 죽은 것"으로 처리되고 있었음.
- **원인**: `_get_dones()`의 유일한 종료조건이 `projected_gravity_b.z > 0`(90도 이상 완전히 뒤집힘)뿐이었음. Playground 원본(`upvector_z < 0`)을 그대로 포팅한 값이라 포팅 버그는 아니지만, 옆으로 쓰러지거나 주저앉는 건 90도를 안 넘어서 전혀 감지가 안 됨. `alive_scale=20`(생존 보너스)만으로 이 상태를 계속 버티는 게 정책 입장에서 남는 장사였음.
- **판정**: **FAIL (reward-hacked collapse)**

## 수정 1 — `min_base_height_ratio=0.6` 종료조건 추가 (커밋 `d20cdba`/`75f2101`)

베이스 높이가 `HOME_BASE_HEIGHT(0.15m)`의 60% 밑으로 떨어지면 방향과 무관하게 종료. `check_joint_stability.sh`로 스모크 테스트 통과 확인 후 재학습 진행.

## Run 2 — 2026-07-26 03:50, num_envs=2048, alive_scale=20.0, min_base_height_ratio=0.6

- **설정**: Run 1과 동일한 `alive_scale=20.0`, 새로 추가된 높이 기반 종료조건만 다름. 사용자 요청으로 컴퓨팅 자원 최대 활용 위해 `num_envs`도 256→2048로 증량(GPU 메모리 27%, 사용률 73%까지만 사용 중이라 여유 있었음).
- **결과**: 3000 iteration 완료. `Mean reward: ~320-340`, `Mean episode length: ~830-930/1000`.
- **실제 확인**: WebRTC로 iteration 2600 체크포인트 확인 — Run 1과 똑같이 몸이 뒤틀린 채 서 있지 못하는 자세. 높이 종료조건을 우회할 수 있는(0.09m 밑으로는 안 떨어지는) 또 다른 붕괴 자세를 찾아낸 것으로 보임.
- **원인 분석**: Playground 원본 설정(`alive=20.0`)을 대조해보니 원본값 그대로였음. 원본은 `imitation_scale=1.0`이지만 `reward_imitation()` 내부의 `w_joint_pos=15.0` 같은 큰 가중치가 참조 동작과 어긋난 자세를 강하게 벌점 처리해서, `alive=20`이 커도 그 벌점이 상쇄시키는 구조. **Stage 1은 `use_imitation=False`라 이 상쇄장치가 아예 없음.** 이론상 에피소드 최대 누적보상 570 중 `alive` 보너스만으로 400(70%)을 차지(1000스텝 × 20.0 × dt=0.02) — "제대로 걷기"보다 "어떻게든 안 죽고 버티기"가 압도적으로 쉬운 지름길이었음. 종료조건을 좁혀도 그 조건만 피하는 또 다른 붕괴 자세를 찾아내는 것으로 확인.
- **판정**: **FAIL (reward-hacked collapse, 종료조건 우회)**

## 수정 2 — `alive_scale` 스윕 (진행 중, 커밋 예정)

`alive_scale`을 20.0 → {2.0, 5.0, 10.0} 3가지로 낮춰서 동시 학습(각 `num_envs` 축소해서 GPU 하나에 병렬 실행). `tracking_lin_vel_scale=2.5`/`tracking_ang_vel_scale=6.0` 대비 생존 보너스 비중을 낮춰서 "서 있기만 해도 이득"이 안 되게 하는 게 목표. `Isaac-OpenDuckMini-Joystick-{v0,Alive5-v0,Alive10-v0}` 3개 gym task로 등록(`source/open_duck_mini_isaaclab/__init__.py`).

## Run 3~5 — 진행 중

| task | alive_scale | num_envs | 상태 |
|---|---|---|---|
| Isaac-OpenDuckMini-Joystick-v0 | 2.0 | TBD | 진행 예정 |
| Isaac-OpenDuckMini-Joystick-Alive5-v0 | 5.0 | TBD | 진행 예정 |
| Isaac-OpenDuckMini-Joystick-Alive10-v0 | 10.0 | TBD | 진행 예정 |

결과 나오는 대로 이 표와 위 형식으로 갱신할 것. `scripts/eval_policy_stability.sh`의 "STANDING RESULT"/"WALKING RESULT" 두 줄로 최종 판정.

---

## Stage 2(imitation) 활성화 시도 — 파이프라인 자체 버그 3개 발견/수정 (2026-07-26)

alive_scale 스윕과 별개로, "지금 것 안 되면 Stage 2로" 계획에 따라 실제로 `generate_reference_motion.sh`를 돌려봄. 세 가지 독립적인 버그를 발견/수정:

1. **번들 URDF가 10일 전 것** — OnShape 재수출/발 위치 CG 수정 전 버전. `scripts/patch_urdf_for_placo.py` 신설(커밋 `7bd81b0`/`78316de`) — Placo가 요구하는 프레임 별칭(`trunk`/`left_foot`/`right_foot`/`head`, 우리 OnShape URDF엔 없음)을 주입해서 실제 파일 생성(심볼릭 링크는 랩PC exFAT에서 불가해서 포기).
2. **placo 버전 미고정** — 최상위 `pyproject.toml`이 `placo`(버전 미지정)라 최신판(0.9.23)이 깔렸는데, `reference_motion_generator`는 실제로 `placo==0.6.3` API를 씀(`replan_timesteps` 등). 0.6.3으로 고정. **0.6.3은 macOS wheel이 없음** — 이 파이프라인은 사실상 랩PC(Ubuntu) 전용.
3. **안테나 관절 참조** — `placo_defaults.json`/`medium.json`/`fast.json`이 `left_antenna`/`right_antenna`를 명령하려 하는데, 우리 로봇엔 안테나 자체가 없음(`grep -c antenna robot/robot.urdf` == 0). 이게 랩PC의 구버전 placo(0.6.3)에서 **네이티브 메모리 손상 크래시**(`free(): corrupted unsorted chunks`, SIGABRT)로 나타났음 — 맥에 임시로 깐 신버전(0.9.23)으로 단계별 재현 스크립트를 돌려서 같은 지점에서 "Joint with name left_antenna not found" 라는 **깔끔한 예외**가 뜨는 걸 확인하고 원인 특정. 안테나 항목 제거로 해결(커밋 `acaeedf`/`6fc8000`).

**검증**: 랩PC에서 `gait_generator.py` 단독 실행 → exit 0, 500프레임 녹화, 실제 출력 json 저장 확인. 크래시는 완전히 해결됨.

**남은 이슈 (크래시와 별개, 후속 작업)**: 첫 성공 녹화의 실측 평균속도(`avg_x_lin_vel=0.3866 m/s`, dx=0.02 명령 기준)가 `auto_waddle.py`의 "medium" 속도필터 범위(0.05~0.15)를 크게 벗어남 — `walk_com_height` 등 Placo 걸음 파라미터가 여전히 원본 대형 로봇 기준으로 남아있어서로 추정. 이대로 216개 전체 스윕을 돌리면 대부분/전부 필터에 걸려 삭제될 가능성 높음 — 우리 로봇 스케일에 맞게 `placo_defaults.json`의 게이트 파라미터 재조정 필요.
