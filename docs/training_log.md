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

| task | alive_scale | num_envs | 최종 reward | 최종 ep_len | steady 높이 | worst upright | 판정 |
|---|---|---|---|---|---|---|---|
| Isaac-OpenDuckMini-Joystick-v0 | 2.0 | 512 | 31.21 | 634.97 | 0.093~0.10m | -0.11 | **FAIL** |
| Isaac-OpenDuckMini-Joystick-Alive5-v0 | 5.0 | 512 | 55.86 | 606.22 | 0.093m | -0.11 | **FAIL** |
| Isaac-OpenDuckMini-Joystick-Alive10-v0 | 10.0 | 512 | 70.87 | 413.42 | 0.105m | -0.06 | **FAIL** |

**결론 (2026-07-26)**: `eval_policy_stability.sh`로 3개 전부 검증 — **셋 다 여전히 무너진 채로 버팀** (steady-state 높이가 종료 임계값 0.09m 바로 위에 붙어있고, upright도 -1(직립)이 아니라 -0.06~-0.11로 거의 옆으로 누움). alive_scale을 20→2/5/10으로 낮춘 것만으로는 reward-hacking 콜랩스가 안 풀림. Disney 논문에서 확인한 대로, alive_scale 조정 같은 국소적 보상 튜닝보다 **imitation reward(Stage 2) 자체를 켜는 게 근본 해법**이라는 가설이 강화됨 — 마침 같은 밤에 Stage 2 파이프라인(궤적 생성기)도 별도로 고쳐서 실제 pkl을 만들어뒀음(위 "안테나 개념 전체 제거"/"4번째 버그" 섹션 참고).

팀원(원우)의 별도 파이프라인(`~/Desktop/miniduck`)도 참고차 확인함 — `root_tilt_exceeded`(30도 종료, 우리 90도보다 훨씬 엄격), `leg_antiphase`/`foot_alternating_contact`(양다리 정지 시 보상 0으로 만들어 "가만히 버티기" 자체를 구조적으로 차단) 같은 흥미로운 설계가 있었으나, 직진 보행 전용 휴리스틱이라 회전/횡이동 명령에서 정책을 왜곡시킬 위험이 크다고 판단해 채택하지 않고 imitation으로 진행하기로 결정.

## Stage 2 최초 실전 투입 (2026-07-26, 커밋 a382fa8/cbfcf79/46d7b7b 등)

`use_imitation=True`, `alive_scale`은 20.0으로 원복(imitation이 다시 상쇄장치 역할을 하므로). 전환 과정에서 **버그 2개 추가 발견**:
- `joystick_env.py`의 `_REPO_ROOT` 계산이 `os.path.dirname()` 4번(소스 트리 depth상 5번 필요)이라 `source/`에서 멈춰있었음 — `reference_motion_pkl`이 이미 `"source/..."`로 시작하는 상대경로라 합쳤을 때 `source/source/...`로 중복되며 `FileNotFoundError`. dirname 5번으로 수정.
- `_current_reference_motion` 텐서가 옛날 `40`차원으로 하드코딩돼있어서(안테나 제거로 36차원 된 것 반영 안 됨) `RuntimeError: shape mismatch [16,36] vs [16,40]`. `REF_FRAME_DIM` import해서 사용하도록 수정.

**스모크 테스트(3 iter, 16 envs) 통과** — Stage 2가 실제 학습 루프에서 처음으로 에러 없이 완주함. 이후 본 학습 시작: `--num_envs 2048 --run_name imitation_v1`, ETA 약 1시간 45분.

---

## Stage 2(imitation) 활성화 시도 — 파이프라인 자체 버그 3개 발견/수정 (2026-07-26)

alive_scale 스윕과 별개로, "지금 것 안 되면 Stage 2로" 계획에 따라 실제로 `generate_reference_motion.sh`를 돌려봄. 세 가지 독립적인 버그를 발견/수정:

1. **번들 URDF가 10일 전 것** — OnShape 재수출/발 위치 CG 수정 전 버전. `scripts/patch_urdf_for_placo.py` 신설(커밋 `7bd81b0`/`78316de`) — Placo가 요구하는 프레임 별칭(`trunk`/`left_foot`/`right_foot`/`head`, 우리 OnShape URDF엔 없음)을 주입해서 실제 파일 생성(심볼릭 링크는 랩PC exFAT에서 불가해서 포기).
2. **placo 버전 미고정** — 최상위 `pyproject.toml`이 `placo`(버전 미지정)라 최신판(0.9.23)이 깔렸는데, `reference_motion_generator`는 실제로 `placo==0.6.3` API를 씀(`replan_timesteps` 등). 0.6.3으로 고정. **0.6.3은 macOS wheel이 없음** — 이 파이프라인은 사실상 랩PC(Ubuntu) 전용.
3. **안테나 관절 참조** — `placo_defaults.json`/`medium.json`/`fast.json`이 `left_antenna`/`right_antenna`를 명령하려 하는데, 우리 로봇엔 안테나 자체가 없음(`grep -c antenna robot/robot.urdf` == 0). 이게 랩PC의 구버전 placo(0.6.3)에서 **네이티브 메모리 손상 크래시**(`free(): corrupted unsorted chunks`, SIGABRT)로 나타났음 — 맥에 임시로 깐 신버전(0.9.23)으로 단계별 재현 스크립트를 돌려서 같은 지점에서 "Joint with name left_antenna not found" 라는 **깔끔한 예외**가 뜨는 걸 확인하고 원인 특정. 안테나 항목 제거로 해결(커밋 `acaeedf`/`6fc8000`).

**검증**: 랩PC에서 `gait_generator.py` 단독 실행 → exit 0, 500프레임 녹화, 실제 출력 json 저장 확인. 크래시는 완전히 해결됨.

**남은 이슈 (크래시와 별개, 후속 작업)**: 첫 성공 녹화의 실측 평균속도(`avg_x_lin_vel=0.3866 m/s`, dx=0.02 명령 기준)가 `auto_waddle.py`의 "medium" 속도필터 범위(0.05~0.15)를 크게 벗어남 — `walk_com_height` 등 Placo 걸음 파라미터가 여전히 원본 대형 로봇 기준으로 남아있어서로 추정. 이대로 216개 전체 스윕을 돌리면 대부분/전부 필터에 걸려 삭제될 가능성 높음 — 우리 로봇 스케일에 맞게 `placo_defaults.json`의 게이트 파라미터 재조정 필요.

---

## 안테나 개념 전체 제거 (2026-07-26, 커밋 `c16aa59`/`4aee8fb`)

`joint_order.py`(`REF_JOINT_NAMES`/`REF_LEG_JOINT_IDX` 16→14), `poly_reference_motion.py`(`REF_FRAME_DIM` 40→36), `rewards.py::reward_imitation`(슬라이스 인덱스 전부)까지 안테나 없는 로봇 기준으로 정리. `tests/test_reward_leg_index_alignment.py` 갱신 후 맥/랩PC 양쪽에서 통과 확인.

## 4번째 버그 발견/수정 — `auto_waddle.py`가 존재하지 않는 `python` 명령 호출 (커밋 `1fc4daa`)

단일 녹화 검증(joint ROM/발접촉교대/root height 그래프로 확인 — 정상 주기 보행 확인됨) 후 실제 240개 전체 스윕을 돌렸는데 "0.09초"만에 "완료"됨 — 로그 파일 240개가 전부 0바이트. 원인: `cmd = ["python", ...]`인데 이 우분투엔 `python` 실행파일이 없음(`python3`만 있음, `which python` → not found). 서브프로세스 240개가 전부 즉시 실패했는데 예외가 조용히 삼켜져서 "성공"처럼 보였음. `python3`로 수정.

**최종 검증**: 수정 후 재실행 → 240/240 녹화 성공, `fit_poly.py`로 피팅 → **최초로 실제 데이터가 든 `polynomial_coefficients.pkl`(2.8MB, 240 항목) 생성**(커밋 `f4cdcbb`/`c831005`). `PolyReferenceMotion` 클래스로 직접 로드해서 `get_reference_motion()` 호출까지 확인 — Stage 2 파이프라인이 처음부터 끝까지 실제로 동작함을 확인.

**아직 남은 버그 (차단 요소 아님)**: `auto_waddle.py`의 속도필터가 `preset_name == "medium"`을 비교하는데 실제 값은 `"0_medium"`처럼 인덱스가 붙어있어서 **필터가 한 번도 매치되지 않음** — 240개가 전부(필터링 없이) pkl에 들어감. 일부는 속도가 매우 낮은 조합도 섞여 있음. Stage 2 학습 품질에 영향을 줄 수 있으나, "파이프라인이 작동하는가"라는 오늘 밤의 목표는 달성됨 — 다음 세션에서 필터 로직과 게이트 파라미터를 같이 손보면 됨.

---

## Run 6 — Stage 2 최초 본학습 (`imitation_v1`, 2026-07-26 09:02~12:14)

- **설정**: `use_imitation=True`, `alive_scale=20.0`(원복), `num_envs=4096`, `max_iterations=3000`. 레퍼런스 모션은 위에서 만든 240개 조합 `polynomial_coefficients.pkl` (속도필터 버그로 필터링 없이 전부 포함된 상태, 그중 하나는 실측 `avg_x_lin_vel=0.3866 m/s`로 원래 의도한 0.05~0.15 밴드를 크게 벗어남 — 이 오염이 이번 결과의 유력한 원인 후보).
- **학습 진행**: 3시간11분(iteration 0→2999) 완주. GPU 77% util, ~6GB/16GB VRAM. 에피소드 길이가 초반(iter 0: 24스텝)엔 꾸준히 늘다가 iter ~1200 이후로는 250~390스텝 구간에서 오르내리며 정체(1000스텝 풀 클리어에 도달 못함). Mean reward는 절대 스케일이 Stage1과 달라(imitation 벌점 포함) 직접 비교 불가 — 참고용으로 마지막 iteration 기준 2~4대.
- **`eval_policy_stability.sh --checkpoint model_2999.pt --num_envs 8 --num_steps 500` 결과**:
  - base height (전체 500스텝): [0.0938, 0.2227] m — 최저점이 종료 임계값(0.09m)에 바로 붙어있음
  - base height (마지막 200스텝): [0.1054, 0.1184] m (HOME=0.150m, 70~79%) — Stage1의 콜랩스보다는 확실히 나음
  - worst-case upright (마지막 200스텝): **-0.0101** (-1이 완벽 직립, 0이 완전 옆으로 누움) — 마지막 구간에도 순간적으로 거의 옆으로 넘어가는 순간이 있었다는 뜻
  - mean leg-joint ROM: 0.7269 rad (임계값 0.15 대비 매우 높음 → PASS)
  - lin-vel tracking error: 0.2142 m/s (임계값 0.15 → **FAIL**)
  - foot-contact toggle: 초당 평균 11.7회/발 (임계값 4 대비 높음 → PASS 조건은 만족하지만, ROM/toggle이 둘 다 높다는 건 "잘 걷는다"보다 "격하게 흔든다"에 더 가까울 수 있음)
  - **STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE**
  - **WALKING RESULT: LIKELY REWARD-HACKING (standing/twitching, not stepping)**
- **판정**: **FAIL.** Stage1(뻣뻣하게 웅크려서 안 죽기)과는 다른 실패 양상 — 다리를 크게, 발을 자주 움직이지만(twitching) 명령 속도를 못 따라가고(vel err 0.21) 순간적으로 거의 넘어짐. "정지해서 버티기"형 리워드해킹은 확실히 깨졌지만, 대신 "격하게 흔들며 버티기"형 실패로 옮겨간 것으로 보임.
- **다음 조치 후보** (미착수, 판단 근거만 남김):
  1. **레퍼런스 모션 오염 가능성이 가장 유력** — `auto_waddle.py` 속도필터 버그로 240개가 필터링 없이 전부 pkl에 들어감. 실측 속도 밴드(중앙값 0.070 m/s)와 크게 어긋나는 조합(예: 0.39 m/s 레코딩)이 섞여 있으면, 명령 속도에 따라 참조 궤적 자체가 물리적으로 무리한 걸음일 수 있음 → 속도필터 버그부터 고치고 재생성 후 재학습이 우선순위 1번.
  2. 3000 iteration이 imitation 학습엔 부족할 수 있음 — episode length가 아직 명확히 수렴하지 않고 정체 중이라 더 돌리면 개선될 여지가 있음(단, 위 1번을 먼저 고치는 게 순서상 맞아 보임).
  3. `patch_urdf_for_placo.py`의 프레임 오프셋이 리빌드 전 구형 URDF에서 상속된 근사치라는 점도 궤적 자체의 물리적 정합성에 영향 줄 수 있음 — 현재 지오메트리로 재도출 필요(기존에도 TODO였음).

---

## Run 7 — Stage 2 재도전 준비: 하루 종일의 근본수정 후 `imitation_v2` 착수 (2026-07-26 22:27~)

Run 6(imitation_v1) 실패 후 이날 하루 동안 잡은 수정 전체 (상세는 decisions.md·리포트 아티팩트):
1. `auto_waddle.py` 속도필터 문자열버그 수정 (`"{i}_medium"` vs `"medium"`)
2. OnShape 재작업(사용자): 질량 비대칭(밀도), `right_knee` 메이트 회전방향, 억제 프레임 해제, 직립 기준자세, 파스너 추가
3. `trunk/left_foot/right_foot/head` 프레임을 OnShape Fastened 메이트로 네이티브 임포트 (미러링된 정확한 오프셋) — `patch_urdf_for_placo.py`는 검증기로 축소
4. `HOME_JOINT_POS` 전부 0 / `HOME_BASE_HEIGHT` 0.15→0.193 (Isaac zero-action PD hold 실측 PASS)
5. `PolyReferenceMotion` 스파스 그리드 KeyError → 최근접 폴백
6. Placo `enable_joint_limits(True)` (기존 False — 전 궤적이 한계 초과 상태였음)
7. `medium.json` walk_com_height 0.205→0.16 (defaults만 고치고 실제 스윕 프리셋 미적용이었던 것), feet_spacing 0.16→0.18(제로포즈 실측 0.183)
8. `medium.json` neck 20°/head −26° 하드코딩 → 0 (구부정한 머리)
9. 무릎 굽힘 방향 강제 (left≤0/right≥0) — 오른무릎 사람식 꺾임 수정
10. Mac 뷰어 stale URDF 교훈: 겉보기 "왼발 분리"는 뷰어용 URDF 사본이 아침 버전이라서. 순수 pinocchio 멀티클립 뷰어 신설(`replay_motion_meshcat.py`)
11. 직진 드리프트 조사: 몸기준 vy≈0.004 (미끄러짐 없음), 무명령 yaw ~2°/s — **업스트림 v1 대조실험에서도 동일 재현** → 플래너 고유 특성으로 결론
12. `fit_poly.py`에서 참조 속도채널(lin x/y, ang z) 평균을 그리드 키(명령값)에 재정렬 — 이미테이션이 드리프트를 가르치지 않도록

**최종 데이터 검증(118개)**: 관절한계 위반 0, 무릎 방향 위반 0, 속도 100% in-band, 발접촉 토글 38/37(이론 ~37).

**⚠️ 발견된 미해결 이슈 — 무릎 90° 포화**: 최종 세트에서 무릎 ROM이 거의 0 (left −1.571~−1.535, right +1.571 고정). 기하 계산상 com 0.16이 요구하는 무릎 굽힘은 ~105°인데 URDF 한계가 ±90°(OnShape 메이트 설정값)라 무릎이 한계에 눌러붙고 나머지를 hip/ankle이 흡수 — 물리적으론 유효하나 걸음 스타일이 뻣뻣한 크라우치로 왜곡됨. (수정 전 궤적들이 103°까지 갔던 것과 정합 — 애초에 ±90°가 실기 가동범위인지 재확인 필요. 업스트림 v2는 무릎 범위 π 사용.) **내일 할 일: 실기 무릎 실제 가동범위 확인 → OnShape 메이트 한계 갱신 → 재임포트·재생성·재학습.**

**Run 7 실행**: 위 상태의 pkl로 `imitation_v2` 학습 시작 (num_envs=4096, 3000 iter, run dir `2026-07-26_22-27-31_imitation_v2`) — 무릎 스타일 이슈에도 학습이 성립하는지 확인하는 실험 성격. TensorBoard :6006 해당 런으로 재스코프.

**Run 7 중지 (2026-07-26 22:4x)**: 사용자 지시("내일 할 거임")로 iteration 초반에 학습 중단, 프로세스 정리 완료(GPU 0%). 무릎 90° 포화 이슈가 발견된 상태라 어차피 내일 무릎 실기 가동범위 확인 → OnShape 한계 갱신 → 재생성 후 다시 시작하는 게 순서상 맞음. run dir `2026-07-26_22-27-31_imitation_v2`는 초기 몇 iteration 체크포인트만 있는 미완주 상태로 남아있음(삭제 안 함).

**Run 7 결과 (2026-07-27 01:56, iteration 2999 완주, `eval_policy_stability.sh` 검증)**:
```
base height (마지막 200스텝): [0.1166, 0.1261] m (HOME=0.150m, 78~84%)
worst-case upright (마지막 200스텝): -0.0143  (-1=완벽 직립, 0=완전 옆으로 누움) — 순간적으로 거의 옆으로 넘어감
mean leg-joint ROM: 0.6630 rad (임계값 0.15 대비 높음 → PASS)
lin-vel tracking error: 0.1727 m/s (임계값 0.15 → FAIL, 근소 초과)
foot-contact toggle: 초당 평균 43.4회/발 (10초간) — imitation_v1의 11.7보다 훨씬 높음, 격한 떨림 신호

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping)
```
**판정: FAIL.** 무릎 ROM 부활·좌우 대칭·관절한계·드리프트까지 레퍼런스 모션 자체는 철저히 검증된 상태로 재학습했음에도 `imitation_v1`과 동일하게 twitching/reward-hacking으로 실패. foot-contact toggle이 43.4회(v1의 11.7회 대비 약 3.7배)로 훨씬 높아진 건 이번이 "가만히 버티기"가 아니라 **더 격하게 떠는** 실패 양상임을 시사.

**근본원인 후보 (확실한 단일원인 특정 안 됨, 사용자 판단 대기)**:
1. `alive_scale=20.0`이 여전히 이론적 스텝당 상한(~33.5)의 60%를 차지 — Stage1 스윕 때 이미 "너무 크면 리워드해킹 유발" 진단이 나온 값인데, imitation 켠 뒤로는 재검증 없이 원복만 해뒀음. 새로 고쳐진(120° ROM) 레퍼런스로 낮춰서 테스트해본 적 없음.
2. `w_joint_pos=15.0`이 매우 크고, 새 레퍼런스의 무릎 ROM이 최대 120°까지 커져서 추종이 어려워졌을 수 있음 — 정책이 매끄러운 추종 대신 목표 근처를 고주파로 떠는 방식으로 국소최적에 빠졌을 가능성 (foot toggle 급증과 정합).
3. 단순히 수렴 부족 — 2.95억 스텝(원본 기본값의 약 2배)은 썼지만, 리워드가 학습 후반까지도 노이즈만 있고 뚜렷한 상승 추세가 없었음(0.9~5.5 구간에서 정체).

세 후보 모두 그럴듯하고 서로 구분할 명확한 증거가 부족해 **자동 재시도는 보류**하고 사용자 판단을 기다림 (지시받은 "명확한 단일원인+저비용수정" 기준 미충족).

## alive_scale × w_joint_pos 스윕 (2026-07-27, num_envs=1024, max_iterations=800 스크리닝)

`imitation_v2` FAIL(twitching 심화) 원인 후보 중 리워드 가중치 축을 좁히기 위한 빠른 스윕. 4개 동시 실행, `A{alive_scale}J{w_joint_pos}` 네이밍.

**학습 중 지표 (iteration 799/800)**:
| 조합 | reward | ep_len | action noise std | value_fn loss |
|---|---|---|---|---|
| A10J10 | 0.02 | 67.34 | 1.00 | 0.0007 |
| A5J15 | 0.00 | 35.18 | **4.35 (폭주)** | **0.0000 (붕괴)** |
| A20J5 | **0.98** | **154.06** | 1.17 | 0.0345 |
| A5J5 | 0.00 | 35.18 | **4.35 (폭주)** | **0.0000 (붕괴)** |

**`eval_policy_stability.sh --num_envs 8 --num_steps 500` 결과** (참고: `imitation_v2` 풀런 베이스라인 toggle=43.4, upright=-0.0143, lin-vel err=0.173):
| 조합 | toggle/10s | worst upright | lin-vel err | leg ROM |
|---|---|---|---|---|
| A10J10 | 124.9 | -0.0049 | 0.220 | 0.453 |
| A5J15 | 70.7 | -0.0007 | 0.417 | 0.686 |
| A20J5 | 104.9 | -0.0014 | 0.225 | 0.493 |
| A5J5 | 77.7 | -0.0055 | 0.411 | 0.742 |

전부 STANDING/WALKING FAIL (예상된 결과 — 총 학습량이 `imitation_v2`의 약 6.7%인 스크리닝이라 절대 성패가 아니라 상대비교 용도).

**결론**: `alive_scale=5`(A5J15, A5J5)는 **학습 자체가 붕괴** — value function이 상수(≈0)로 무너지고 action noise std가 4.35까지 폭주(사실상 정책이 랜덤에 가까워짐). `alive_scale=10~20`은 안정적으로 학습됨. 그중 **`A20J5`(alive=20 유지, w_joint_pos만 15→5)가 4개 중 압도적으로 건강** — reward/ep_len 최고, toggle도 A10J10보다 낮음. → **`w_joint_pos=15`가 너무 가혹했다는 가설이 `alive_scale` 가설보다 더 유력**하다는 방향성 확보. `A20J5` 조합으로 본학습(num_envs=4096, max_iterations=3000) 재개를 사용자에게 제안, 승인 대기.

## Run 8 — `imitation_v3` (2026-07-27 03:46, A20J5, num_envs=4096, max_iterations=3000)

사용자 승인("응 시작해") 후 스윕 우승 조합(A20J5)으로 본학습. run dir `2026-07-27_03-46-08_imitation_v3`.

**리워드 크기 의문 조사 (학습 중 사용자 지적: "스텝당 최대 리워드가 거의 30인데 지금 한자리수인게 말이 안 됨")**:
- `_get_rewards()`의 `reward = torch.sum(...) * dt`에서 `dt=step_dt=0.02`를 빠뜨리고 이전에 "~33.5"라고 말한 것은 오산 — 정정하면 스텝당 이론 상한은 **~0.67**.
- rsl_rl `on_policy_runner.py` 소스 확인: TensorBoard의 `Train/mean_reward`(콘솔 "Mean reward")는 스텝당 평균이 아니라 **에피소드 종료 시에만 flush되는 전체 누적합**(`rewbuffer`). 에피소드 길이 300~450스텝 기준 정상이면 누적 200~300은 나와야 하는데 실측 1~4 — 스텝당 실질 평균은 이론 상한의 1~2% 수준.
- `_get_rewards()` 마지막 줄에 `torch.clamp(reward, 0.0, 10000.0)`이 있음을 재확인 — 페널티 항 합이 순간적으로 음수가 되는 스텝은 전부 0으로 깎임. 이게 낮은 누적치의 유력한 설명 후보로 지목, `scripts/reward_breakdown.py`(신규) 작성해 항목별 실측 시도 → 학습과 GPU 동시 점유로 원인불명 크래시(1차), 학습 완료 후 재시도 예정.

**최종 결과 (2026-07-27 06:2x, iteration 2999 완주, `model_2999.pt`, `eval_policy_stability.sh --num_envs 8 --num_steps 500`)**:
```
worst-case upright (마지막 200스텝): -0.4396  (v2의 -0.0143보다 직립에 더 가까움)
mean leg-joint ROM: 0.5590 rad
lin-vel tracking error: 0.3932 m/s
foot-contact toggle: 79.1회/발/10초 — v2(43.4)보다 1.8배 악화

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping)
```
**판정: FAIL.** 스윕(800 iter)에서 가장 건강해 보였던 A20J5가 3000 iter까지 풀스케일로 가니 toggle이 오히려 v2보다 악화 — `w_joint_pos` 단독 하향이 근본 해법은 아니었음을 시사. 자세(upright)는 개선됐지만 발 접촉이 훨씬 더 불안정해진, 다른 종류의 twitching으로 보임.

## Disney BD-X 논문 대조 → 접촉기반 종료조건 도입 (2026-07-27)

사용자가 Disney Research의 BD-X 로봇 논문(Grandia et al. 2024 — Open Duck Mini의 원본 설계 레퍼런스)과 우리 보상함수/종료조건을 대조한 분석(아티팩트 `f20e0cf8`)을 공유. 핵심 발견 두 가지:
1. **`alive_scale=20`은 Disney 원본 숫자 그대로**였음 — 다만 Disney는 imitation 항(목 관절 추종 가중치 100!)을 처음부터 항상 켜놓고 학습해서 alive=20이 보상을 지배하지 못하게 상쇄됨. 우리는 이미 imitation을 켠 상태(Stage 2)이므로 이 항목은 구조적으로는 이미 반영돼 있음.
2. **종료조건**: 논문 V-B절 — "머리 또는 몸통이 지면에 닿으면 즉시 종료"(접촉 기반). 우리는 높이비율(`min_base_height_ratio`)/뒤집힘(`flipped`) 기준만 썼는데, Run 1(높이조건 없음)과 Run 2(높이조건 추가 후)에서 각각 그 조건들을 우회하는 붕괴 자세를 정책이 찾아낸 전례가 있음 — 접촉 기반은 자세가 어떻게 뒤틀리든 우회 불가능해서 더 근본적.

**조치**: `joystick_env.py`에 `ContactSensor`로 `trunk_assembly`/`head_pitch_assembly`(URDF의 `head` 링크는 관성이 없어 물리 바디로 안 남고 부모에 병합되므로, 실제 바디명은 `head_pitch_assembly` — 이 오타로 최초 커밋은 `find_bodies`에서 즉시 크래시, eval 실행 시 발견해 바로 수정) 접촉력을 읽어 `_get_dones()`의 기존 조건에 OR로 추가 (커밋: 몸통/머리 접촉 기반 종료조건 추가 → head 바디명 수정). 기존 높이/뒤집힘 조건은 유지(대체 아님, 추가).

**검증 완료 (2026-07-27 06:4x)**: `imitation_v4_contactterm_validation`(num_envs=64, max_iterations=200) 정상 완주, 에피소드 길이 13→50까지 상승 후 42~45에서 안정 — 즉시-종료 버그 없음 확인.

**WebRTC로 `imitation_v3`(model_2999.pt) 직접 시각 확인 (`play.py --livestream 2`)**: 사용자 육안 관찰 — "시작하자마자 관절이 특정각도에 고착되면서 팡 하고 튕겨나감 → 터미네이트". eval 수치(toggle=79.1, reward-hacking 판정)와 정합되는 시각적 확인. 관절이 한계각 근처에서 고착 후 순간적으로 큰 힘이 실려 튕겨나가는 양상 — 단순 "제자리 떨림"보다는 관절한계/PD게인 근처에서의 불안정한 힘 스파이크에 가까워 보임 (원인 특정은 안 됨, 참고용 기록).

**다음**: `imitation_v4`(A20J5 + 접촉종료, num_envs=4096, max_iterations=3000) 본학습 시작.

## Run 9 — `imitation_v4` (A20J5 + 접촉기반 종료조건) — 완주 실패(크래시) + FAIL

run dir `2026-07-27_11-53-51_imitation_v4`. iteration 2900까지는 정상(에피소드 길이 373~576, 최대치 근처로 안정적 — v3보다 훨씬 건강해 보이는 궤적이었음), 그러다:

```
iter 2978: value_function loss = 63.96 (정상)
iter 2979: value_function loss = 385,798,126.4        ← 폭발 시작
iter 2980: value_function loss = 799,433,533,738,188.75
iter 2981: value_function loss = 2,828,388,348,591,223,603,200.0
iter 2982: value_function loss = 3.17×10^27
iter 2983: value_function loss = 4.93×10^32 → RuntimeError: normal expects all elements of std >= 0.0
```

단 5 iteration 만에 value function loss가 기하급수적으로 폭주 → 정책의 액션 분포 표준편차가 NaN이 되며 iteration 2983/3000(99.4%)에서 크래시. `model_2999.pt`는 없음, 마지막 정상 체크포인트는 `model_2900.pt`(96.7%).

**`model_2900.pt` eval 결과**:
```
worst-case upright (마지막 200스텝): -0.0013 (거의 완전히 옆으로 누움 — v3의 -0.4396보다도 나쁨)
mean leg-joint ROM: 0.4336 rad
lin-vel tracking error: 0.1980 m/s
foot-contact toggle: 78.9회/발/10초 (v3의 79.1과 사실상 동일)

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping)
```
**판정: FAIL.** toggle이 v3와 거의 동일해서, 접촉기반 종료조건 도입 자체는 이 트위칭 실패 양상을 해결하지 못했음을 시사(하지만 종료조건 우회 문제 자체는 별개로 여전히 유효한 개선). 학습 후반부 value function 폭주는 새로운 실패 유형 — 원인 미조사(NaN 유발 지점의 관측치/리워드 스파이크 등 후속 조사 필요, 지금은 보류).

**다음**: 사용자 지시로 스윕 방향을 반전 — `imitation_v5`(A30J25: alive_scale 20→30, w_joint_pos 5→25) 시작.

## WebRTC 육안 확인 → 크로치(crouch) 리워드해킹 발견 + 대응 (2026-07-27)

`imitation_v4`의 `model_2900.pt`를 `play.py --livestream 2`로 사용자가 직접 확인. 관찰: "터미네이션 안될려고 무슨 플랭크같이 머리랑 발로 버티는 동작", "시작하자마자 땅이랑 겹치는지 튕겨나가고 행동이 고착됨".

**진단 (`scripts/contact_diagnostic.py`, 신규 작성)**: model_2900.pt로 200스텝 롤아웃하며 몸통/머리 접촉력과 base_z를 프레임별로 로깅.
```
trunk/head 접촉력: 200스텝 내내 8개 env 전부 정확히 0.000N (단 한 번도 0 아님)
base_z: 스폰 직후 ~0.21 → 100+스텝 동안 0.133~0.134에서 완벽히 안정
```
**분석**: `min_base_height_ratio=0.6` 기준 붕괴 판정 높이 = `HOME_BASE_HEIGHT(0.193)×0.6 = 0.1158m`. 정책이 찾은 크로치 자세(0.133~0.134m, ratio≈0.69)는 이보다 살짝 높아서 안 걸림. 자기충돌(`enabled_self_collisions=False`)이 꺼져있어 머리가 자기 다리를 뚫고 들어가도 물리적으로 힘이 안 생기므로 접촉기반 종료조건도 안 걸림. **뒤집힘/높이/접촉 세 종료조건을 전부 동시에 피하는 자세를 정확히 찾아낸 것** — "붕괴됐지만 안 죽는" 리워드해킹의 새 버전.

**조치**:
1. `min_base_height_ratio`: 0.6 → 0.75 (이 크로치 자세가 새 기준 밑으로 들어감)
2. 접촉기반 종료조건에 양쪽 다리 `knee_and_ankle_assembly`(1~4) 추가 — 트렁크/머리가 안 닿아도 다리를 접어서 자세를 낮추는 경우까지 감지
3. (참고, 결과적으로 불필요했지만 시도함) 접촉 임계값 자체를 발 리워드용과 분리해 1.0N→0.5N으로 낮춤 — 실측 결과 힘이 진짜 0이라 임계값 조정만으론 해결 안 됐음을 확인, 별도 임계값 분리는 유지(향후 유용할 수 있음)

검증(num_envs=64, max_iterations=150) 정상 완주, 에피소드 길이 40~45로 안정, 새 바디명(`knee_and_ankle_assembly*`) 크래시 없음 확인 후 `imitation_v5` 시작.

## Run 10 — `imitation_v5` (A30J25 + 크로치 방지 수정) 시작 (2026-07-27, run dir 로그 확인)

num_envs=4096, max_iterations=3000, ETA~2h48m. alive_scale=30(20→30), w_joint_pos=25(5→25) — 스윕 근거와 반대 방향의 사용자 지시 시도. 실패 시 `imitation_v6`(imitation_scale=2.0, init_noise_std=2.0 추가)로 자동 진행 예정.

**결과 (2026-07-27 17:26, iteration 2999 완주, `eval_policy_stability.sh`)**:
```
worst-case upright (마지막 200스텝): -0.0792
mean leg-joint ROM: 0.4235 rad
lin-vel tracking error: 0.1887 m/s
foot-contact toggle: 159.1회/발/10초 — 지금까지 전체 런 중 최악 (v2=43.4, v3=79.1, v4=78.9)

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping)
```
**판정: FAIL, 그것도 최악.** alive_scale·w_joint_pos를 동시에 올리는 방향이 명확히 역효과였음 — 문헌조사(아래 참고)의 경고("생존 보너스 과다 → 전진 없이 서있기만")와 정합.

에피소드 길이도 특이했음: v4 대비 같은 iteration 지점에서 6~8배 짧았다가(크로치 방지 수정으로 종료조건이 훨씬 자주 발동한 결과로 추정) 후반부(iter 2349)엔 128까지 회복 — 크로치 자체는 확실히 못 쓰게 막았지만 대신 toggle이 급증하는 다른 실패 양상으로 이동한 것으로 보임.

`reward_breakdown.py`로 alive/imitation 상쇄 가설을 실측하려 했으나 GPU 자원경합 없이도 3회 연속 조용히 크래시 — 스크립트 자체가 불안정하다고 판단, 추가 재시도 없이 정성적 판단으로 대체(공식 기반 추론: alive=30의 raw 기여도 스텝당 +0.6인데 w_joint_pos=25의 관절오차 페널티가 학습 초반 이를 쉽게 상쇄해 클램프로 0이 되는 스텝이 대부분일 것으로 추정, 미확정).

## Run 11 — RSI(Reference State Initialization) 도입 → `imitation_v6` (2026-07-27)

사용자가 "실측하고 네가 방향성 정해서 알아서 돌려, 내 개입없이"로 전권 위임. v5가 최악의 결과였고 문헌조사에서 이미 "alive/imitation 동시 상승"에 대한 근거가 약하다고 나왔던 것과 종합해, 사전승인됐던 `imitation_v6`(가중치 추가상승+노이즈2배) 대신 문헌조사에서 나온 더 근거있는 대안인 **RSI**로 방향 전환.

**발견**: `_reset_idx()`가 매 에피소드 `self._imitation_i[env_ids] = 0`으로 항상 레퍼런스 모션 phase=0에서 시작하고 있었음 — DeepMimic(Peng et al. 2018, 이 리워드 계보 전체의 원조)이 쓰는 RSI(에피소드를 레퍼런스 클립의 랜덤 지점에서 시작)가 빠져있었음. v3~v5 전부 이 부분은 안 건드리고 가중치만 조정했었음.

**구현**: `_imitation_i`를 `[0, gait_period_steps)` 균등샘플로 바꾸고, 다리 관절(`ACT_LEG_JOINT_IDX`)의 스폰 시 자세/속도를 그 샘플된 phase의 레퍼런스 프레임 값으로 직접 세팅.

**버그 1회 발견/수정**: 첫 검증(`rsi_validation`, num_envs=64, 200 iter)에서 에피소드 길이가 8 근처에 완전히 고정 — 스폰 직후 거의 즉시 종료되는 심각한 버그. 원인: `ACT_LEG_JOINT_IDX`(액추에이터 순서)를 네이티브(USD) 순서 텐서인 `joint_pos`/`joint_vel`에 그대로 인덱싱해서 엉뚱한 관절에 레퍼런스 값이 쓰였음 — `self._joint_ids`로 액추에이터→네이티브 순서 매핑 필요. 수정 후 재검증(`rsi_validation2`)에서 에피소드 길이 35~39로 정상화 확인.

**결정**: RSI 효과를 A30J25(방금 최악으로 확인됨)가 아니라 더 나은 베이스라인인 **A20J5** 위에서 격리해서 테스트하기로 판단 (변수 하나씩 검증하는 원칙 유지). `imitation_v6` = A20J5 config + 접촉기반 종료조건 + 크로치 방지(높이비율 0.75, 무릎/발목 접촉) + RSI(신규), num_envs=512(사용자 요청 — 데이터량은 8배 줄지만), `--livestream 2`(headless 대신 WebRTC 렌더링, 사용자가 실시간으로 보고 싶어함), max_iterations=3000, ETA~2h32m 시작.

**중단 사고 (iteration ~250/3000, 18:59)**: 사용자가 WebRTC 화면에서 시뮬레이션 출력을 USD로 바꿨다가 화면이 깨짐 — 로그에 `OGN deregister omni.physx.fabric` 발생 후 학습 루프 자체가 멈춤(GPU util 0%, 타임스텝 정지, 프로세스는 살아있으나 응답없음). 사용자 지시로 이 런의 로그/체크포인트 전체 삭제(`2026-07-27_18-46-36_imitation_v6` 디렉토리) 후 재시작.

재시작 첫 시도에서 `omni.physx.fabric.plugin CUDA error: invalid argument (DirectGpuHelper.cpp:752)`가 연속 발생 — 강제종료(`kill -9`)가 GPU 렌더 상태를 깨끗이 정리 못 하고 남긴 것으로 추정. `--headless`로는 CUDA 에러 0건 정상 작동 확인(렌더 파이프라인만 손상, 물리연산 자체는 정상). 빈 스트리밍 세션(`isaac-sim.streaming.sh`)만 단독으로 띄워 렌더러가 살아있는지 확인 → CUDA 에러 0건, 정상 로드 — 즉 일시적 GPU 상태 문제였고 시간이 지나며 회복된 것으로 보임. 이후 `train.py --livestream 2`로 재시도 → 정상 작동(CUDA 에러 0건). `imitation_v6` 최종적으로 새 run dir(`2026-07-27_19-07-37_imitation_v6`)로 19:07경 재시작, ETA~2h33m.

## imitation_v6 결과 — 지금까지 중 가장 유의미한 진전 (2026-07-27 21:50, iteration 2999 완주)

**eval 결과 (`model_2999.pt`)**:
```
worst-case upright (마지막 200스텝): -0.0040
mean leg-joint ROM: 1.3072 rad — 역대 최대 (v3=0.559, v4=0.4235, v5=0.4235의 2.3~3배)
lin-vel tracking error: 0.4304 m/s — 오히려 악화 (v4=0.198, v5=0.189보다 나쁨)
foot-contact toggle: 29.4회/발/10초 — 역대 최저, v2(43.4, Stage1 베이스라인)보다도 낮은 첫 사례

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping) — 자동판정 임계값 기준
```
**판정: eval 스크립트 자동판정은 여전히 FAIL이지만, 지표 조합의 성격이 이전과 확연히 다름.** toggle이 극적으로 낮아지고 ROM이 2배 이상 커진 건 "제자리에서 버티기/편법"이 아니라 **실제로 크게 움직이며 걷기를 시도하는 패턴**과 정합 — 다만 아직 균형을 못 잡아 속도추종은 악화. WebRTC로 사용자가 직접 확인: "계속 넘어짐". IMU 버그 의심 제기 → 코드 재검토(가속도계는 `lin_acc_w = Δv/dt + gravity_bias_w`로 중력 정상 포함 확인, observation엔 orientation 직접 안 들어가지만 이건 Playground 원본도 동일한 구조라 v1~v6 공통이지 v6만의 문제 아님) → **IMU 버그보다는 "더 크고 실제 걷기에 가까운 동작을 시도하다 아직 균형이 안 따라오는" 상태**로 결론.

## Run 12 — `imitation_v7`: RSI 유지 + num_envs 스케일업 (2026-07-28)

사용자 전권 위임 하에 판단: v6는 처음으로 리워드해킹이 아닌 "진짜 걷기 시도" 패턴을 보였으나, num_envs=512(사용자가 실시간 시각화 위해 4096→512로 낮췄던 것 — 데이터량 8분의 1)라 단순히 학습 데이터 부족으로 아직 안정화가 덜 됐을 가능성이 높다고 판단. 새 레버(대칭손실 등) 추가하기 전에 **RSI 그대로 유지 + num_envs만 4096으로 복귀(headless, 실시간 시각화는 이번엔 포기)** — 새 코드 변경 없는 순수 스케일업이라 별도 검증 없이 바로 시작.

사용자가 "최대로 자원써서 빠르게" 요청 → num_envs=8192로 먼저 시도(Disney 논문 규모) → **오히려 ETA가 2h50m→4h48m으로 악화** (GPU가 이미 4096에서 70% 활용 중이라 컴퓨트 병목, 메모리는 여유 있었지만 관계없었음) → 4096이 이 GPU에서 실질적 최적점으로 판단, 4096으로 복귀. `imitation_v7` 시작, ETA~2h38m.

## Disney 논문 vs RSI — "간단한 모방학습만 쓰고싶다" (2026-07-28)

사용자 질문 "디즈니꺼는 rsi 씀?" → 논문 원문(arxiv.org/html/2501.05204v1) 직접 확인 → **RSI 언급 전혀 없음**. 논문의 phase 관련 언급은 전부 "회전 방향 커맨드에 따라 실시간 제어 중 phase를 어디서 시작할지"(런타임 애니메이션 블렌딩) 얘기지, 학습 에피소드 초기화 얘기가 아님. RSI는 DeepMimic(Peng et al. 2018, 이 리워드 계보의 더 깊은 원조)에서 온 기법이지 Disney 자체 레시피가 아님.

사용자 반응: "디즈니의 그 간단한 모방학습만 쓰고싶은데 너무 거추장스러운거 다는게 아니라..." — RSI 같은 Disney 원본에 없는 추가 기법을 계속 쌓는 것에 대한 우려. 명확화 질문(AskUserQuestion)을 시도했으나 사용자가 거부, 텍스트로만 응답하라는 시스템 지시가 내려와 직접 해석해서 진행:
- **RSI는 끔** (Disney 원본엔 없음)
- **접촉기반 종료조건 + 크로치 방지(높이비율 0.75, 무릎/발목 접촉)는 유지** — 이건 Disney 논문에 실제로 나오는 방식(머리/몸통 접촉시 종료)이라 "거추장스러운 추가"가 아니라 오히려 Disney 방식에 더 맞춘 것
- **A20J5 가중치도 유지** — alive_scale=20은 애초에 Disney 원본 숫자 그대로

**구현**: RSI 코드를 삭제하는 대신 `cfg.use_rsi` 플래그로 토글 가능하게 만듦 (커밋 `218db78`/`ca28e5d`) — `use_rsi=False`면 `_reset_idx()`가 항상 phase=0으로 리셋하고 다리 관절도 기본자세로 스폰(v1~v5 원래 동작과 동일). 신규 config `JoystickEnvCfg_A20J5_NoRSI` (gym task `Isaac-OpenDuckMini-Joystick-A20J5NoRSI-v0`) 등록. 코드 삭제가 아니라 토글로 만든 이유: "RSI 켠 버전"(v6/v7)과 "Disney 원본대로(RSI 없음)"를 같은 코드베이스에서 직접 비교하기 위함.

**다음**: `imitation_v7`(RSI 켬) 완주 대기 중 — 완료되면 eval+시각확인 후, 곧바로 짧은 검증(num_envs=64, 150~200 iter)으로 no-RSI 경로에 새 버그 없는지 확인한 뒤 `imitation_v8` = A20J5NoRSI(Disney 원본 방식) 본학습 시작 예정. v7(RSI)과 v8(No-RSI)을 직접 비교해서 RSI가 실제로 도움이 되는지 처음으로 검증하는 실험.

## imitation_v7 결과 — FAIL, num_envs 스케일업이 오히려 악화 (2026-07-28 05:09, iteration 2999 완주)

**eval 결과 (`model_2999.pt`)**:
```
worst-case upright (마지막 200스텝): -0.0025 (v6의 -0.0040보다도 나쁨, 거의 완전 sideways)
mean leg-joint ROM: 1.0561 rad (v6=1.3072보다 낮지만 이전 런들(0.42~0.56)보다는 여전히 큼)
lin-vel tracking error: 0.2891 m/s (v6=0.4304보다는 개선, v4/v5보다는 나쁨)
foot-contact toggle: 90.6회/발/10초 — v6(29.4)보다 3배 악화, v3(79.1)/v4(78.9) 수준으로 회귀

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping)
```
**판정: FAIL, v6보다 명확히 악화.** 예상 밖의 결과 — "v6가 안 좋았던 건 num_envs=512로 인한 데이터 부족"이라는 가설을 반박함. num_envs를 512→4096(8배)로 늘렸는데 toggle이 오히려 3배 나빠짐. 학습 로그상 최종 에피소드 길이(89~91)는 v6(64~82)보다 오히려 길었는데, eval에서는 더 불안정하게 나온 것도 흥미로운 불일치 — 학습 중 수집된 노이즈 있는 배치 통계와 실제 정책 품질이 어긋날 수 있음을 시사. num_envs 자체의 인과 효과인지, 단순 학습 런간 시드/변동성인지는 이 실험만으론 확정 불가 (같은 설정 반복실험 없이 num_envs만 바꿨으므로). 수치가 명확한 FAIL(v6보다 악화)이라 시각 확인 생략, 바로 기록.

## Run 13 — `imitation_v8`: RSI 끄고 Disney 원본 방식 (2026-07-28)

계획대로 진행. RSI가 도움이 되는지 불확실해진 상황(v6는 개선, v7은 악화)이라 더더욱 "RSI 없는 버전"과의 직접 비교가 필요해짐. `use_rsi=False`(A20J5NoRSI) 검증(num_envs=64, 200 iter — 트레이스백 없음, 에피소드 길이 40~45 정상 범위) 통과 후 본학습(`num_envs=4096`, `--headless`) 시작, ETA~2h51m. run dir `2026-07-28_05-14-05_imitation_v8`.

**학습 중 관찰**: 학습 로그 지표(에피소드 길이, value_function loss)가 v6/v7보다 훨씬 좋아 보였음 — iter~700대에서 에피소드 길이 60.51(v6=38.78, v7=32.72), value_function loss는 0.06~0.14 사이로 학습 끝까지 매우 낮고 안정적(v6/v7은 같은 구간에서 1.2~2.8까지 올라갔었음).

## imitation_v8 결과 — FAIL, 학습 지표는 최고였지만 eval은 v7과 비슷하거나 더 나쁨 (2026-07-28 07:37, iteration 2999 완주)

**eval 결과 (`model_2999.pt`)**:
```
worst-case upright (마지막 200스텝): -0.0028
mean leg-joint ROM: 0.9768 rad
lin-vel tracking error: 0.3546 m/s
foot-contact toggle: 96.7회/발/10초 — v7(90.6)보다도 살짝 나쁨, v6(29.4)보다 훨씬 나쁨

STANDING RESULT: LIKELY STILL COLLAPSED/UNSTABLE
WALKING RESULT:  LIKELY REWARD-HACKING (standing/twitching, not stepping)
```
**판정: FAIL.** 학습 중 지표(에피소드 길이·value_function loss)는 v6/v7보다 압도적으로 좋아 보였는데, eval에서는 v7과 비슷하거나 오히려 더 나쁜 결과 — **학습-타임 지표와 eval-time 실제 정책 품질이 괴리되는 패턴이 세 번째로 재확인됨.**

### RSI on/off 종합 결론 (v6/v7/v8 3자 비교)

| run | RSI | num_envs | toggle | 판정 |
|---|---|---|---|---|
| v6 | 켬 | 512 | **29.4 (최고)** | FAIL |
| v7 | 켬 | 4096 | 90.6 | FAIL |
| v8 | 끔 | 4096 | 96.7 (최악) | FAIL |

**v7↔v8(num_envs 동일, RSI만 다름)만 놓고 보면 RSI가 켜진 v7(90.6)이 꺼진 v8(96.7)보다 미세하게 나음** — RSI가 아주 약간은 도움이 됐다는 뜻일 수 있으나, 차이(90.6 vs 96.7, 6.7%)가 크지 않아 노이즈 범위일 가능성도 있음. 반면 **압도적으로 제일 좋았던 v6는 RSI on + num_envs=512** 조합 — RSI 유무보다 num_envs=512(적은 병렬환경)라는 조건 자체가 핵심이었을 가능성을 시사하지만, v6는 반복실험이 없어 우연(시드 변동성)일 가능성도 배제 못 함.

**종합**: 3번의 시도(v6/v7/v8) 전부 FAIL, "RSI를 끄고 Disney 원본대로 단순화"해도 개선 안 됨. 사용자에게 이 결과를 명확히 보고하고 다음 방향 논의 필요 — 자동으로 또 다른 대규모 실험을 시작하지 않고 대기.

**v7 WebRTC 육안 확인**: 사용자 직접 관찰 — "그냥 꼬꾸라지는데" — toggle 90.6 FAIL 판정과 정합되는 시각적 확인.

**다음**: 사용자 요청으로 `reward_breakdown.py`를 GPU 여유 상태에서 재시도(v8 체크포인트 대상) — 각 리워드 항목의 실제 스텝당 기여도와 클램프(`clip(reward,0,10000)`)가 얼마나 자주/얼마나 깎아먹는지 실측해서 리워드 계수 재조정(shaping)의 근거로 삼을 예정. 이전 시도들(v5 대상) 전부 원인불명 크래시로 실패했었음 — 이번엔 학습 프로세스 없이 GPU 완전 유휴 상태에서 재시도.
