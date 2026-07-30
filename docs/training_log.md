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

## 리워드 항목별 실측 → 쉐이핑 수정 → `imitation_v9` (2026-07-28)

사용자 요청("리워드 계수 쉐이핑해봐, 각 리워드 벡터가 실제로 무슨값이 나오는지 기록하고").

**`reward_breakdown.py` 6회 연속 크래시 → 재작성**: `parse_env_cfg` 도입 후 원인불명으로 매번 조용히 죽음(OOM/시그널 흔적 없음). 안정적으로 작동해온 `contact_diagnostic.py`의 뼈대를 그대로 재사용한 `scripts/reward_breakdown_v2.py`를 새로 작성 → 즉시 성공. 교훈: 검증된 스크립트 패턴에서 벗어나지 말 것.

**측정 결과 (imitation_v8 최종정책, 8 env × 150 step = 1200 env-step)**:
```
tracking_lin_vel   +0.0201/step
tracking_ang_vel   +0.0115/step
torques            -0.0006/step
action_rate        +0.0000/step
alive              +0.4000/step   <- 가장 큰 양수
stand_still        -0.0048/step
imitation          -1.0112/step   <- 나머지 전부를 압도
SUM (클램프 전)     -0.5850/step

클램프로 0이 된 스텝: 979/1200 (81.6%)
클램프가 깎아먹은 총량: 스텝당 평균 0.664
```

**결정적 발견**: imitation이 "안 되는" 게 아니라 **너무 세서** 합계를 음수로 끌었고, `_get_rewards`의 `clip(reward, 0, ...)`이 그걸 0으로 만들어 **전체 스텝의 81.6%에서 학습 신호가 완전히 소멸**하고 있었음. 리워드가 계속 한 자릿수로 낮게 찍힌 이유가 이것.

**근본 원인**: `reward_imitation()` 안에서 **관절각 항만 유일하게 상한이 없음**(`-err²×w`). 같은 함수의 속도추종 항들은 전부 `exp(-err)`로 [0,1]에 갇혀 있는데, 관절 10개 오차가 쌓이면 이 항 혼자 다른 항 전부를 압도.

**조치** (`JoystickEnvCfg_A20J5_Bounded`, 측정값에서 유도한 3개 연동 변경):
1. `imitation_bounded_joint_pos=True` — 관절각 항도 `exp(-w×err²)`로 [0,1] 제한
2. `imitation_w_joint_pos` 5.0 → **0.25** — exp 안에서 w는 가중치가 아니라 **예민도**. 5로 두면 v8 실측 오차(sum err²≈10.6)에서 `exp`가 0.0000으로 포화돼 기울기가 사라짐(다른 방식의 학습불능). 0.25면 그 지점에서 0.07, 개선될수록 0.61(err²=2)→0.88(err²=0.5)로 부드럽게 상승
3. `imitation_scale` 1.0 → **4.0** — 제한을 걸면 imitation 상한이 ~6으로 줄어 alive(20)에 눌리고, v1~v5를 망친 "가만히 서있기" 리워드해킹이 재발할 위험. ×4로 상한(~24)을 alive와 맞춤

RSI는 끈 상태 유지(v7 vs v8에서 유의미한 차이 없었고 사용자가 Disney 원본 방식 선호) — 이번 변경은 리워드 쉐이핑만 격리해서 검증.

**검증 (64env, 200iter)**: 크래시 없음, 에피소드 길이 ~50, **리워드 13~15**(이전 런들은 같은 초기 단계에서 0.3~1.3).

**수정 후 재측정 (검증 체크포인트 model_199)**:
```
imitation          -1.0112 -> -0.0414/step
SUM (클램프 전)     -0.5850 -> +0.3746/step
클램프된 스텝        81.6%  -> 0.0%
```
매 스텝 리워드가 양수가 되어 정책이 모든 경험에서 학습 신호를 받게 됨. (imitation이 아직 살짝 음수인 건 관절**속도** 항이 여전히 상한 없는 이차 페널티이기 때문 — 크기가 작아 합계를 음수로 끌지는 못함, 향후 필요시 같은 방식으로 처리 가능.)

## Run 14 — `imitation_v9` (A20J5Bounded, 리워드 쉐이핑 적용) 시작

`num_envs=4096`, `--headless`, `max_iterations=3000`, ETA~2h46m. run dir `2026-07-28_13-37-08_imitation_v9`. 초기 리워드 15.81(v1~v8은 전부 한 자릿수)로 시작.

## Run 15~16 — `imitation_v10` / `imitation_v11` 전부 FAIL, 그리고 **진짜 근본원인 발견** (2026-07-28)

### v9/v10의 실패와 사용자 육안 확인

v9(A20J5Bounded) 완주 후 v10까지 돌렸으나 둘 다 FAIL. 사용자 WebRTC 관찰:

> "앞으로 가질 않네... 관절이 학습 자세에서 부들부들 거리고 안넘어 지려고 조금씩만 움찔거림. 발도 붙여진 상태"

리워드해킹 예측이 그대로 재현됨.

### 근본원인 — **액션 공간이 레퍼런스 자세에 물리적으로 닿지 못했음** (8σ 문제)

사용자 가설("이미테이션과 현실의 링크나 조인트 위치가 안 맞는 거 아냐? 확인해봐")에서 출발해 특정.

관절 목표값은 `target = default_joint_pos + action × action_scale(0.25)`으로 계산된다. 그런데
- `default_joint_pos` = HOME = **전부 0** (다리 쭉 편 자세)
- 레퍼런스 보행이 요구하는 무릎 각도 ≈ **2.03 rad**

→ 필요한 액션 = 2.03 / 0.25 = **8.12**. PPO 정책의 초기 가우시안은 `init_noise_std=1.0`이므로 이는 **8σ 바깥**. 탐험으로 도달 불가능한 영역이었다.

**v1~v9의 리워드 계수 튜닝은 전부 전제가 무너진 상태에서 한 것.** 어떤 가중치를 줘도 로봇은 레퍼런스 자세를 만들 수 없었으므로 아홉 번 똑같이 실패한 것이 당연했다.

### 조치 — HOME과 READY 분리

사용자 제안("홈은 0이고, 학습 시작할 때나 실제로 작동할 때 모드를 변경하면서 딱 준비상태가 되는 거지")대로 두 자세를 개념적으로 분리:

- **HOME** = 실제 하드웨어의 물리적 휴지 자세 (전부 0, base 0.193 m)
- **READY** = 레퍼런스 보행의 평균 자세 = 학습용 `init_state`/`default_joint_pos`
  (hip_pitch ~1.11, knee ±2.03, ankle ±0.98, base 0.121 m — `scripts/settle_pose.py` 실측)

`joint_order.py`에 `READY_JOINT_POS`/`READY_BASE_HEIGHT` 추가, `robot_cfg.py`가 이걸로 초기화.

**효과 실측**: joint_pos 오차 79° → **7.8°**, joint_pos_rew 0.012 → **0.510**, imitation −1.011 → **+0.290**/step, 클램프 스텝 81.6% → **0%**, 필요 액션 8.1σ → **1.30σ**.

부수적으로 잠복 버그 하나 제거: `_reset_idx`의 관절 랜덤화가 **곱셈** 방식이라 HOME(=0)에서는 조용한 no-op였고, READY의 2.03 rad 무릎에는 [1.0, 3.0]으로 흩뿌려 관절한계(±2.094)를 넘길 뻔했다. 덧셈(±0.05 rad)으로 교체.

**레퍼런스는 건드리지 않음** — 웅크린 보행은 의도된 것이라는 사용자 지시("그건 레퍼런스를 우리가 그렇게 잡은 건데... 이미 관절각도 제한 같은 건 레퍼런스에 맞게 수정함"). `walk_com_height`를 임의로 올렸던 변경은 되돌림.

### v11 (`JoystickEnvCfg_Walk`) — 리워드해킹 대응 + 다리만 학습

항목별 감사로 '가만히 서있기'가 alive 100% / ang_vel_xy 100% / lin_vel_z 82% / contact 67% / joint_pos 51%를 공짜로 받는 구조임을 특정하고:
1. `swing_only_contact` — 레퍼런스가 들라는 발을 실제로 들었을 때만 보상(서있으면 0)
2. `alive_scale` 20 → 10
3. `tracking_lin_vel_scale` 2.5 → 10.0
4. `lock_head_joints=True` — 사용자 제안대로 머리 4관절 잠그고 다리 10개만 학습. READY의 목은 Z자(neck +45°, head +45° — 이 URDF에선 **같은 부호**가 상쇄 조합, 시각 확인 완료)

검증: 서있기 0.734 → 0.583/step, lock_test 액션 ±3.0에도 머리 0.034 rad(다리 0.178 rad).

**결과: FAIL.** 사용자 지적("지금 리워드가 v10과 같은 양상")이 정확했다:

| iter | v10 스텝당 | v11 스텝당 |
|---|---|---|
| 80 | 0.409 | 0.252 |
| 160 | 0.339 ↓ | 0.238 ↓ |
| 240 | 0.313 ↓ | 0.223 ↓ |
| 320 | 0.304 ↓ | 0.216 ↓ |

두 런 모두 에피소드 길이는 늘고 스텝당은 계속 하락 — 절대값만 낮아졌을 뿐 행동 양상 동일. 시각 확인: **"또 부들부들거림, 발 안뜸 혹은 아주 약간 뜸, 걸을려고 안 함"**. 3시간 더 돌려 같은 결말을 확인하는 대신 iter ~320에서 중단하고 진단.

## 리워드 항목별 재측정 → **lin_vel_xy의 무딘 민감도가 진범** (2026-07-28)

`scripts/imit_internals2.py` 신규 작성(기존 `imit_internals.py`에 Walk 태스크 매핑, `swing_only_contact` 반영, **발 토글/실제 속도 측정** 추가). v11 `model_300` 측정:

```
실제 속도 0.064 m/s   (레퍼런스 0.265 m/s)  <- 거의 안 움직임
lin_vel_z  +0.954/1.0    ang_vel_xy +0.220/0.5   lin_vel_xy +0.556/1.0
joint_pos  +0.379/1.0    ang_vel_z  +0.258/0.5   contact    +0.205/~0.63
joint_vel  -0.056        <- 유력 후보로 지목했던 항. 전체 2.52의 2%로 무관
발 접촉 토글 0.72/step   (미세 떨림이 토글로 잡힘, 실제 보행 아님)
```

**서있는 채로 이미테이션 리워드의 ~92%를 수거 가능.** 결정타는 `lin_vel_xy` — 걷기를 값매기는 **유일한** 항인데, `exp(−k·err²)`의 k=8에서는 레퍼런스 속도를 통째로 틀려도(err² = 0.265² = 0.070) `exp(−0.56) = 0.57`점이 나온다. **리워드가 서있기와 걷기를 구분하지 못했다.**

기각된 가설 두 개(기록용):
- `joint_vel`이 상한 없는 이차 페널티라 움직임을 억제한다 → **틀림**. 실측 −0.056으로 무의미.
- 정책이 보행 위상을 관측 못 해 평균 자세에 머문다 → **틀림**. `imitation_phase`(cos/sin)가 관측에 정상 포함(`joystick_env.py:289–304`).

## Run 17 — `imitation_v12` (`JoystickEnvCfg_Walk2`) 시작

측정값에서 직접 유도한 변경:

| 항목 | v11 | v12 | 근거 |
|---|---|---|---|
| `k_lin_vel_xy` | 8 | **20** | 제자리 0.56 → 0.25, 정상보행은 ~1.0 유지 |
| `w_lin_vel_z` | 1.0 | **0.1** | 0.954/1.0 공짜, 최대 freebie |
| `w_ang_vel_xy` | 0.5 | **0.1** | 레퍼런스 자체 분산 0.0000 = 구조적 변별력 0 |
| `w_contact` | 1.0 | **2.0** | 서있기로 못 버는 유일한 항인데 가장 작았음 |
| `alive_scale` | 10 | **3** | 걷든 서든 동일한 값이라 변별 신호를 희석 |
| `use_rsi` | 끔 | **켬** | 아래 참조 |

**RSI 복구 근거**: v7/v8의 "RSI 유무 차이 없음" 결론은 `default_joint_pos`가 아직 HOME이던 시절, 즉 레퍼런스 자세가 정책의 액션 분포 8σ 밖에 있던 조건에서 나온 것이다. RSI가 초기화해 넣던 걸음 중간 자세는 **어떤 액션으로도 유지 불가능**했으므로 그 실험은 아무것도 측정하지 못했다. READY로 도달 가능해진 지금, 랜덤 위상의 걸음 중간에서 시작하는 것은 현재 갇힌 국소최적(= 보행의 평균 자세인 READY에 머무는 것. 위상이 어긋난 어설픈 추종은 가만히 있는 것보다 점수가 낮으므로 합리적 선택이 된다)의 표준 탈출법이다.

`k=20`은 "정확한" 값(제자리를 0.05로 만들려면 k≈43)이 아니라 의도적 절충 — k≈43은 반속도 이하 전 구간의 기울기까지 평탄하게 만들어, `w_joint_pos=5`에서 이미 겪은 exp 포화 실패를 재현한다.

**검증** (`reward_at_ready.py`, Walk2): 서있기 리워드 0.734(v10) → 0.583(v11) → **0.444(v12)**, 클램프 0.1%. Mac 단위테스트 7/7 통과, `reward_imitation` 기본 인자 하위호환 확인(v1~v11 재현값 5.0 불변).

**런 조건**: 사용자 요청으로 headless가 아닌 **`--livestream 2` 시각화 학습**. `num_envs=4096`은 렌더링 시 CUDA illegal memory access(RTX 5080 16GB VRAM 한계)로 즉사하여 **1024**로 조정. `max_iterations=3000`, ETA ~4h30m.

**판정 기준**: 서있기 = 0.444/step 실측. 스텝당이 0.40~0.50에서 정체하며 에피소드 길이만 늘면 해킹 재발, **≥0.55면 보행 신호**.

## Run 18~24 — 비대칭 크리틱과 gamma: 리워드가 아니라 가치 추정이 문제였다 (2026-07-29)

### 전환점 — 사용자의 질문

리워드 계수를 열한 차례 조정했으나(v10~v19) 매번 국소 개선에 그쳤다. 방향을 바꾼 계기는
사용자의 질문이었다: **"깃허브랑 보상 구조, ANN 등등은 같지?"**

대조해보니 upstream(Open_Duck_Playground)과 10개 축에서 달랐고, 그중 하나가 결정적이었다.

### 발견 1 — 비대칭 크리틱 (imitation_v20)

upstream은 가치 네트워크에 `privileged_state`를 준다(mujoco_playground가
`value_obs_key="privileged_state"`로 설정). 이 포팅은 `state_space=0`으로 그걸 생략했고 —
"v1 포팅의 의도적 단순화"라고 주석까지 달려 있었다 — **크리틱이 정책과 똑같은 노이즈 섞인
101차원만 보고 있었다.**

문제는 이렇다. 정책이 제한된 관측만 보는 건 의도된 것이다(실제 로봇에 그 센서만 있으니까).
그러나 크리틱은 학습에만 쓰이고 로봇에 탑재되지 않는다. 크리틱까지 같은 시야로 묶으면,
동일한 관측이 들어와도 실제로는 미끄러지는 중인지 정상 보행 중인지 구분하지 못한 채 평균적인
가치를 추정한다. PPO는 어드밴티지 `A = R − V(s)`로 학습하므로 **상황에서 기인한 리턴의 편차가
전부 어드밴티지 노이즈로 유입된다.** 리워드를 아무리 정교하게 설계해도 그 신호가 오염된
어드밴티지를 거쳐 전달되는 구조였다 — 즉 **리워드 설계보다 상류에 병목이 있었다.**

크리틱 전용 관측을 205차원으로 구성했다: 정책 관측 + 무노이즈 센서 + 정책이 못 보는 실제
몸통 속도 + 관절 토크 + 발 속도 + 몸통 높이 + 36차원 레퍼런스 프레임 전체.
(upstream 대비 누락: `feet_air_time`은 이 환경이 추적하지 않고, `imitation_i`는 위상으로 대체.)

**리워드를 한 글자도 바꾸지 않았으므로 스텝당 리워드를 직접 비교할 수 있었다** — 이전의 모든
버전 간 비교가 스케일 차이로 오염돼 있던 것과 다르다:

| iter | v17 스텝당/에피 | v20 스텝당/에피 |
|---|---|---|
| 120 | 0.1135 / 133 | **0.2244 / 297** |
| 240 | 0.0994 / 143 | **0.2973 / 372** |

전진 고정 재생에서 `vy = −0.005` (v17은 ±0.15, v16은 ±0.30). 사용자 판정: **"그거 빼면 제일
안정적임 지금까지"**.

### 부수 발견 — 스폰 높이와 RSI가 서로를 모르고 있었다

재생 시작마다 로봇이 "뿅" 튀어오르는 것을 사용자가 관찰했다. `_reset_idx`는 루트 z를 항상
`READY_BASE_HEIGHT`로 두는데 RSI는 다리 관절만 랜덤 위상의 레퍼런스 자세로 덮어쓴다 —
두 코드가 서로를 참조하지 않는다. 맥에서 pinocchio 순기구학으로 위상별 필요 높이를 재니
116.7~126.4 mm로 9.7 mm 변동했고, 발이 가장 낮은 위상에서 **지면을 5.4 mm 파고들어** PhysX가
로봇을 튕겨내고 있었다. `SPAWN_BASE_HEIGHT = 0.130`을 `READY_BASE_HEIGHT`(종료 판정 기준)와
분리했다. **효과: 스텝당은 그대로, 에피소드 길이 +35%** (218→296, 372→485).

### 발견 2 — gamma (imitation_v24)

v20은 upstream의 PPO 하이퍼파라미터도 함께 이식했는데 이 부분은 역효과였다. brax의
`num_minibatches=32`를 그대로 옮기면 rsl_rl에서는 iteration당 128회 업데이트가 되고(v17은 20회),
KL이 초과해 rsl_rl의 adaptive 스케줄이 lr을 최저 한계 1e-5까지 깎았다. brax는 lr 고정에 KL
적응이 없으므로 **하이퍼파라미터는 PPO 구현 간에 이식되지 않는다.**

원인 분리를 시도하며 한 번 틀렸다. v22와 v20b를 "minibatch 차이"로 설명했지만 실제로는 다섯
개가 함께 달랐고(steps 20/24, minibatch 32/4, epochs 4/5, gamma 0.97/0.99, lr 초기값),
**v23에서 minibatch를 4→16으로 네 배 올렸는데 곡선이 v22와 완전히 겹쳤다**(iter 160에서 둘 다
0.192). minibatch는 무관했고, 앞선 귀속은 근거가 없었다.

남은 후보 gamma를 단독으로 시험한 것이 v24다:

| iter | v22 (γ=0.99) | v24 (γ=0.97) | v20b |
|---|---|---|---|
| 160 | 0.192 | **0.296** | 0.268 |
| 400 | 0.228 | **0.331** | 0.324 |
| 560 | — | **0.348** | 0.350 |

**gamma 하나로 +53%.** 그리고 v20b와 달리 lr이 살아 있다(4.4e-4~2.0e-4 vs v20b의 1e-5 고착).
그래서 v20b가 동결된 0.345를 넘어 **0.366까지 갔다.**

**왜 gamma가 이렇게 컸는지가 발견 1과 같은 이야기다.** 0.97의 유효 시간지평은 약 33스텝이고
이 로봇의 보행 주기는 27스텝이다 — 딱 한 주기. 0.99는 100스텝, 세 주기 반을 본다. 주기적
과제에서 크리틱에게 세 주기 앞을 추정하라고 하면 가치 추정이 흐려진다. **두 발견 모두
"가치 추정을 정확하게 만드는 것"이지 리워드를 정교하게 만드는 것이 아니었다.**

### v24 최종 측정 (model_2999, 명령 고정 롤아웃)

| 명령 | 실제 | 달성률 | v20b |
|---|---|---|---|
| 전진 +0.15 | +0.148 | **99%** | 87% |
| 후진 −0.15 | −0.131 | 87% | 99% |
| 좌 +0.20 | +0.142 | **71%** | 58% |
| 우 −0.20 | −0.162 | **81%** | 67% |

**정지 명령**(그동안 측정 도구에서 빠져 있던 조건, 이번에 추가):

| | v20b | v24 |
|---|---|---|
| 잔류 속도 | 109 mm/s | **46 mm/s** |
| 요 흔들림 \|평균\| | 0.692 rad/s | **0.142** |
| 발 토글 | 155/10s | **58** |
| 발 접지 | 1.37/2 | **1.84/2** |

주기성 판정 WALKING-LIKE, 우세 주파수 1.90 Hz(기준 1.85), 관절 RMS 6.5~9.2°.

### 정리

| 변경 | 효과 |
|---|---|
| 비대칭 크리틱 (101→205) | 스텝당 3배 — 결정적 |
| gamma 0.99 → 0.97 | +53%, lr 동결 해소 |
| entropy 0.01 → 0.005 | +57% (v21→v22) |
| 스폰 121 → 130 mm | 에피소드 길이 +35% |
| minibatch 4/16/32 | 무관 (v23이 확인) |

**남은 과제:** 좌우 보행이 71~81%로 전진(99%)보다 낮고, 정지 시 46 mm/s 잔류 표류가 있다.
다지형(경사·요철)은 아직 시작하지 않았다 — 평지 명령 추종이 안정된 뒤가 맞다.

## v25 — path frame, 그리고 자기충돌 조사 (2026-07-30)

### imitation_v25 — 직진성을 위한 path frame

v24를 명령 순환 재생으로 보니 직진을 못 했다. 전진 명령만 주는데 요가 순간
±1.5 rad/s로 요동쳤고, 좌우 명령에서는 옆으로 가는 대신 비틀거리며 회전했다.
평균은 0에 가까워 측정치(좌 71% / 우 81%)로는 드러나지 않던 결함이다.

원인은 명령이 순수 rate라는 것이었다. `yaw_rate=0`은 "지금 회전하지 마라"이지
"원래 방향으로 돌아와라"가 아니고 `vy=0`도 마찬가지다. 한 번 휘면 그 사실이
정책의 관측 어디에도 없어서 되돌릴 수단이 없다 — 볼 수 없는 것을 요구하고 있었다.

Disney BD-X는 path frame으로 푼다: "a path frame that integrates these velocity
commands over time"를 유지하고 정책의 상태를 그 프레임 기준으로 표현한다.
적분된 경로 대비 **횡방향 오차와 방향 오차**를 관측(3차원)과 리워드에 추가했다.
종방향 오차는 뺐다 — path frame은 *명령* 속도를 적분하므로 로봇이 조금이라도
느리면(0.148 vs 0.15) 무한히 쌓이고, 그 보정을 보상하면 뒤처졌을 때 무리하게
가속하는 쪽으로 학습된다.

**v25 @1500 (v24는 @2999, 절반 학습량):**

| 명령 | v24 @2999 | v25 @1500 |
|---|---|---|
| 전진 | 0.148 (99%) | 0.133 (89%) |
| 후진 | −0.131 (87%) | **−0.148 (99%)** |
| 좌 | 0.142 (71%) | **0.164 (82%)** |
| 우 | −0.162 (81%) | **−0.179 (90%)** |

정지가 특히 좋아졌다: 잔류 속도 46 → **2 mm/s**, 발 토글 58 → **20/10s**,
발 접지 1.84 → **1.98/2**. 거의 양발을 딛고 선다.

경로 이탈(v25에서 처음 측정 가능): 방향 오차 RMS가 전 방향 **2.9~4.7°**,
최대 12.2°. v24에서 순간 ±1.5 rad/s(≈86°/s)로 요동치던 것과 근본적으로 다르다.

남은 결함: 좌측 이동의 횡방향 이탈이 247 mm로 우측(143 mm)의 1.7배다.
발 태스크스페이스에서 본 "오른발이 왼발보다 22% 크게 움직임"과 같은 뿌리일 수 있다.

### 자기충돌 — 조사 후 보류

사용자가 재생에서 다리와 몸통이 겹치는 것을 보고 제기했다. 논문 확인 결과
Disney는 자기충돌을 켤 뿐 아니라 종료 조건으로 쓴다. 우리는 계속 꺼져 있었고,
v4의 "플랭크" 자세가 접촉력 0.000 N으로 모든 종료를 빠져나간 것도 이 때문이었다.

켜보니 **에피소드가 1스텝 만에 끝났다.** pinocchio 정확 메시로 재보니 물리가
아니라 충돌 형상 근사의 문제였다 — READY에서 비인접 링크쌍 91종 중 관통 0개,
최소 간격 10.2 mm인데 PhysX는 수천 N을 만들었다. IsaacLab의 `convert_urdf.py`가
`UrdfConverterCfg.collider_type`을 설정하지 않아 기본값 `convex_hull`이 쓰인 탓이다.

`scripts/convert_urdf_cd.py`로 convex_decomposition 재변환까지 해봤다.
`neck_pitch↔neck_yaw`(4340 N)는 해소됐지만 두 쌍이 남았다 —
`head_pitch↔xm430`(1847 N, 정확 메시로는 13.8 mm 여유),
`trunk↔roll_to_pitch`(1089 N, 11.3 mm). 둘 다 체인에서 두 관절 떨어져 있어
PhysX가 자동 제외하지 않는데, **설계상 관절 하우징이 부모 부품 안에 끼워지는
구조라 어떤 볼록 근사로도 겹친다.** 전역 자기충돌은 이 기구 구조에서 쓸 수 없고,
Disney가 머리↔몸통 한 쌍만 집어 쓴 것도 같은 이유로 보인다 — 전역이 아니라 선별이다.

보류하되 실측은 남긴다 (v25 model_1500, 정확 메시):

| | 다리↔몸통 최소 | 접촉 비율 |
|---|---|---|
| 레퍼런스 | 7.1 mm | **0.0%** |
| 정책 | 0.0 mm | **56.7%** |

**레퍼런스는 한 번도 안 닿는데 정책은 절반 이상 닿는다 — CAD나 레퍼런스가 아니라
정책 결함이다.** 접촉 시점의 관절 오차가 원인을 특정해준다:

| 관절 | 접촉 시 | 비접촉 시 |
|---|---|---|
| right_hip_roll | −8.0° | +0.9° |
| left_hip_yaw | −8.9° | −1.5° |
| right_hip_yaw | 메시 간격과 상관 −0.540 | |

무릎·발목은 거의 무관하다. **고관절 roll/yaw가 레퍼런스에서 8~9° 벗어나 다리를
안쪽으로 돌리는 것**이 접촉의 원인이다. 실기 이식 전에는 반드시 닫아야 한다.

곁가지로 확인한 것: 런타임에 쓸 싼 대체 지표(링크 원점 간 거리)는 못 쓴다.
레퍼런스든 정책이든 원점 거리는 115 mm로 거의 같은데 메시 간격은 0~17 mm로
갈린다(상관 −0.333). 원점 거리는 고정 형상이 지배해 실제 간격을 반영하지 않는다.

---

## imitation_v27 — 정책 관측에 중력 방향 (2026-07-30)

v26이 v25@1500의 두 배(3000 iter)를 학습하고도 명령 추종이 나빠진 것을 파고든
결과다. 재보니 몸통이 안 잡혀 있었다: 몸통 roll RMS 7.20°(보행 중 8.2°, 정지
4.1°), 모방 리워드의 `ang_vel_xy` 항이 최댓값의 **3.6%**로 사실상 0점.
고관절 roll/yaw 이탈은 3.9/4.5°로 오히려 무릎·발목(5.3°)보다 작았다 —
**고관절이 특별히 나쁜 게 아니라 몸통 자세 자체가 안 잡힌다.**

그런데 정책 관측에 중력 방향이 없었다. 자이로(각속도)와 가속도계만으로 기울기를
유추해야 하는데, 가속도계는 중력과 실제 가속이 섞인다. **크리틱은 노이즈 없는
중력을 이미 보고 있었다** — 정책만 못 보는 비대칭이었다.

v27은 그 3차원만 더한다. 리워드는 한 글자도 안 건드렸다.

**상류를 빠뜨린 게 아니다.** Playground도 gravity를 계산만 하고 state에 안 넣는다
(joystick_env.py 모듈 docstring에 원래 적혀 있던 사실). v27은 위 측정을 근거로
한 **의도적 이탈**이다. 실기에서도 IMU로 추정 가능한 양이라 sim2real 관점에서
정당하다.

### 결과 — 같은 학습량(1500 iter)에서 비교

v26이 준 교훈이 "학습량이 다르면 비교가 성립하지 않는다"였으므로 1500에서 맞댔다.
그리고 **%보다 오차로 본다** — v26@1500은 전진 0.169로 오버슈트해서 "113%"가
되는데, 그게 좋다는 뜻이 아니다.

| 명령 | v26@1500 오차 | v27@1500 오차 |
|---|---|---|
| 정지 | 0.006 | 0.010 |
| 전진 | 0.024 | **0.005** |
| 후진 | 0.030 | **0.017** |
| 좌 | 0.038 | **0.037** |
| 우 | **0.031** | 0.033 |
| 회전 | 0.021 | **0.015** |
| **평균** | **0.0250** | **0.0195** |

**평균 추종 오차 22% 감소.** 그리고 노린 지표가 정확히 움직였다:
모방 리워드의 `ang_vel_xy` 항 0.0184 → **0.0250 (+36%)**.

### 그런데 몸통 안정화 자체는 미미하다

| | v26 | v27@1500 |
|---|---|---|
| 몸통 roll RMS | 7.20° | **6.77°** (−6%) |
| 몸통 pitch RMS | 3.88° | 4.79° (**+23%, 악화**) |
| 고관절 roll 이탈 | 3.91° | 5.03° (악화) |
| 고관절 yaw 이탈 | 4.47° | **2.90°** (−35%) |
| 무릎/발목 이탈 | 5.33° | **4.56°** (−14%) |

roll은 조금 좋아졌지만 pitch는 나빠졌다. **가설의 기전이 생각과 다를 수 있다** —
중력 관측이 몸통 자세를 잡아줘서 추종이 좋아졌다기보다, 다른 경로로 도움이 된
것일 수 있다. 추종 개선은 실측이지만 "몸통을 안정화해서"라는 설명은 확정이 아니다.

### 더 오래 학습하면 나빠지는 현상은 v27에서도 재현됐다 (약하게)

| | 평균 추종 오차 |
|---|---|
| v27@1500 | **0.0195** |
| v27@2000 | 0.0213 |

v25@1500 > v26@2999 에 이어 두 번째다. **이 리워드 구조에서는 오래 학습할수록
명령 추종을 잃는다.** 수렴 판정으로 iter 2082에서 멈췄는데(직전 500 iter +0.88%),
리워드 곡선은 평평해져도 행동은 계속 나빠지고 있었다는 뜻이다.

### 현재 최고

**v27@1500, 평균 추종 오차 0.0195** (`2026-07-30_06-01-03_imitation_v27/model_1500.pt`)

### 남은 가장 큰 결함: 좌우

v27@1500에서도 좌 0.037 / 우 0.033 으로, 전진(0.005)의 **7배**다. 여섯 방향 중
횡방향만 유독 나쁘고 이건 v13부터 계속 그렇다. 다음 실험은 여기를 친다.

---

## imitation_v28 — 로봇을 더 세운다 (2026-07-30)

사용자 요청. 기존 로봇은 121 mm 로 상당히 웅크리고 걸었다.
레퍼런스 `walk_com_height` 0.16 -> 0.175 로 재생성 → 안착 높이 **121 -> 136 mm (+12%)**.
무릎이 12.4° 펴졌다. 필요 최대 액션은 1.30 -> 1.46 (v1~v9 를 죽인 값은 8.1, 안전).

**측정을 두 번씩 했다.** 단일 측정은 못 믿는다 — 전진 오차가 v28@1500 에서
0.004 와 0.032 로 흔들렸다.

| 명령 | v27@1500 (2회) | v28@1500 (2회) |
|---|---|---|
| 정지 | 0.010 / 0.008 | 0.010 / 0.011 |
| 전진 | 0.005 / 0.005 | 0.004 / 0.032 |
| 후진 | 0.017 / 0.014 | 0.022 / 0.015 |
| **좌** | 0.037 / 0.042 | **0.018 / 0.016** |
| **우** | 0.033 / 0.036 | **0.021 / 0.015** |
| 회전 | 0.015 / 0.020 | 0.008 / 0.011 |
| **평균** | 0.0195 / 0.0208 | **0.0138 / 0.0167** |

### 확실한 것과 불확실한 것을 갈라 적는다

**확실: 횡방향 오차가 절반으로 줄었다.** 좌 0.040 -> 0.017, 우 0.035 -> 0.018.
두 번 다 같은 방향이고 잡음 폭보다 훨씬 크다. 이건 v13 부터 이어진 이 프로젝트
최대 결함이었고 인수인계 문서에도 미해결로 적혀 있었다.

**불확실: 왜 좋아졌는지.** 바꾼 것은 로봇 키뿐인데 좌우가 좋아졌다. 다리가
펴지면서 횡방향 지지 기저나 발 궤적 여유가 달라진 것으로 추측되지만 확인하지
않았다. 전진 오차는 양쪽 다 흔들려 개별 수치로 판단하면 안 된다.

**주의: v27 과 v28 은 리워드 총합을 비교하면 안 된다.** 레퍼런스가 다르므로
모방 리워드가 재는 대상 자체가 다르다("스케일 착시"의 변종). 레퍼런스와 무관한
명령 추종 오차로만 비교했다.

### 오래 학습하면 나빠진다 — 세 번째 재현

| | 평균 추종 오차 |
|---|---|
| v28@1500 | **0.0138 / 0.0167** |
| v28@2000 | 0.0160 |

v25@1500 > v26@2999, v27@1500 > v27@2000 에 이어 세 번째다. 수렴 판정으로
iter 2068 에서 멈췄는데(최근 500 iter +0.02%), **리워드는 평평해진 뒤에도
행동은 계속 나빠지고 있었다.**

### 현재 최고

**v28 의 `model_1500.pt`** — 평균 추종 오차 약 0.015
(`2026-07-30_08-19-45_imitation_v28/model_1500.pt` 계열, `odm list` 로 확인).

---

## imitation_v29 — 좌우 미러 손실 (2026-07-30) — **이득 없음**

v28 설정 그대로 + rsl-rl 5.0.1 의 `symmetry_cfg` 미러 손실(계수 0.5).
관측·리워드·레퍼런스 전부 v28 과 같고 손실 항만 하나 더했다.

미러 매핑은 가정하지 않고 레퍼런스 데이터에서 유도했다
(`scripts/diag/derive_mirror.py`). 순진한 "roll/yaw 부호 반전"은 hip_roll /
knee / ankle 세 관절이 틀리는데, 이 URDF 는 좌우 관절 축이 대칭이 아니기
때문이다(같은 자세에서 left_knee -1.785 vs right_knee +1.816).

### 결과 — v28 과 동률

| | 평균 추종 오차 (각 2회) |
|---|---|
| v28@1500 | 0.0138 / 0.0167 → **0.0153** |
| v29@1500 | 0.0177 / 0.0148 → 0.0163 |
| v29@1600 | 0.0152 / 0.0160 → 0.0156 |

**노린 표적(좌우)에서도 이득이 없다.** 표본 4개씩:

| | 좌 | 우 |
|---|---|---|
| v27 | 0.037 0.042 | 0.033 0.036 |
| **v28** | 0.018 0.016 | 0.021 0.015 |
| v29 | 0.028 0.030 0.016 0.013 | 0.015 0.016 0.022 0.041 |

v27 -> v28 의 횡방향 개선은 표본이 겹치지 않아 실재한다. v28 -> v29 는
평균이 그대로이고 **분산만 커졌다.**

### 분석 도중 내렸다가 철회한 결론

v29@1500 만 보고 "미러 손실이 오히려 좌우 대칭을 깨뜨렸다"고 적었다
(좌 0.028/0.030 vs v28 의 0.017). 그런데 v29@1600 에서는 좌가 0.016/0.013 으로
멀쩡하고 대신 **우가 0.041** 로 나빴다. **어느 쪽이 나쁜지가 측정마다 뒤바뀐다** —
계통적 결함이 아니라 정책이 비대칭 모드 사이를 오가는 것이다. 철회한다.

여기서 배운 것: **명령별 오차는 표본당 ±0.02 수준으로 흔들린다.** 한두 번
측정으로 특정 명령의 좋고 나쁨을 말하면 안 된다. 평균은 그보다 안정적이다.

### 왜 안 통했을까 (추측, 미검증)

미러 손실은 최고점 0.0188 에서 0.0131 까지 내려가고 멈췄다. 그 값을 RMS 로
환산하면 관절각 약 0.03 rad 인데, `derive_mirror.py` 가 잰 **레퍼런스 자체의
좌우 잔차가 0.02~0.05 rad** 다. 즉 레퍼런스가 완벽히 대칭이 아니라서 정책도
그 이상 대칭해질 수 없고, 미러 손실은 이미 바닥에 닿아 있었다. **강제할 대칭이
애초에 데이터에 없었던 셈이다.**

확인하려면 레퍼런스를 대칭화해서(좌우 평균) 다시 생성해 봐야 한다. 미착수.

### 현재 최고는 그대로 v28

**v28 의 `model_1500.pt`** — 평균 추종 오차 약 0.0153.

---

## v30 — 레퍼런스 높이 175 -> 190 mm (`JoystickEnvCfg_Taller`)

사용자 요청("robot height more up"). 레퍼런스 모션을 랩PC 에서 `walk_com_height`
만 올려 재생성하고(`scripts/setup/gen_reference_remote.sh`), `READY_BASE_HEIGHT`
0.1360 -> 0.1539, 스폰 0.1437 -> 0.1614 을 맞춰 붙였다. 3000 iter 완주.

### 결과 — 목적은 달성, 추종은 손해

**이 실험은 "얼마나 크게 서서 걷는가" 를 묻는데, 그동안 어느 측정에도 몸통
높이가 남지 않아 판정 자체가 불가능했다.** `gait_compare.py` 에 `base_h` 를
추가하고 두 버전을 같은 조건으로 다시 쟀다 (과도구간 100스텝 제외, 4 env).

| 명령 | v28 (ref_h175) | v30 (ref_h190) | 차이 |
|---|---|---|---|
| 정지 | 132.6 mm | 139.7 mm | +7.1 |
| 앞 | 133.5 mm | 155.0 mm | +21.6 |
| 뒤 | 150.8 mm | 166.5 mm | +15.7 |
| 좌 | 142.6 mm | 161.4 mm | +18.7 |
| 우 | 146.5 mm | 160.9 mm | +14.5 |
| 회전 | 140.7 mm | 161.8 mm | +21.1 |
| **전체** | **141.1 mm** | **157.6 mm** | **+16.4** |

설계 의도(레퍼런스 +15 mm, 스폰 +17.7 mm)와 일치한다. 산포는 그대로다
(σ 11.5 -> 11.4 mm) — 더 크게 서면서 덜 안정해지지는 않았다.

정지 명령만 +7.1 mm 로 유독 적다. 두 버전 모두 **정지 명령에서 가장 낮게
주저앉는다** (v28 132.6 mm 로 걸을 때보다 낮다). `reward_imitation` 이
cmd_norm <= 0.01 에서 게이트로 꺼지고 `cost_stand_still` 만 남기 때문으로
보이나, 미검증.

### 추종 오차 — v28 보다 나쁘다

| | 평균 추종 오차 |
|---|---|
| v28@1500 | 0.0138 / 0.0167 / 0.0170 → **0.0158** |
| v30@1500 | 0.0245 / 0.0157 → 0.0201 |
| v30@2999 | 0.0245 / 0.0222 → 0.0233 |

**최고 성능은 v28 의 `model_1500.pt` 로 그대로다.** v30 은 키를 얻고 추종을
47% 잃은 교환이지 개선이 아니다. 다만 되돌릴 대상은 아니다 — 목적이 달랐다.

v30@1500 두 측정이 0.0245 와 0.0157 로 갈렸다. **한 번 재고 버전을 판정하지
말 것** 이라는 v29 의 교훈이 또 나왔다.

"오래 학습하면 추종이 나빠진다" 가 v30 에서도 재현됐다 (0.0201@1500 ->
0.0233@2999). v26 / v27 / v28 에 이어 네 번째다.

### 곁가지 — 좌우 비대칭의 원인에서 관절이 빠졌다

그래프를 규약(`docs/graph_conventions.md`)에 맞춰 다시 그리다 나온 결과다.
좌우 관절을 비교하려면 **부호 반전만으로는 부족하고 반주기 이동이 필요하다** —
두 다리는 반주기 어긋나 걷기 때문이다. 정렬 없이 빼면 위상차를 비대칭으로
읽는다. 레퍼런스(정의상 좌우 대칭)로 확인:

| | 이동 없음 | 반주기 이동 |
|---|---|---|
| hip_roll | 14.0° | 2.8° |
| 발목 | 9.8° | 2.2° |
| 무릎 | 6.9° | 4.5° |

정렬 후 v28 의 좌우 차이는 1.4~2.8°, **레퍼런스 자체의 잔차 0.4~3.7° 와 같은
수준**이다. 정책은 이미 데이터가 허용하는 만큼 대칭이다. 횡 이탈 비대칭
(좌 98 / 우 84 mm)은 관절 수준에서 오지 않는다.

이 값(0.007~0.065 rad)은 v29 항목에서 `derive_mirror.py` 가 다른 계산으로
얻은 레퍼런스 잔차 0.02~0.05 rad 과 일치한다. **레퍼런스를 대칭화해서 재생성**
하는 실험을 두 경로가 같이 가리킨다 — 다음 후보.

---

## v31 후보 조사 — 레퍼런스의 좌우 비대칭, 세 가설 모두 기각

v30 이후 다음 실험으로 "레퍼런스를 대칭화한다"를 잡았다. v29 항목과 v30 항목이
서로 다른 계산으로 같은 곳을 가리켰기 때문이다. **학습을 걸기 전에 레퍼런스
쪽에서 먼저 쟀고, 세 가설이 차례로 죽었다.** GPU 는 한 번도 쓰지 않았다.

### 측정 방법

거울 명령 쌍 (dx, dy, dth) 와 (dx, -dy, -dth) 를 각각 생성해, 한쪽의 왼다리와
다른 쪽의 오른다리를 **부호 반전 + 반주기 이동** 후 비교한다. 완벽히 대칭이면 0.

반주기 이동은 **보간**해야 한다. 주기가 27 스텝(홀수)이라 `np.roll` 로 13 만
옮기면 0.5 스텝이 어긋나고, 그 오차가 잔차를 26% 부풀린다 (2.06 -> 1.53°).
처음에 정수 이동으로 쟀다가 다시 잡았다.

### 가설 1 — yaw 격자가 비대칭이다 (**실재하는 결함, 그러나 주범 아님**)

랩PC `auto_gait.json` 의 `min/max_sweep_theta = ±0.3`, `granularity = 0.07`.
0.6 이 0.07 로 나누어떨어지지 않아 격자가

    -0.30 -0.23 -0.16 -0.09 **-0.02  +0.05** +0.12 +0.19 +0.26

이 된다. **0 이 격자에 없고, 118개 명령 중 부호 반전 짝을 가진 것이 하나도
없다.** dx (-0.04~0.06/0.02) 와 dy (±0.03/0.02) 는 멀쩡하다 — yaw 만 어긋났다.
좌회전과 우회전의 모방 목표가 서로 다른 위치의 격자에서 보간돼 온 셈이다.

`±0.28` 로 옮기면 간격 0.07 과 명령 9종을 그대로 두고 대칭이 된다
(`gen_reference_remote.sh --yaw-sweep 0.28`). 생성해 보니 거울짝 없는 명령이
118/118 -> 7/119 로 떨어졌다. **그런데 좌우 잔차는 줄지 않았다.**

### 가설 2 — `foot_zmp_target_y = -0.04` (기각)

medium 프리셋의 이 값이 placo 로 그대로 넘어간다. 지지발 프레임 기준 ZMP 목표를
한쪽으로 4 cm 미는 값이고 발 간격이 0.14 m 라, 좌·우 지지발에 같은 부호로
걸리면 대칭이 구조적으로 깨질 만하다. 0 으로 두고 재생성 -> **오히려 나빠졌다.**

### 가설 3 — 다항식 피팅이 대칭을 깬다 (기각)

`fit_poly.py` 가 `RankWarning: Polyfit may be poorly conditioned` 를 쏟아내서
의심했다. 원본 기록(`recordings_*/`, placo 출력)에서 직접 재봤다.

| | 좌우 잔차 |
|---|---|
| placo 원본 모션 (최적 정렬) | **1.47°** |
| 다항식 피팅 후 | 1.53° |

피팅이 더하는 것은 0.06° 뿐이다. 최적 이동량은 13 프레임(50 FPS, 주기 0.54 s
= 27 프레임의 절반)으로 예상과 맞는다.

### 정리

| 레퍼런스 | 좌우 잔차 (보간 정렬) |
|---|---|
| `ref_h175_symyaw` (yaw 격자 대칭) | **1.53°** |
| `ref_h175_symyaw_zmp0` (+ zmp_y=0) | 1.76° |
| placo 원본, 피팅 전 | 1.47° |

**약 1.5° 의 좌우 비대칭은 placo 보행 생성 단계에 내재한다.** 격자도, ZMP 횡
목표도, 다항식 피팅도 아니다.

그리고 이 값은 v28 정책이 보이는 좌우 차이 1.4~2.8° 와 같은 크기다. **정책은
이미 레퍼런스만큼 대칭이다.** 횡 이탈 비대칭(좌 98 / 우 84 mm)을 레퍼런스에서
찾는 줄기는 여기서 닫는다 — 남은 곳은 폐루프 쪽(몸통 자세, 접지 타이밍)이다.

### 남겨 둔 것

yaw 격자 비대칭은 **고쳐야 할 실재하는 결함**이다. 주범이 아니었을 뿐이고,
0 회전 명령이 격자에 없는 것은 그 자체로 이상하다. `--yaw-sweep` 옵션과
`ref_h175_symyaw.pkl` 을 남긴다. 다음에 레퍼런스를 새로 만들 일이 있으면
`--yaw-sweep 0.28` 을 기본으로 쓸 것.

---

## v31 — 고관절 이탈 상한 (**실패, 접촉이 세 배로**)

실기 이식이 목표로 확정되고 접촉 = 파손이 되면서 최우선 과제가 됐다.
v28 을 정확 메시로 재니 접촉 11.5% (v25 의 56.7% 에서 크게 하락 — 아무 대책도
안 넣었는데 고관절 이탈이 같이 줄어서다). 그래서 이탈을 직접 묶었다.

모터 목표를 레퍼런스 ±(yaw 8.4°, roll 20.0°) 로 클램프. 한계는 v28 자신의
p90 에서 잡았고, 접촉률 11.5% 와 겹치니 "꼬리만 자른다"는 계산이었다.

### 결과 — 정반대

| | v28 | v31 |
|---|---|---|
| 접촉률 | 11.5 % | **30.2 %** |
| 간격 중앙값 | 7.5 mm | **1.6 mm** |
| 고관절 이탈 \|평균\| | 7.03° | **9.44°** |
| 추종 오차 @1500 | 0.0158 | 0.0163 |
| 학습 | — | 1760 iter 수렴, 크래시 0, 경고 없음 |

추종은 안 상했는데 **접촉이 세 배가 됐다.** 설정은 정상 적용됐다
(`params/env.yaml` 에 `hip_dev_limit_*` 확인).

### 왜 틀렸나

두 가지를 동시에 잘못 봤다.

**1. 목표를 묶어도 실제 관절각은 안 묶인다.** 클램프는 `_motor_targets` 에
걸리는데 실측 이탈은 오히려 7.03° -> 9.44° 로 늘었다. PD 제어와 접촉력 때문에
실제 위치가 목표를 넘어간다. **묶어야 할 양을 안 묶었다.**

**2. 꼬리만 잘리지 않았다.** 간격 중앙값이 7.5 -> 1.6 mm 다. 상위 10% 만
영향받았다면 중앙값은 그대로여야 한다. 고관절 목표가 잘리자 정책이 그 관절의
제어 권한을 잃고 **자세 전체가 몸통 쪽으로 밀렸다.** torso_vs_hip 이 |r| = 0.404
"부분적으로 얽혀 있음" 으로 경고한 그대로다 — 균형 권한을 뺏은 대가를
넘어짐이 아니라 자세 변형으로 치렀다.

### 진짜 결함은 설계 단계에 있었다

**어느 관절이 어느 방향으로 갈 때 간격이 줄어드는지 모르는 채로 대칭으로
잘랐다.** v25 기록에 `right_hip_roll 접촉 -8.0° / 비접촉 +0.9°` 처럼 **부호가
있는** 단서가 이미 있었는데 크기만 보고 설계했다. 안쪽으로 도는 것만 막으면
될 일을 양방향으로 막았으니 균형에 쓰는 바깥쪽 여유까지 없앤 셈이다.

`leg_trunk_clearance.py --dump` 로 포즈별 (간격, 부호 있는 이탈) 을 남기게
고쳤다. v32 는 그 회귀 결과를 보고 설계한다.

### 최고 성능은 그대로 v28

`model_1500.pt` — 추종 0.0158, 접촉 11.5%. v31 은 되돌린다.

---

## 안전 기준이 5 mm 로 바뀌었다 — 그리고 하드 관절 제한은 못 쓴다

사용자가 실기 기준을 확정했다: **다리가 몸통에 닿으면 액추에이터가 깨진다.
5 mm 이상 떠 있어야 하고, 원칙은 엄격해야 한다.**

### 접촉률은 문제를 3분의 1 로 과소평가하고 있었다

| 기준 | v28 위반율 |
|---|---|
| 접촉 (<= 0 mm) | 13.0 % |
| < 3 mm | 29.3 % |
| **< 5 mm** | **38.0 %** |

여태 11.5% 를 보며 판단해 왔는데, 실기 기준으로는 **38%** 다.
`leg_trunk_clearance.py --safe-mm` 로 기준을 바꿨다.

### 소프트 벌점(v32)으로는 보장이 안 된다

리워드 항은 통계적으로 줄일 뿐 "절대 안 닿는다"를 약속하지 못한다. 파손이
걸린 조건에서는 도구가 잘못됐다.

### 안전 영역은 박스가 아니다 (`scripts/diag/safebox.py`)

하드 관절 제한이면 물리가 강제하므로 보장이 된다 -- v31 이 실패한 이유(목표만
묶임)도 해당되지 않는다. 그래서 **레퍼런스가 쓰는 관절 범위 안이 전부 안전한지**
쟀다. 안전하면 그 박스를 관절 제한으로 걸면 끝이다.

    박스 안 무작위 4000개 · 5 mm 위반 8.0% · 최소 간격 0.0 mm

    가장 위험한 자세:
      left_hip_pitch  범위의 95%   left_ankle     범위의 85%
      left_hip_yaw    범위의  4%   left_knee      범위의 16%

**관절 각각은 전부 레퍼런스 범위 안인데 조합이 충돌한다.** 사용자가 지적한
그대로다 -- "DOF 는 CAD 클램프 각 안에서 자유롭게 움직이고, 특정 동작에서
자기충돌한다". 관절별 상한/하한으로는 표현할 수 없는 제약이다.

### 상류에도 대책이 없다

`Open_Duck_Playground` 의 MJCF 에는 `<contact>` 블록이 아예 없고, 관절 제한은
CAD 서보 한계 그대로다 (hip_yaw ±30°, hip_roll ±25°). 레퍼런스가 실제로 쓰는
범위(hip_yaw -8.2~+6.3°, hip_roll -11.2~+18.2°)의 네 배다. 빌려올 해법이 없다.
다만 **조일 여유는 아주 크다**는 뜻이기도 하다.

### 남은 길: CBF-QP 안전 필터

관절 간 결합을 표현할 수 있는 것은 런타임 안전 필터뿐이다. 장점이 하나 더
있다: **정책을 다시 학습시키지 않는다.** v28 의 추종 성능(0.0158)을 그대로 두고
안전만 얹는다 -- v31/v32 처럼 학습을 흔들지 않는다.

  - Acc-CBF-QP  https://arxiv.org/html/2607.14488v1
  - SPARK       https://arxiv.org/pdf/2605.19009
  - SHIELD      https://arxiv.org/html/2505.11494v3

관건은 h(q) = 간격(q) - 5 mm 를 50 Hz 로 푸는 것이다. 형상쌍이 4872 개라 정확
메시는 불가능하다. 소수의 쌍이 지배적이면 캡슐 근사로 해석적으로 풀 수 있다 --
`scripts/diag/pairs.py` 로 확인 중.

## v32 — 안쪽 방향만, 실측 관절각에 벌점 (**성공**)

v31 의 재설계. 측정으로 방향을 먼저 특정하고(양다리가 안쪽으로 돌 때만 접촉),
한쪽 방향만, 모터 목표가 아니라 실측 관절각에, 임계 3° 로 걸었다.
3000 iter 완주, 크래시 0, 경고 없음.

### 결과 — 접촉이 사라졌다

| | v28 | v31 | **v32** |
|---|---|---|---|
| 5 mm 위반 | 38.0 % | (미측정) | **1.0 %** |
| 접촉 (<=0) | 13.0 % | 30.2 % | **0.0 %** |
| 최소 간격 | 0.0 mm | 0.0 mm | **4.0 mm** |
| 간격 중앙값 | 6.7 mm | 1.6 mm | **13.8 mm** |
| 추종 오차 @1500 | 0.0158 | 0.0163 | 0.0200 |

**중앙값이 6.7 -> 13.8 mm 로 분포 전체가 몸통에서 멀어졌다.** v31 이 7.5 -> 1.6 mm
로 밀렸던 것과 정확히 반대다. 같은 목표를 노린 두 실험이 정반대로 갈린 이유는
하나뿐이다 -- v31 은 양방향을 목표에서 잘랐고, v32 는 안쪽만 실측값에서 눌렀다.

남은 1.0% 는 전부 회전 명령이다 (6.0%, 최소 4.0 mm). 회전은 레퍼런스 자체도
5.6 mm 로 가장 빡빡한 명령이라, 정책만의 문제가 아닐 수 있다.

대가는 추종 0.0158 -> 0.0200 (+27%), 주로 횡방향 (좌 0.034~0.039 / 우
0.032~0.047, v28 은 0.018 / 0.021). hip roll 은 횡이동을 만드는 관절이므로
거기에 벌점을 준 직접적 결과다.

### 그래도 이것으로 끝이 아니다

사용자 원칙은 **엄격**이다. 1.0% 는 0 이 아니고, 리워드 항은 구조적으로 0 을
보장하지 못한다. 다만 v32 는 CBF 필터를 **현실적으로** 만든다: 정책이 이미
거의 안전하므로 필터가 개입할 일이 드물고, 그만큼 추종을 덜 해친다.

### CBF 가 가능해졌다 — 지배적 링크 쌍이 4 개뿐 (`scripts/diag/pairs.py`)

형상쌍 4872 개를 50 Hz 로 다 푸는 것은 불가능하다. 그런데 v28 위반 87 건의
최소 거리 쌍을 세어 보니 **전부 몸통 대 정강이 4 쌍**이었다:

    trunk_assembly <-> knee_and_ankle_assembly_3   36%
    trunk_assembly <-> knee_and_ankle_assembly_2   34%
    trunk_assembly <-> knee_and_ankle_assembly     23%
    trunk_assembly <-> knee_and_ankle_assembly_4    7%

다른 쌍은 0 건이다. **캡슐 4 쌍이면 h(q) 를 해석적·미분가능하게 계산할 수 있다.**
CBF-QP 의 유일한 실무 장애물이 사라졌다.

### 최고 성능 정리

  추종만 보면       v28 model_1500  (0.0158, 5 mm 위반 38%)
  실기 기준이면     v32 model_2999  (0.0200, 5 mm 위반 1.0%, 접촉 0%)

실기 이식이 목표이므로 **기준선을 v32 로 옮긴다.**

---

## CBF 안전 필터 — 정적 검증 통과 (5 mm 위반 38% -> 0%)

리워드로는 보장이 안 된다. v32 가 5 mm 위반을 38% -> 1.0% 로 낮췄지만 1.0% 는
0 이 아니고, 액추에이터 파손이 걸린 조건에서 "거의" 는 답이 아니다.
보장은 런타임 필터에서만 나온다.

### 형상 근사는 두 번 실패했다

| 근사 | 보수성 | 실제 대비 오차 |
|---|---|---|
| 단일 캡슐 | 안전 | 몸통 반지름 **112.8 mm** |
| 구 집합 64+24 | **위반 0건** | gap 평균 **30.8 mm** |

둘 다 안전 방향으로는 맞지만, 실제 여유가 5~14 mm 인 문제에서 30 mm 를 적게
보고하면 모든 자세가 관통으로 판정된다. 반지름 2 mm 까지 줄이려면 몸통에만
구 7000 개가 필요해 50 Hz 에 안 맞는다.

### 차원을 줄이는 쪽이 답이었다 (`scripts/diag/check_barrier_dims.py`)

몸통이 베이스 링크이므로 한쪽 정강이와의 거리는 **그 다리 5 관절만의 함수**다.
측정으로 확인:

    반대쪽 다리 무관성   평균 0.0000 mm · 최대 0.0000 mm
    좌우 미러 대칭       평균 0.2021 mm · 최대 0.2208 mm

10 차원이 5 차원이 되고, 함수 하나로 양쪽 다리를 덮는다. 형상을 덮으려 애쓰는
대신 **함수를 직접 배우면 된다.**

### 장벽함수 (`scripts/diag/fit_barrier.py`)

레퍼런스 박스를 40% 넓힌 영역에서 4 만 자세를 정확 메시로 재고(16.8% 가 5 mm
미만이라 결정 경계가 잘 덮인다) SiLU MLP 를 피팅했다. 과대추정에 3 배 벌점을
줘서 오프셋이 작아지게 했다.

    검증 평균 오차      0.583 mm
    최대 과대추정       1.680 mm  -> 보수 오프셋
    미러 잔차           0.220 mm  -> 추가 오프셋
    총 안전 여유        약 1.9 mm   (구 집합의 30.8 mm 와 비교)

**h(q) = net(q) - 1.90 mm** 로 쓰면 근사가 실제보다 낙관적일 수 없다. 그게
"보장" 이 서는 자리다.

### 필터 (`source/open_duck_mini_isaaclab/safety_filter.py`)

**QP 를 풀지 않는다.** 표준 CBF-QP 는 솔버가 필요하지만 여기서는 다리당 제약이
스칼라 하나뿐이라 각 스텝이 닫힌 형태(뉴턴 투영)로 나온다.

**한 번의 CBF 스텝이 아니라 반복 투영이다.** 표준 CBF 스텝은 위반량의 alpha 배만
줄이므로 결과가 여전히 안전 집합 밖일 수 있다. 보장이라 부르려면 실제로 안에
넣어야 한다 -- 최대 12 회 반복하되, 위반한 환경만 건드린다.

### 검증 — 정확 메시로 (`scripts/diag/verify_filter.py`)

학습된 h 로 판정한 것을 학습된 h 로 확인하면 검증이 아니다. hppfcl 로 쟀다.
가장 나쁜 궤적인 v28(위반 38%)에 걸었다:

| | 원본 v28 | 필터 통과 |
|---|---|---|
| 5 mm 위반 | 38.0 % | **0.0 %** |
| 접촉 | 13.0 % | **0.0 %** |
| 최소 간격 | 0.0 mm | **6.36 mm** |
| 중앙값 | 6.7 mm | 9.2 mm |

최소 6.36 mm 는 기준 5 mm + 오프셋 1.9 mm 와 맞는다 -- 설계대로다.
비용은 작다: 자세의 55% 를 건드렸으나 관절 이동은 평균 0.54°.

### 남은 것

정적 자세 검증이다. **폐루프에서 돌려봐야 한다** -- 필터가 매 스텝 목표를 밀면
정책이 그 다음 관측에서 다르게 반응하므로, 궤적 전체가 달라진다. 추종 열화와
넘어짐을 그때 측정한다.

랩PC 는 이 시점에 응답이 없었다(공용 장비). 로컬 venv 에 pinocchio 4.1 을 깔아
검증을 옮겼으므로, 이 파이프라인은 이제 랩PC 없이 돈다.
