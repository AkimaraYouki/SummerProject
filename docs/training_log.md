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
