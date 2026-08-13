# 잿슨 배포 코드 검증 (2026-08-10, 중력·전류 수정 이후)

`~/rl_walk.py` (538 행) 와 `~/rustypot_hwi.py` (313 행) 를 심 쪽
`joystick_env.py` 와 한 항목씩 대조했다. 비교 기준은 추측이 아니라 **v40 이
실제 학습에 쓴 설정 덤프** (`logs/.../imitation_v40/params/env.yaml`) 다.

결과: **12 항목 일치, 1 항목 불일치.**

---

## 일치 확인된 것

### 1. 관측 벡터 구성 — 순서·차원 완전 일치

| # | 블록 | 차원 | 심 (`_get_observations`) | 잿슨 (`rl_walk.py:446`) |
|---|---|---|---|---|
| 1 | gyro | 3 | `imu.data.ang_vel_b` | `gyro` |
| 2 | accel | 3 | `imu.data.lin_acc_b` (중력 포함) | 생 ACC (0x08) |
| 3 | command | 7 | `_command` | `[vx,vy,wz,0,0,0,0]` |
| 4 | joint_pos_rel | 14 | `joint_pos − default` | `pos − READY_ARR` |
| 5 | joint_vel_scaled | 14 | `joint_vel × 0.05` | `vel × DOF_VEL_SCALE` |
| 6–8 | 액션 이력 ×3 | 42 | `_last_act` 등 | `last_act` 등 |
| 9 | motor_targets | 14 | `_motor_targets` | `motor_targets` |
| 10 | contact | 2 | `_get_foot_contact()` | `feet.get()` |
| 11 | imitation_phase | 2 | `[cos, sin]` | 같음 |
| 12 | path_error | 3 | `_path_error()` | `[0, 1, 0]` 고정 |
| 13 | projected_gravity | 3 | `robot.data.projected_gravity_b` | GRV(0x2E) 정규화 |
| | **합** | **107** | | `assert obs.shape[0] == 107` |

`path_error` 를 `[0,0,0]` 이 아니라 **`[0,1,0]`** 으로 둔 것이 맞다 —
세 칸은 `[lateral, cos(yaw_err), sin(yaw_err)]` 이고 오차 0 이면 cos=1 이다.
틀리기 쉬운 자리인데 제대로 돼 있다.

### 2. READY 자세 — 14축 전부 0.000° 차

`env.yaml` 의 `robot.init_state.joint_pos` 와 잿슨 `READY_JOINT_POS` 를
직접 비교했다. **최대 차 0.00000 rad.** 관측 4번(`joint_pos_rel`)의 기준점이라
여기가 어긋나면 전 축이 조용히 오프셋된다.

### 3. 스칼라 상수

| | 심 (env.yaml) | 잿슨 |
|---|---|---|
| `action_scale` | 0.25 | 0.25 |
| `dof_vel_scale` | 0.05 | 0.05 |
| `max_motor_velocity` | 4.82 | 4.82 |
| 제어 주기 | `dt 0.002 × decimation 10` = 50 Hz | `DT = 1/50` |
| 보행 주기 | 27 스텝 | `GAIT_PERIOD_STEPS = 27` |

### 4. 머리 명령 4칸

심의 `neck_pitch_range` … `head_roll_range` 가 **전부 `(0.0, 0.0)`** 이다.
즉 학습 중에도 이 4칸은 항상 0 이었고, 잿슨이 0 을 보내는 것이 맞다.
(범위가 0 이 아니었다면 관측 분포가 어긋났을 자리다.)

### 5. 머리 잠금

심은 `action_w_delay[:, head_idx] = 0` 로 **액션을** 0 으로 만들고
`target = default + 0×scale = default` 가 된다.
잿슨은 `target[HEAD_IDX] = READY_ARR[HEAD_IDX]` 로 **목표를** 직접 고정한다.
경로는 다르지만 결과가 같다. ✓

### 6. 고관절 클램프·안전필터 — v40 에선 둘 다 꺼져 있다

`env.yaml`: `hip_dev_limit_yaw: null`, `hip_dev_limit_roll: null`,
`safety_filter_ckpt: null`. 따라서 `_hip_lim = None`, `_safety = None` 이고
심에서도 이 두 단계는 건너뛴다. 잿슨이 생략한 것이 맞다.
**다만 이건 v40 한정이다** — 클램프를 켠 설정으로 학습한 정책을 올릴 때는
잿슨에도 같은 클램프를 넣어야 한다.

### 7. 액션 저역통과

v40 은 `action_lowpass_alpha: 0.0`, `..._standstill: 0.0` 이라 심에서 필터
블록 자체가 실행되지 않는다. 잿슨은 `action_filt = α·prev + (1−α)·a` 를
무조건 계산하지만 α=0 이면 항등이라 같다. ✓
명령 크기에 따라 α 를 섞는 smoothstep 공식도 심과 같은 식을 쓴다.

### 8. 속도 제한

양쪽 다 `max_delta = 4.82 × 0.02 = 0.0964 rad/step`, 직전 목표 기준 clip.
심은 고관절 클램프·안전필터 **뒤에** 걸고, 잿슨은 그 둘이 없으니 순서도 동일. ✓

### 9. `motor_targets` 시점 — 일치

심: `_pre_physics_step(a_t)` 에서 갱신 → 그 뒤 `_get_observations` 가 읽는다.
그 관측이 `a_{t+1}` 을 만든다. → **생성하려는 액션 기준 lag 1.**
잿슨: 관측을 만들 때 `motor_targets` 는 직전 루프 값(= `a_{k−1}` 에서 나온 것),
그 관측이 `a_k` 를 만든다. → **lag 1.** ✓

### 10. 모방 위상 시점 — 일치

심은 `_pre_physics_step` 안(390행)에서 증가시키고 `_get_observations`(441행)가
증가된 값을 읽는다. 잿슨은 관측에 현재 값을 쓰고 추론 뒤 증가시킨다.
두 경우 모두 **제어 스텝마다 정확히 1 씩 진행**하며, 정지(`‖cmd‖ ≤ 0.01`)에서
위상을 0 에 묶는 것도 같다. ✓

### 11. 중력 부호 규약 — 일치

심 `projected_gravity_b` 는 직립에서 `(0, 0, −1)`.
잿슨 `−v/‖v‖` 도 직립에서 `(0, 0, −1)` (로그 첫 줄 실측
`(0.0394, −0.0488, −0.9980)`). ✓
그리고 **accelerometer 칸(2번)에는 생 ACC 를 그대로 유지**한 것이 맞다 —
심의 그 칸도 `lin_acc_b` 라 접지 충격이 들어 있는 값이다. 두 칸의 출처가
다르다는 걸 정확히 구분했다.

### 12. 접지

심은 접촉력 > 1.0 N 을 이진화. 잿슨은 발 스위치 이진값. 의미 일치. ✓

---

## 불일치 1건 — 액션 이력이 실기에서 한 스텝 더 최신이다

### 무엇이 다른가

심 `joystick_env.py`:
* 444행 `state = torch.cat([... self._last_act ...])` ← **관측 조립**
* 515–517행 `_last_last_last_act ← _last_last_act; _last_last_act ← _last_act;
  _last_act ← _actions` ← **이력 시프트 (관측 조립 뒤)**

즉 스텝 `t` 끝에 나오는 관측은 `_last_act = a_{t−1}` 을 담고, 그 관측이
`a_{t+1}` 을 만든다. **생성하려는 액션 기준 lag 2.**

잿슨 `rl_walk.py`:
* 446행 관측 조립 (`last_act` = `a_{k−1}`)
* 466행 추론 → `a_k`
* 495–497행 시프트

**lag 1.**

정리하면 세 이력 블록 전부가 한 칸씩 어긋난다:

| | 심 | 잿슨 |
|---|---|---|
| `last_act` | a_{t−1} | a_{k−1} … 이지만 **기준 액션이 한 스텝 다르다** |
| 생성 액션 기준 지연 | **2, 3, 4** | **1, 2, 3** |

`motor_targets`(9번)와 위상(10번)은 양쪽 다 lag 1 로 맞는데, 액션 이력만
어긋난다. 심에서 `_motor_targets` 는 `_pre_physics_step`(관측 **전**)에서,
`_last_act` 는 `_get_observations` 끝(관측 **후**)에서 갱신되기 때문이다.
이 한 스텝 차이가 잿슨에는 재현돼 있지 않다.

### 크기

관측 107 칸 중 **42 칸**(3 × 14)이 영향을 받는다. 실기 로그 실측:

| | 값 |
|---|---|
| 다리 10축 액션 RMS 크기 | 0.953 |
| 인접 스텝 변화 RMS | **0.230** (크기의 24 %) |
| 14축 1차차분² 합 (실기) | 0.6095 → 축당 0.209 |
| 14축 1차차분² 합 (심 v40) | 0.0404 → 축당 0.054 |

즉 잘못된 칸을 넣으면 42 개 입력이 축당 **0.21** 만큼(액션 크기의 약 1/4)
어긋난 값으로 들어간다. 심에서는 축당 0.054 였을 양이라, **실기 액션이 심보다
3.9 배 거칠어서 이 불일치의 영향도 그만큼 커진다.**

치명적이진 않다 — 넘어짐의 주원인은 §3 의 토크-속도 결합이다. 하지만
공짜로 없앨 수 있는 계통 오차다.

### 고치는 법 (한 칸 지연 추가)

`rl_walk.py` 495–497행을 이렇게 바꾼다:

```python
# 심은 관측을 조립한 **뒤에** 이력을 민다(joystick_env.py:444 vs 515).
# 그래서 심의 관측이 담는 last_act 는 "직전 액션"이 아니라 "그 전 액션"이다.
# 여기서도 한 칸을 미뤄야 학습 때와 같은 입력이 된다.
last_last_last_act = last_last_act
last_last_act = last_act
last_act = pending_act        # 한 스텝 묵힌 것
pending_act = action.astype(np.float32)
```

`pending_act = np.zeros(14, dtype=np.float32)` 를 `last_act` 들과 같은 자리
(340행 근처)에서 초기화한다.

**검증법**: 고친 뒤 정지 명령으로 돌려서 관측 42 칸이 한 스텝 밀렸는지
CSV 로 확인한다. 자세 지표가 좋아지는지는 부차적이다 — 이건 성능 개선이
아니라 **train/test 일치** 수정이다.

---

## 잿슨 후속 리포트에 대한 교차검증

`hw_v40_followup_2026-08-10.md` 의 주장 중 내가 독립으로 확인한 것:

| 주장 | 확인 |
|---|---|
| §5.6 내 피치 부호가 뒤집혔다 | **맞다.** 심 기준은 `standstill_pose.py:73` 의 `atan2(+gx, −gz)`. 내 `analyze_hw_log.py` 는 `atan2(−gx, −gz)` 였다. 고쳤고, 실기값은 **+35.01° 앞쏠림**이다 |
| §3 `DCMotor` 가 토크-속도 직선을 구현한다 | **맞다.** `actuator_pd.py` `_clip_effort()` 에 `torque_speed_top = saturation_effort × (1 − joint_vel / velocity_limit)` 가 그대로 있다 |
| §3 현재 `ImplicitActuator` 는 두 한계를 독립으로 건다 | **맞다.** v40 `env.yaml` 에 `effort_limit_sim: 4.1`, `velocity_limit_sim: 4.82` 가 결합 없이 들어 있다 |
| §1.2 accelerometer 칸은 생 ACC 유지가 맞다 | **맞다.** 심의 그 칸은 `imu.data.lin_acc_b` 다 |

§4 의 `left_knee` 간헐 정지(전류 0.003 A, 목표에서 30° 벌어진 채 240 ms)는
로그만으로는 나도 더 못 좁힌다. 물리 점검이 맞다.

---

## 결론

배포 브리지는 **관측 13 블록 중 12 개가 심과 정확히 일치**하고, 상수·자세·
게이트 조건도 전부 맞다. 남은 불일치는 액션 이력 한 칸이며 위 패치로 닫힌다.

다음 학습(v41)에서 `DCMotorCfg` 로 바꾸기 전에 이 패치를 넣어두면, 그 다음
실기 시험에서 관측 불일치를 원인 후보에서 완전히 뺄 수 있다.
