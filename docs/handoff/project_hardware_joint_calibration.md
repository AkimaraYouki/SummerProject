# 실기 관절 영점·방향 맞추기 — 새 세션 시작점

> **✅ 2026-08-06 완료됨.** 여기서 세운 목표(영점·부호)는 14축 전부 끝났다.
> 결과와 그 뒤에 이어진 통신·IMU 실측은 **`project_hardware_bringup_2026-08-06.md`**
> 에 있다. 그쪽을 먼저 읽을 것. 이 문서는 절차와 안전 제약의 근거로 남긴다.
>
> 요약: 영점은 잴 것이 없었다(조립 때 2048 = 기구학적 0). 부호는 14축 확정,
> `source/open_duck_mini_isaaclab/hardware_map.py` 에 고정. 아래 §"부호 규약" 이
> 미결로 남긴 **hip_roll 은 MIRROR_SIGN(+1) 쪽이 맞는 것으로 닫혔다.**

작성 2026-08-06. 실제 로봇을 조립한 뒤 **시뮬 관절 좌표계와 다이나믹셀 14축을 일치**시키는
작업의 출발점. 목표는 두 가지뿐이다: **영점(zero)** 과 **부호(direction)**.

---

## 하드웨어

| 항목 | 값 |
|---|---|
| 액추에이터 | XM430-W350 (model 1020) ×14, ID 1~14 |
| 프로토콜 / 보드레이트 | 2.0 / 57600 |
| 인터페이스 | U2D2 (FT232H `0403:6014`) |
| **포트 위치** | **Jetson** `parksuho@192.168.137.7` (`suhojetson.local`) 의 `/dev/ttyUSB0` |
| SDK | Jetson에 dynamixel-sdk 4.0.5 설치됨 |
| 정렬 스크립트 | Jetson `~/home_position.py` — 핑 → 위치제어모드 → Goal 2048 → 도달확인, `--release`로 토크 해제 |

**데스크탑에는 U2D2가 없다.** `/dev/ttyUSB*`를 데스크탑에서 찾으면 헛돈다.
접속은 공개키 등록돼 있어 비밀번호 없이 된다:

```bash
ssh -o BatchMode=yes parksuho@suhojetson.local '<cmd>'
```

> 주의: `192.168.137.8` (계정 `ubuntu`, hostname `jetson`) 은 AK70/CAN 쓰는 **다른 장비**다.

**틱 환산**: 2048 = 중앙. 1 tick = 360/4096 = 0.0879° = **0.001534 rad**.
따라서 0.05 rad ≈ 33 tick.

---

## 관절 순서 — 절대 바꾸지 말 것

`source/open_duck_mini_isaaclab/joint_order.py` 의 `ACTUATOR_JOINT_NAMES`.
**정책 액션 벡터 레이아웃과 동일**하다. 재정렬하면 ONNX 내보내기와 실기 이식이 조용히 깨진다.

| # | 관절 | 하한 (rad) | 상한 (rad) | 하한 (°) | 상한 (°) |
|---:|---|---:|---:|---:|---:|
| 0 | left_hip_yaw | -0.524 | 0.524 | -30 | 30 |
| 1 | left_hip_roll | -0.436 | 0.436 | -25 | 25 |
| 2 | left_hip_pitch | -0.524 | 1.222 | -30 | 70 |
| 3 | left_knee | -2.094 | 2.094 | -120 | 120 |
| 4 | left_ankle | -1.571 | 1.571 | -90 | 90 |
| 5 | neck_pitch | -0.349 | 1.134 | -20 | 65 |
| 6 | head_pitch | -0.785 | 0.785 | -45 | 45 |
| 7 | head_yaw | -2.793 | 2.793 | -160 | 160 |
| 8 | head_roll | -0.524 | 0.524 | -30 | 30 |
| 9 | right_hip_yaw | -0.524 | 0.524 | -30 | 30 |
| 10 | right_hip_roll | -0.436 | 0.436 | -25 | 25 |
| 11 | right_hip_pitch | -0.524 | 1.222 | -30 | 70 |
| 12 | right_knee | -2.094 | 2.094 | -120 | 120 |
| 13 | right_ankle | -1.571 | 1.571 | -90 | 90 |

---

## ⚠️ URDF의 `axis` 는 전부 `0 0 1` — 방향 정보가 아니다

`onshape-to-robot` 은 회전축을 **로컬 Z로 고정**하고 실제 방향을 joint `<origin rpy>` 에 넣는다.
그래서 `axis` 만 보면 14개가 전부 똑같아 보이고, 부호를 못 읽는다. 부모 프레임에서 푼 실제 축:

| # | 관절 | origin rpy (°) | 부모 프레임 회전축 |
|---:|---|---|---|
| 0 | left_hip_yaw | (180, 0, 0) | (0, 0, −1) |
| 1 | left_hip_roll | (−90, 0, −90) | (+1, 0, 0) |
| 2 | left_hip_pitch | (180, −90, 0) | (+1, 0, 0) |
| 3 | left_knee | (0, 0, 0) | (0, 0, +1) |
| 4 | left_ankle | (0, 0, 0) | (0, 0, +1) |
| 5 | neck_pitch | (90, 0, 0) | (0, −1, 0) |
| 6 | head_pitch | (180, 0, 0) | (0, 0, −1) |
| 7 | head_yaw | (90, 0, 0) | (0, −1, 0) |
| 8 | head_roll | (0, −90, 0) | (−1, 0, 0) |
| 9 | right_hip_yaw | (180, 0, 0) | (0, 0, −1) |
| 10 | right_hip_roll | (90, 0, −90) | (−1, 0, 0) |
| 11 | right_hip_pitch | (0, 90, 0) | (+1, 0, 0) |
| 12 | right_knee | (−180, 0, 0) | (0, 0, −1) |
| 13 | right_ankle | (0, 0, 0) | (0, 0, +1) |

주의: 축이 각자의 **부모 링크 프레임** 기준이라 좌우를 바로 비교하면 안 된다.
`left_hip_yaw` / `right_hip_yaw` 만 부모가 `trunk_assembly` 로 같아서 직접 비교가 되는데,
**둘 다 (0,0,−1)** 이다.

---

## 부호 규약 — 데이터로 확정된 것

`READY_JOINT_POS_H175` 는 양다리가 물리적으로 대칭인 기립 자세다. 그래서 좌우 값의
부호 관계가 곧 미러 규약이다:

| 관절 | left | right | 관계 |
|---|---:|---:|---|
| hip_pitch | +0.9910 | +1.0114 | **같은 부호** |
| knee | −1.7852 | +1.8163 | **반대** |
| ankle | +0.8647 | −0.8754 | **반대** |
| hip_yaw | +0.0003 | −0.0005 | 반대 (값이 작아 참고만) |
| hip_roll | +0.0213 | −0.0092 | 부호는 반대인데 **크기가 2.3배 달라 미러쌍이 아님** |

`safety_filter.py` 의 `MIRROR_SIGN` (symmetry.py 에서 유도·검증, 미러 잔차 0.22 mm):

```
(hip_yaw, hip_roll, hip_pitch, knee, ankle) = (−1, +1, +1, −1, −1)
```

→ hip_pitch / knee / ankle 은 READY 와 **일치**한다.
→ **hip_roll 만 둘이 어긋난다** (MIRROR_SIGN 은 같은 부호, READY 는 반대 부호).
READY 쪽 값이 워낙 작고 크기도 안 맞아서 결정적이지 않다. **실기에서 반드시 직접 확인할 것.**

---

## 영점 검증용 기준 자세

`READY_JOINT_POS_H175` (base height **0.1360 m**, 스폰 높이 0.1437 m):

```
left_hip_yaw   0.0003   right_hip_yaw   -0.0005
left_hip_roll  0.0213   right_hip_roll  -0.0092
left_hip_pitch 0.9910   right_hip_pitch  1.0114
left_knee     -1.7852   right_knee       1.8163
left_ankle     0.8647   right_ankle     -0.8754
neck_pitch 0  head_pitch 0  head_yaw 0  head_roll 0
```

> 머리 4축이 0인 건 의도적이다. Z자 자세(neck/head_pitch = 0.785)로 바꾸면 무게중심이
> x로 14.3 mm 이동해서(−8.8 → −23.1 mm) 발 길이 60 mm 로봇이 앞으로 넘어진다.
> **자세만 바꾸면 안 되고 그 자세로 재학습해야 한다** (v33이 그 시도, iter 663에서 중단됨).

---

## 절차 제안

1. **ID ↔ 관절 이름 매핑 확정.** 지금 레포에 이 매핑이 **없다** — 이게 첫 산출물이다.
   ID 하나씩 토크 걸고 살짝 움직여서 어느 관절인지 눈으로 확인.
2. **영점 오프셋 측정.** `home_position.py --release` 로 토크 끄고, 각 관절을 URDF 0 자세
   (기구학적 영점) 에 손으로 맞춘 뒤 present position 읽기. 그 값이 각 ID의 오프셋 tick.
3. **부호 확정.** 관절 하나씩 +0.05 rad (≈ +33 tick) 만 이동시키고, 실제 움직인 방향을
   시뮬(`odm play`) 에서 같은 관절에 +0.05 준 것과 대조. 다르면 부호 −1.
4. **결과 고정.** `source/open_duck_mini_isaaclab/hardware_map.py` 같은 단일 파일에
   `(관절이름 → ID, 오프셋 tick, 부호)` 를 박아두고, 이후 모든 실기 코드가 그것만 임포트하게.
   `joint_order.py` 가 시뮬에서 한 역할과 같은 것을 실기에서 하는 파일.

---

## ⚠️ 안전 — 이 로봇에서 제일 중요한 제약

**자가충돌이 액추에이터를 부순다.** 사용자 지정 기준 **5 mm 클리어런스**.

- 특히 **고관절을 안쪽(inward)으로 돌리는 방향**이 위험하다. v31에서 이걸 대칭으로
  잘랐다가 접촉률이 3배(11.5% → 30.2%) 로 악화된 전례가 있다 — 방향을 모르고 자르면 안 된다.
- 조그는 **URDF 한계 안에서, 한 번에 한 관절만.**
- 무릎 ±2.094 rad (±120°) 가 가동범위가 제일 크다. **무부하(로봇 들어올린 상태)에서만** 크게 움직일 것.
- 시뮬 쪽 안전장치: `safety_filter.py` 의 `ClearanceFilter` (CBF 런타임 필터). 실기 검증
  기준선은 v32+필터에서 **5 mm 위반 0.0% / 접촉 0.0% / 최소 간격 7.5 mm / 계산 0.881 ms**.

---

## 참고 파일

| 파일 | 내용 |
|---|---|
| `source/open_duck_mini_isaaclab/joint_order.py` | 14축 순서 (단일 진실원천) |
| `source/open_duck_mini_isaaclab/safety_filter.py` | CBF 필터, `MIRROR_SIGN` |
| `source/open_duck_mini_isaaclab/tasks/velocity/joystick_env_cfg.py` | `READY_JOINT_POS_H175`, 높이 상수 |
| `robot/robot.urdf` | 관절 한계·origin. 총 질량 2.3388 kg / 링크 20개 |
| `docs/onshape_import.md` | CAD 재임포트 절차 (질량은 밀도로 넣을 것 — 오버라이드는 관성이 안 따라옴) |
| `scripts/diag/leg_trunk_clearance.py` | 정확 메시 클리어런스 측정 |
| `demo.sh` | `drive` / `measure` / `safety` / `graphs` 등 실행 진입점 |
