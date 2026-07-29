---
name: openduck-bam-pd-gains
description: "Open Duck Mini XM430 PD gain(stiffness/damping) 재측정 작업 — BAM 실측 재검증 TODO, 현재값과 참고자료"
metadata:
  type: project
  originSessionId: 55b3e7a7-b0a7-4f32-9e42-95ee5ed8ba9f
  modified: 2026-07-26T17:26:24.784Z
---

**결론 (2026-07-27): 사용자가 현재 BAM 파라미터(stiffness=37.65, damping=1.352)를 "쓸만함"으로
판정 — 재측정 없이 그대로 유지하기로 함.** PD 게인은 twitching의 원인 후보에서 제외됨. 아래는
그 판정 전까지의 조사 배경/참고자료 기록 (재검토 필요해지면 참고).

**현재 상태 (2026-07-27 기준, 판정 전 시점)**: `imitation_v2` 학습이 twitching/reward-hacking으로 재실패했고,
레퍼런스 모션 데이터 자체는 이미 철저히 검증된 상태([[project_openduck_autonomous_training]]
참고)라 원인에서 제외됨. 남은 후보 중 하나로 **PD 게인(stiffness/damping)이 실제 XM430
액추에이터 특성과 안 맞아서, underdamped 상태로 목표각 근처에서 진동(twitching)하는 것 아니냐**는
가설이 나옴 — 지금 시뮬레이션 값이 미검증 상태이기 때문.

**지금 쓰는 값** (`source/open_duck_mini_isaaclab/robot_cfg.py`, `scripts/convert_urdf.sh`
양쪽에 동기화돼있어야 함):
```
stiffness = 37.65   # N*m/rad
damping   = 1.352    # N*m*s/rad
armature  = 0.01432
friction  = 0.0761
effort_limit_sim   = 4.1   # N*m (XM430-W350 stall torque @ 12.0V)
velocity_limit_sim = 4.82  # rad/s (no-load speed @ 12.0V, 46rpm)
```

**출처**: `~/Desktop/robot make/bam_xm430_params/m4.json` (BAM으로 이 XM430 개체를 실측
특성화한 결과)의 kt/R/friction_viscous를, BAM 자체 소스코드
(`bam/bam/actuator.py::VoltageControlledActuator.to_mujoco()`)의 변환 공식에 대입:
```
stiffness = (1/128) * kp_register(800) * vin(12.0) * max_pwm(1.0) * kt/R
damping   = friction_viscous + kt**2/R
```
전체 유도 과정과 각 항의 근거(kp_register=800은 Dynamixel Wizard로 실기 확인됨,
vin=12.0은 실측 12.20V와 근접)는 `docs/decisions.md`의 "PD 게인" 절에 전부 있음 — **새 세션
시작하면 먼저 그 파일을 읽을 것**.

**오늘 웹서치로 찾은 교차검증 자료**:
- [IsaacLab Discussion #2627](https://github.com/isaac-sim/IsaacLab/discussions/2627) —
  같은 XM430-W350으로 Robotis OP3(휴머노이드)를 Isaac Lab에서 굴린 사례.
  실측 튜닝값 stiffness=45.0, damping=1.5 — 우리 값(37.65/1.352)과 damping은 거의 일치,
  stiffness만 ~20% 차이. 단 이 스레드 자체가 "P gain 올려도 rise time 안 바뀜" 미해결
  이슈로 끝나서, `IdealPDActuator` 클래스 자체의 한계일 가능성도 언급됨 — 우리가 쓰는
  액추에이터 클래스가 뭔지 확인해볼 가치 있음.
- BAM 원본 저장소([Rhoban/bam](https://github.com/Rhoban/bam))는 MX-64/MX-106/eRob80
  계열만 공식 피팅해뒀고 **XM430은 없음** — 즉 `m4.json`은 팀 자체 커스텀 측정치이지
  검증된 공식값이 아님. 재측정 가치가 있음을 뒷받침.

**TODO — 보류됨 (2026-07-27 사용자 판정으로 불필요)**:
~~1. XM430 실물을 BAM으로 재측정~~
~~2~5. 재계산·robot_cfg.py/convert_urdf.sh 갱신·재학습 검증~~
→ 전부 스킵. twitching 원인 조사는 리워드 가중치 스윕([[project_openduck_autonomous_training]])
쪽에 집중.

**참고**: 지금(2026-07-27) 별도로 `alive_scale`×`w_joint_pos` 리워드 가중치 조합 스윕
(A10J10/A5J15/A20J5/A5J5, [[project_openduck_autonomous_training]])이 이 세션에서 자율로
진행 중 — BAM 재측정은 그 결과가 안 풀릴 경우의 다음 후보로 대기 중인 상태.
