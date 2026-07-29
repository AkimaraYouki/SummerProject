# 어디에 뭐가 있나

2026-07-30 작성. `scripts/`에 34개 파일이 평평하게 쌓여 있어서 뭘 열어야 할지
알 수 없다는 지적을 받고, 폴더를 나누면서 같이 쓴 지도다.

## 먼저 이것만 알면 된다

거의 모든 일은 `odm` 하나로 한다. 나머지는 대부분 "그때 한 번 쓴 것"이다.

    odm train  [ver] [iters] [envs]   학습
    odm watch  [ver]                  진행 한 줄
    odm play   [ver]                  재생 (네이티브 창)
    odm play   [ver] --joystick       Xbox 패드로 실시간 조종
    odm measure[ver]                  6방향·주기성·리워드 성분
    odm tb                            텐서보드 (v26부터 겹침)
    odm test                          테스트 전부
    odm stop / odm list

## 실제 코드는 여기 (source/)

RL 환경과 정책 설정. **고칠 게 있으면 십중팔구 여기다.**

    source/open_duck_mini_isaaclab/
      tasks/velocity/
        joystick_env.py        환경 본체 — 관측·리워드·종료·리셋
        joystick_env_cfg.py    버전별 설정 (v13, v17, v24, v25/v26 Path ...)
        rewards.py             리워드 항 계산식
        observations.py        관측 잡음·지연 버퍼
        events.py              도메인 무작위화
      agents/
        rsl_rl_ppo_cfg.py      PPO·네트워크 설정 (obs_groups 포함)
        rsl_rl_compat.py       rsl-rl 2.x -> 5.0.1 이식 계층
      reference_motion/
        poly_reference_motion.py   레퍼런스 보행을 다항식에서 복원
      joint_order.py           관절 순서·인덱스 — 여기 틀리면 전부 조용히 틀린다
      robot_cfg.py             로봇 자산(USD)·액추에이터 설정
      joystick_input.py        Xbox 패드 (joydev 직접 읽기, 의존성 없음)

## scripts/ — 목적별로 나눠뒀다

**최상위 = 매일 쓰는 것.** `odm`이 부르거나, 손으로 자주 부르는 것들.

    odm                    진입점
    _isaaclab_launch.py    gym.register 를 먼저 돌리는 런처 shim
    play_fixed_cmd.py      재생 (명령 고정/순환/조이스틱, 오버레이, HUD)
    gait_compare.py        6방향 명령 추종      ┐
    joint_periodicity.py   관절 주기성          ├ odm measure 3단계
    imit_internals2.py     모방 리워드 내부     ┘
    train_health.py        학습 중 건강 체크 (텐서보드 이벤트만 읽음)
    joystick_check.py      패드 입력 눈으로 확인 (--raw 로 축 번호)
    leg_fk.py              다리 순기구학 — play_fixed_cmd 가 import 한다
    mujoco_infer.py        sim2real 검증 (ONNX 이식 때 쓸 것, 아직 미착수)

**scripts/diag/ — 진단.** 뭔가 이상할 때만 연다. 대부분 한 번 쓰고 결론이
`docs/training_log.md`에 적힌 것들이다.

    reward_breakdown.py / _v2.py  리워드 항별 기여도 (이제는 텐서보드에도 나옴)
    reward_at_ready.py            READY 자세로 가만히 있을 때 각 항이 주는 값
    imit_internals.py             imit_internals2 의 옛 버전
    contact_diagnostic.py         몸통/머리/발 접촉력 실측
    selfcol_diag.py               자기충돌 켜면 어떤 종료가 터지는지
    eval_policy_stability.py(.sh) 정책 안정성 지표
    check_joint_stability.py(.sh) 액추에이터 게인 점검 (RL 무관)
    ref_stats.py                  레퍼런스 신호 분포
    ref_vs_robot.py               레퍼런스 vs 실제 관절각
    pose_detail.py                몸통 roll/pitch/yaw + 드리프트
    settle_pose.py / viz_home.py  홈 자세 안착·시각화
    calc_home.py / head_level.py  홈 자세·머리 수평 계산
    lock_test.py                  머리 관절 잠금이 실제로 먹는지

**scripts/setup/ — 자산 준비.** 로봇을 처음 들여올 때 한 번.

    convert_urdf.sh / convert_urdf_cd.py   URDF -> USD (cd 는 convex decomposition)
    patch_urdf_for_placo.py                Placo 가 요구하는 프레임 별칭 확인
    generate_reference_motion.sh           레퍼런스 보행 생성 (placo 필요)

**scripts/legacy/ — `odm`이 대체했다.** 참고용으로만 남겼다.

    train.sh  play.sh

## 나머지

    tests/                 Isaac Sim 없이 도는 테스트. `odm test` 로 한 번에.
    docs/
      map.md               (이 파일)
      training_log.md      v1~v26 무엇을 왜 바꿨고 어떻게 됐는지 — 가장 중요한 기록
      decisions.md         포팅 단계에서의 설계 결정
      handoff/             기계를 옮기며 남긴 인수인계 + 세션 메모리
      isaaclab_setup.md, onshape_import.md, webrtc_streaming.md
    robot/                 URDF / USD / 메시
    reference_motion_generator/   레퍼런스 보행 생성기 (upstream 코드)
    logs/rsl_rl/...        학습 런 (체크포인트 + 텐서보드 이벤트)
    outputs/               일회성 산출물

## 알아둘 함정

- **경로에 공백이 있다** (`Extreme SSD`). `$(ls ...)`를 `for`에 쓰면 단어가
  쪼개진다. glob 이나 `while read` 를 쓸 것.
- **Isaac Sim 두 개 동시 실행 금지.** 한쪽이 조용히 죽는다. `odm`이 막는다.
- **ROS Humble 의 PYTHONPATH 가 샌다.** Isaac 파이썬(3.11)에 ROS 의 3.10
  site-packages 가 들어와 엉뚱한 곳에서 죽는다. `odm test` 는 그래서
  `env -u PYTHONPATH` 로 돈다.
- **드라이버 595.84 는 RTX 렌더러를 깬다.** 580.173.02 를 쓴다.
  (`docs/handoff/README.md` 참고)
