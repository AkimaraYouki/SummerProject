"""Torch port of Open_Duck_Playground's playground/common/rewards.py
(the subset actually active in joystick.py's reward dict) plus
playground/open_duck_mini_v2/custom_rewards.py::reward_imitation.

All functions are batched: every tensor argument is [N, ...] and every
function returns an [N] tensor. Every formula/constant below is copied
verbatim from the JAX source — see docs/decisions.md for the one place
(reward_imitation's leg-joint index alignment) that needed real derivation
rather than direct transcription.
"""

from __future__ import annotations

import torch

from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX, REF_LEG_JOINT_IDX


def reward_tracking_lin_vel(commands: torch.Tensor, local_lin_vel: torch.Tensor, tracking_sigma: float) -> torch.Tensor:
    """commands: [N,>=2], local_lin_vel: [N,>=2] (base-frame x,y linear velocity)."""
    y_tol = 0.1
    error_x = (commands[:, 0] - local_lin_vel[:, 0]) ** 2
    error_y = torch.clamp(torch.abs(local_lin_vel[:, 1] - commands[:, 1]) - y_tol, min=0.0)
    lin_vel_error = error_x + error_y**2
    return torch.exp(-lin_vel_error / tracking_sigma)


def reward_tracking_ang_vel(commands: torch.Tensor, ang_vel: torch.Tensor, tracking_sigma: float) -> torch.Tensor:
    """commands: [N,>=3], ang_vel: [N,>=3] (base-frame gyro, z = yaw rate)."""
    ang_vel_error = (commands[:, 2] - ang_vel[:, 2]) ** 2
    return torch.exp(-ang_vel_error / tracking_sigma)


def cost_torques(torques: torch.Tensor) -> torch.Tensor:
    return torch.sum(torques**2, dim=-1)


def cost_action_rate(act: torch.Tensor, last_act: torch.Tensor) -> torch.Tensor:
    return torch.sum((act - last_act) ** 2, dim=-1)


def cost_foot_lift(feet_pos_w: torch.Tensor, offset: torch.Tensor, clearance: float) -> torch.Tensor:
    """스윙 발이 **접지 발보다 얼마나 높이 뜨는지** 벌한다. `clearance` 까지는 공짜.

    ## 왜 이 항이 필요한가

    2026-08-13~14 에 레퍼런스로 발 들림을 낮추려 두 번 시도해 두 번 다 실패했다.

        레퍼런스 발 들림   44.8 -> 23.2 mm  (v51 에서 반토막)
        실제 걸음 발 들림  22.5 -> 21.4 mm  (거의 안 변함)

    `reward_imitation` 이 비교하는 것은 **관절각**이지 발 위치가 아니다. 관절각을
    대충 맞추면서도 발은 다른 데로 갈 수 있고, 실제 발 궤적은 균형 요구가
    지배한다. 그래서 발 위치를 **직접** 벌해야 한다.

    ## 접지 발 기준인 이유

    월드 z 를 그대로 쓰면 지형 높이와 발바닥 두께가 섞인다. 두 발 중 낮은 쪽을
    지면으로 보면 그 차이가 곧 **발 여유(clearance)** 이고, 경사·요철에서도
    그대로 뜻이 통한다.

    ## clearance

    이 아래는 벌하지 않는다. 0 으로 두면 딛고 있는 발까지 밀어붙여 발을 끌게
    된다. 다리 마디가 157 mm 이고 실측 발 들림이 22 mm(14 %) 였으므로,
    사람 보행(1~2 %)과 그 사이 어딘가가 목표다.

    ## ⚠️ offset 을 빼야 한다 — 안 빼면 상수를 벌한다

    `body_pos_w` 는 발 **링크 원점**이지 발바닥이 아니다. 이 로봇은 좌우 발
    링크의 z 오프셋이 서로 반대라서(FOOT_FIXED 가 좌 -19.55 / 우 +19.55 mm),
    **양발이 나란히 땅에 있어도 두 원점의 z 가 39 mm 차이난다.**

    2026-08-14 에 이걸 빠뜨리고 학습해서 v53 을 한 번 날렸다. 가만히 서 있는
    정지 명령에서도 이 항이 -0.2415 (리워드 예산의 45 %) 를 먹었고, 정책은
    줄일 방법이 없으니 걸음을 망가뜨리는 쪽으로 갔다 — 보상 318.8 -> 100.4,
    에피소드 561 -> 320. **서 있는데 벌점이 0 이 아니면 그 항은 틀린 것이다.**

    Args:
        feet_pos_w: (N, 2, 3) 월드 좌표 발 링크 위치.
        offset: (N, 2) 또는 (1, 2). 양발이 땅에 있을 때의 각 발 z 오프셋.
            환경이 첫 리셋에서 한 번 재서 넘긴다.
        clearance: 이만큼(m)까지는 공짜.
    """
    z = feet_pos_w[..., 2] - offset
    lift = z - z.min(dim=1, keepdim=True).values
    over = torch.clamp(lift - clearance, min=0.0)
    return torch.sum(over**2, dim=-1)


def cost_torso_ang_vel(ang_vel_b: torch.Tensor) -> torch.Tensor:
    """몸통의 **roll·pitch 각속도**를 벌한다 — 토르소를 부드럽게.

    2026-08-14, 사용자: "보행시 토르소를 최대한 부드럽게 제어할 수 있게".

    지금까지 몸통 흔들림은 `reward_imitation` 안의 `ang_vel_xy` 항으로만
    간접 다뤘다. 그건 **레퍼런스 각속도를 따라가라**는 추종 항이라, 레퍼런스
    자체가 흔들리면 흔들림을 그대로 배운다. 실측 roll 진폭이 심 21.0° ·
    실기 20.4° 로 같았던 것이 그 증거다 — 배포 문제가 아니라 학습된 걸음이다.

    이 항은 레퍼런스와 무관하게 **각속도 자체를 0 으로 끌어당긴다.** yaw(z)는
    빼는데, 제자리 회전 명령이 yaw 각속도를 요구하기 때문이다. roll·pitch 는
    어떤 명령에서도 커야 할 이유가 없다.

    Args:
        ang_vel_b: (N, 3) 몸통 기준 각속도 (자이로).
    """
    return torch.sum(ang_vel_b[:, :2] ** 2, dim=-1)


def cost_joint_accel(joint_acc: torch.Tensor) -> torch.Tensor:
    """관절 **가속도**를 벌한다 — 다리 움직임을 부드럽게.

    `action_rate`/`action_jerk` 는 **정책 출력**의 변화를 본다. 그건 명령의
    매끄러움이지 실제 다리의 매끄러움이 아니다 — PD 게인과 부하가 사이에
    끼어 있어서, 부드러운 명령도 다리에서는 덜컹일 수 있다.

    이 항은 관절이 실제로 얼마나 급하게 속도를 바꾸는지를 직접 본다. 급가속은
    곧 큰 토크이고, 실기에서 그것이 착지 피크 2.4 g 와 전압 강하로 나타났다.

    Args:
        joint_acc: (N, J) 다리 관절 가속도 (rad/s^2).
    """
    return torch.sum(joint_acc ** 2, dim=-1)


def cost_foot_impact(feet_pos_w: torch.Tensor, feet_vel_b: torch.Tensor,
                     offset: torch.Tensor, band: float) -> torch.Tensor:
    """지면 근처에서 **빠르게 내려오는 것**을 벌한다 — 부드러운 착지.

    2026-08-14. 사용자가 계속 말한 "퉁퉁거린다" 의 정체를 드디어 잡았다.

    접지 전환만 세면 v53 이 주기 195 ms, v55 가 210 ms 로 나와 **레퍼런스
    540 ms 의 4 분의 1** 이었다. 그래서 "발을 7 Hz 로 떨고 있다" 고 판단했는데
    **틀렸다.** 접지 신호를 디바운스해서 다시 재니:

        디바운스     없음   20 ms   40 ms   60 ms
        v48           480     540     540     540
        v53           195     385     490     540
        v55@2k        210     480     500     540
        v55@3k        340     430     490     540

    **전부 540 ms 로 수렴한다.** 걸음 주기는 처음부터 옳았고, 다른 것은 한
    스텝 안에서 발이 **착지하며 튀는 정도**였다. 걸음이 가짜였던 게 아니라
    착지가 거칠었던 것이다.

    ## 왜 상태를 안 쓰나

    "착지 순간" 을 잡으려면 이전 스텝의 접지를 들고 있어야 한다. 대신 지면
    **근처** 에서 하강 속도를 벌하면 같은 것을 상태 없이, 그리고 매끄럽게
    가르칠 수 있다 — 착지 직전부터 기울기가 생기므로 정책이 미리 감속한다.
    접지 순간에만 벌하면 이미 늦어서 고칠 방법이 없다.

    양발이 땅에 있으면 lift 가 둘 다 0 이라 가중치는 1 이지만 하강 속도가
    0 이므로 비용도 0 이다 — 정지에서 상시 세금이 붙지 않는다.

    Args:
        feet_pos_w: (N, 2, 3) 월드 좌표 발 링크 위치.
        feet_vel_b: (N, 2, 3) 몸통 기준 발 속도.
        offset: (1, 2) 좌우 발 링크의 z 오프셋. `cost_foot_lift` 주석 참고.
        band: 이 높이(m) 아래에서 선형으로 가중치가 1 까지 오른다.
    """
    z = feet_pos_w[..., 2] - offset
    lift = z - z.min(dim=1, keepdim=True).values
    near = torch.clamp(1.0 - lift / band, min=0.0)
    v_down = torch.clamp(-feet_vel_b[..., 2], min=0.0)
    return torch.sum(v_down ** 2 * near, dim=-1)


def cost_foot_slip(feet_vel_b: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """**땅에 닿아 있는 동안** 발이 수평으로 움직이면 벌한다.

    2026-08-14, 사용자: "발이 좀 미끄러지는데". 발에 3 mm 고무패드를 붙여
    이중지지 요동은 0.672 -> 0.167 rad/s 로 잡혔지만 아직 남아 있다.

    미끄러짐은 마찰만의 문제가 아니다. 발이 **수평 속도를 가진 채로 착지**
    하면 마찰계수가 얼마든 그 속도만큼 쓸린다. 그래서 접지 마스크를 곱해
    "딛고 있는 발은 정지해 있어야 한다" 를 직접 가르친다. 다리 보행에서
    가장 표준적인 미끄러짐 항이다.

    `cost_foot_lateral` 과 다르다: 저쪽은 **스윙 중** 좌우 흔들림을 줄여
    몸통 roll 을 잡고, 이쪽은 **접지 중** 앞뒤·좌우 쓸림을 없앤다.
    `cost_foot_clearance` 와 합치면 "수직으로 들어서 수직으로 내리고, 딛는
    동안은 붙잡는다" 가 된다 — 사용자가 말한 "직선으로 움직이게".

    Args:
        feet_vel_b: (N, 2, 3) 몸통 기준 발 속도.
        contact: (N, 2) 접지 여부 0/1.
    """
    vxy2 = torch.sum(feet_vel_b[..., :2] ** 2, dim=-1)
    return torch.sum(vxy2 * (contact > 0.5).float(), dim=-1)


def cost_foot_clearance(feet_pos_w: torch.Tensor, feet_vel_b: torch.Tensor,
                        offset: torch.Tensor, target: float) -> torch.Tensor:
    """스윙 중 발을 **목표 높이로 유지**한다. 낮아도 높아도 벌한다.

    ## `cost_foot_lift` 와 무엇이 다른가

    `cost_foot_lift` 는 "너무 높으면 벌" 이라 한 방향만 본다. 2026-08-14 에
    발에 3 mm 고무패드를 붙이고 나서 **발이 바닥을 쓸기 시작했다.** 그때
    필요한 것은 낮추는 쪽이 아니라 **목표 높이로 붙잡는 것**이다.

    실측: 스윙 여유가 중앙값 5.8 mm · p90 10.6 mm 인데, 몸통이 roll ±10° 로
    흔들리면 반스탠스 폭 70 mm 에서 발 모서리가 **12 mm** 내려간다. 즉 roll
    만으로도 스윙 발이 바닥에 닿는다. 패드가 미끄러지지 않게 되자 그것이
    곧바로 끌림으로 나타났다.

    ## 수평 속도로 가중하는 이유

    `(높이 - 목표)^2` 만 쓰면 발을 떼는 순간과 딛는 순간까지 벌한다 — 그때는
    발이 지면에 있는 것이 맞는데도. 수평 속도를 곱하면 **발이 실제로 스윙
    하는 동안에만** 걸린다. 이·착지 순간은 수평 속도가 작아 자연히 빠진다.
    다리 보행에서 흔히 쓰는 형태다.

    Args:
        feet_pos_w: (N, 2, 3) 월드 좌표 발 링크 위치.
        feet_vel_b: (N, 2, 3) 몸통 기준 발 속도.
        offset: (1, 2) 양발이 땅에 있을 때의 z 오프셋. `cost_foot_lift` 주석 참고.
        target: 목표 여유 (m). 실측 roll 이 발을 12 mm 내리므로 그보다 커야 한다.
    """
    z = feet_pos_w[..., 2] - offset
    lift = z - z.min(dim=1, keepdim=True).values
    vxy = torch.norm(feet_vel_b[..., :2], dim=-1)
    return torch.sum((lift - target) ** 2 * vxy, dim=-1)


def cost_foot_lateral(feet_vel_b: torch.Tensor,
                      cmd_vy: torch.Tensor | None = None,
                      gate_max: float = 0.0,
                      gate_floor: float = 0.0) -> torch.Tensor:
    """발의 **좌우(몸통 y) 속도**를 벌한다 — "발을 최소한만 움직여라".

    2026-08-14 실측: 한 번 스윙에 발이 위로 23 mm 뜨는 동안 **옆으로 30~41 mm**
    움직인다. 수직성(상승/좌우)이 0.47~0.78 로 1 을 못 넘는다. 사용자가 다른
    빌더의 로봇을 보고 "거의 수직으로 든다" 고 한 것과 정반대다.

    발을 옆으로 던지려면 몸통이 무게를 그만큼 옮겨야 하므로, 이것이 roll 이
    ±10° 로 흔들리는 직접 원인이다.

    전후(x) 속도는 벌하지 않는다 — 보폭은 명령 속도가 정하는 값이라 줄이면
    명령을 못 따라간다 (76 mm × 1/0.54 s = 0.14 m/s ≈ 명령 0.15).

    Args:
        feet_vel_b: (N, 2, 3) **몸통 기준** 발 속도. 월드로 주면 로봇이 도는
            동안 전후 성분이 y 로 섞여 들어온다.
    
    ## 명령 상대형 (2026-08-14 추가)

    `cmd_vy` 를 주면 **명령한 좌우 속도를 뺀 나머지**만 벌한다. 원래 식은
    명령과 무관하게 발의 좌우 속도를 벌해서, 옆으로 걸으라는 명령 자체를
    막았다 — v55 의 옆걸음이 명령의 56~64 % 로 무너진 원인이다.

    `cmd_vy = None` 이면 예전 식 그대로다 (v53·v55 재현용).

    ## 게이트 (2026-08-15 추가)

    v61 에서 명령 상대형만으로는 부족한 것이 드러났다. 앞뒤(0.0155)와
    회전(0.0104)은 v59 그대로 지켰고 roll RMS 도 6.66 -> 4.97 로 합격선을
    통과했는데, **옆걸음만 0.0263 -> 0.0534 로 두 배**가 되어 종합 점수를
    떨어뜨렸다.

    이유는 기하다. 옆으로 걸으려면 **스윙 발이 몸통보다 더 빨리** 옆으로
    가서 다음 디딤 자리를 잡아야 한다. 그러니 명령 속도를 빼도 남는 성분이
    크고, 그것을 벌하면 옆걸음 자체가 느려진다.

    `gate_max` 를 주면 |cmd_vy| 에 비례해 항을 **낮춘다** — |cmd_vy| = 0 에서
    1 배, `gate_max` 에서 `gate_floor` 배. 옆 명령이 없는 구간(전진·후진·
    회전·정지)에서는 억제가 그대로 남아 roll 이득을 지킨다.
    """
    v_y = feet_vel_b[..., 1]
    gate = None
    if gate_max > 0.0 and cmd_vy is not None:
        # 옆 명령이 셀수록 항을 낮춘다. |cmd_vy| = gate_max 에서 gate_floor.
        g = 1.0 - (1.0 - gate_floor) * torch.clamp(cmd_vy.abs() / gate_max, max=1.0)
        gate = g.unsqueeze(-1)
    if cmd_vy is not None:
        # 명령한 좌우 속도만큼은 공짜다. 옆으로 걸으라고 해 놓고 발이 옆으로
        # 움직이는 것을 벌하면 그 명령을 수행할 방법이 없다 — 2026-08-14 에
        # v55 의 옆걸음이 명령의 60 % 로 무너진 것이 정확히 그 때문이었다.
        # cmd_vy = 0 이면 예전 식과 완전히 같아서, 전진·정지에서 얻었던
        # 안정성은 그대로 남는다.
        v_y = v_y - cmd_vy.unsqueeze(-1)
    c = v_y ** 2
    if gate is not None:
        c = c * gate
    return torch.sum(c, dim=-1)


def cost_action_jerk(act: torch.Tensor, last_act: torch.Tensor, last2_act: torch.Tensor) -> torch.Tensor:
    """액션의 **2차차분**을 벌한다 — 진동 그 자체를 겨냥한 항.

    `cost_action_rate`(1차차분)로는 진동을 못 잡는다. 1차차분은 "빠른 움직임" 을
    벌하는 것이라 정상 보행도 똑같이 벌받고, 진동만 골라 누를 수가 없다.
    2차차분은 방향이 뒤집힐 때만 커진다 — 같은 크기의 두 신호를 비교하면:

        매끄러운 램프        1차차분^2 0.0001   2차차분^2 0.0000
        매 스텝 부호 반전    1차차분^2 1.0000   2차차분^2 4.0000

    **매끄러운 궤적은 2차차분이 정확히 0 이다.** 그래서 보행을 깎지 않으면서
    진동만 벌할 수 있다.

    실기 로그(정지 10 초)에서 실측한 값: 1차차분^2 0.148 / 2차차분^2 0.441.
    진동 성분이 지배적이고, 이것이 액션 방향 반전 68.3% 의 정체다.

    저역통과 필터 대신 이것을 쓰는 이유:
      * 필터는 결과를 흡수해 정책을 게으르게 만든다. 실제로 필터로 학습한 v36 의
        원시 액션 요동이 0.0638 로 무필터 v35(0.0537)보다 **컸다**.
      * 필터는 위상 지연(20~47 ms)을 낳아 추종을 깎고 외란 극복의 여유를 먹는다.
        이 항은 지연이 0 이다.
      * 배포 때 필터를 실기에 복제할 필요가 없다 (정책-필터 결합이 사라진다).
    """
    return torch.sum((act - 2.0 * last_act + last2_act) ** 2, dim=-1)


def reward_alive(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.ones(num_envs, device=device)


def cost_stand_still(
    commands: torch.Tensor,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    default_pose: torch.Tensor,
) -> torch.Tensor:
    """ignore_head=False path only (the one joystick.py actually calls).

    `default_pose` 는 호출부가 정한다. 보통은 env 의 default_joint_pos 지만,
    cfg.standstill_joint_pos 가 있으면 그쪽이 들어온다 — 액션의 원점(보행 평균
    자세)과 정지 목표 자세는 목적이 달라서 분리했다 (joystick_env_cfg.py 주석).
    """
    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)
    pose_cost = torch.sum(torch.abs(qpos - default_pose), dim=-1)
    vel_cost = torch.sum(torch.abs(qvel), dim=-1)
    return (pose_cost + vel_cost) * (cmd_norm < 0.01).float()


def cost_upright_standstill(
    commands: torch.Tensor,
    projected_gravity_b: torch.Tensor,
) -> torch.Tensor:
    """정지 명령일 때 **몸통이 수직에서 벗어난 만큼** 벌한다.

    왜 관절각이 아니라 중력벡터인가. v34u 는 "정지 목표 관절각" 을 몸통이 수직이
    되도록 FK 로 풀어서 넣었는데, 실측 몸통 피치가 오히려 +8.19 -> +20.93 도로
    나빠졌다. 관절각은 몸통 기울기의 **간접** 손잡이다 — 발바닥이 지면에 눕도록
    물리가 몸통을 돌려버리면 목표 관절각을 맞춰도 몸통은 안 선다. 게다가
    `cost_stand_still` 계수(-0.2)는 imitation 의 1/30 이라 자세를 강제할 힘도 없다.

    `projected_gravity_b` 는 직립일 때 정확히 (0,0,-1) 이므로 x,y 성분이 그대로
    기울기다 (g_x = sin(pitch), g_y = -sin(roll)cos(pitch)). 이걸 직접 벌하면
    어느 관절로 세우든 정책이 알아서 고른다.

    제곱을 쓰는 이유: 수직 근처에서 기울기가 0 으로 죽어 마지막 1 도를 두고
    다른 항과 싸우지 않는다. 8.19 도에서 0.0203, 20.93 도에서 0.1276 이다.
    """
    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)
    tilt = projected_gravity_b[:, 0] ** 2 + projected_gravity_b[:, 1] ** 2
    return tilt * (cmd_norm < 0.01).float()


def cost_leg_symmetry(
    commands: torch.Tensor,
    qpos: torch.Tensor,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    mirror_sign: torch.Tensor,
) -> torch.Tensor:
    """정지 명령일 때 **좌우 다리 자세가 거울이 아닌 만큼** 벌한다.

    왜 정지에서만인가. 실측해 보면 **보행은 이미 거의 완벽히 대칭**이고
    (모든 관절 1.03 도 이내) **정지만 크게 어긋난다** — 무릎 15.2 도,
    hip_roll 12.3 도. env 별 표준편차가 2.0~2.3 도로 좁아서 무작위가 아니라
    32 개 환경이 전부 같은 쪽으로 치우친 일관된 편향이다. 좌 무릎이 보행 때보다
    17 도 더 굽으므로 한쪽 다리에 기대어 서는 자세다.

    이유는 구조적이다. 정지에서는 imitation 이 꺼지고(`cmd_norm > 0.01` 게이트)
    위상도 0 에 묶이며, 남는 것은 `cost_stand_still`(-0.2) 뿐인데 그 목표 자세
    (레퍼런스 평균)조차 2 도 비대칭이다. **좌우 대칭을 요구하는 항이 하나도
    없어서** 정책이 편한 짝다리를 찾은 것이다. 보행이 멀쩡한 건 imitation 이
    켜져서 레퍼런스가 양쪽을 다 붙잡아 주기 때문이다.

    보행에도 걸지 않는 이유: 보행은 원래 좌우가 **반주기 어긋난** 운동이라
    같은 시각의 좌우 관절각이 거울이면 안 된다. 정지에서만 의미가 있다.

    거울 부호는 joint_order.LEG_MIRROR_PAIRS — URDF 에서 FK 로 확정한 것이고,
    그 규칙에서 로봇 모델 자체는 대칭이다(발 위치 오차 0.5 mm, 질량 동일).

    제곱을 쓰는 이유는 `cost_upright_standstill` 과 같다 — 대칭 근처에서
    기울기가 죽어 마지막 1 도를 두고 다른 항과 싸우지 않는다.
    """
    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)
    err = qpos[:, right_idx] - mirror_sign * qpos[:, left_idx]
    return torch.sum(err * err, dim=-1) * (cmd_norm < 0.01).float()


def reward_imitation(
    base_lin_vel_w: torch.Tensor,  # [N,3] — see docs/decisions.md frame note below
    base_ang_vel_w: torch.Tensor,  # [N,3]
    joints_qpos: torch.Tensor,  # [N,14] actuator order (joint_order.ACTUATOR_JOINT_NAMES)
    joints_qvel: torch.Tensor,  # [N,14]
    contacts: torch.Tensor,  # [N,2] bool/float, left then right
    reference_frame: torch.Tensor,  # [N,36], see poly_reference_motion.py docstring
    commands: torch.Tensor,  # [N,7]
    w_joint_pos: float = 15.0,
    bounded_joint_pos: bool = False,
    swing_only_contact: bool = False,
    k_lin_vel_xy: float = 8.0,
    w_lin_vel_z: float = 1.0,
    w_ang_vel_xy: float = 0.5,
    w_contact: float = 1.0,
    w_stance_violation: float = 0.0,
    w_joint_pos_amp: float = 1.0,
) -> torch.Tensor:
    """Direct port of custom_rewards.py::reward_imitation.

    Frame note: the reference frame's linear/angular velocity slices were
    recorded in the WORLD frame by the reference-motion generator
    (gait_generator.py's `world_linear_vel`/`world_angular_vel`). We compare
    against Isaac Lab's `root_lin_vel_w`/`root_ang_vel_w` (also world frame)
    for that reason. Playground's own MJX version instead reads MuJoCo's
    raw floating-base qvel, whose angular component is expressed in the
    body's local frame per MuJoCo's freejoint convention — a likely
    frame mismatch already present in the original code, not something
    reproduced here. If reward curves look off during the Stage 3 Ubuntu
    smoke test, this is the first place to check.
    """
    # 2026-07-28 (imitation_v12): w_lin_vel_z / w_ang_vel_xy / w_contact and
    # the lin_vel_xy exp sharpness became arguments. scripts/imit_internals2.py
    # measured imitation_v11's final policy standing still (base speed
    # 0.064 m/s against the reference's 0.265) and still collecting:
    #   lin_vel_z  0.954/1.0   ang_vel_xy 0.220/0.5   lin_vel_xy 0.556/1.0
    # i.e. ~92% of the raw imitation total was reachable without walking.
    # lin_vel_xy is the term that is *supposed* to price walking, and at the
    # default sharpness k=8 being wrong by the entire reference speed
    # (err^2 = 0.265^2 = 0.070) still pays exp(-0.56) = 0.57. The term simply
    # could not tell standing from walking. Defaults below are unchanged so
    # v1-v11 stay reproducible; JoystickEnvCfg_Walk2 supplies the new values.
    w_lin_vel_xy = 1.0
    w_ang_vel_z = 0.5
    w_joint_vel = 1.0e-3

    cmd_norm = torch.linalg.norm(commands[:, :3], dim=-1)

    ref_joint_pos = reference_frame[:, 0:14][:, REF_LEG_JOINT_IDX]  # [N,10]
    ref_joint_vel = reference_frame[:, 14:28][:, REF_LEG_JOINT_IDX]  # [N,10]
    ref_foot_contacts = reference_frame[:, 28:30]  # [N,2]
    ref_lin_vel = reference_frame[:, 30:33]  # [N,3]
    ref_ang_vel = reference_frame[:, 33:36]  # [N,3]

    joint_pos = joints_qpos[:, ACT_LEG_JOINT_IDX]  # [N,10]
    joint_vel = joints_qvel[:, ACT_LEG_JOINT_IDX]  # [N,10]

    lin_vel_xy_rew = (
        torch.exp(-k_lin_vel_xy * torch.sum((base_lin_vel_w[:, :2] - ref_lin_vel[:, :2]) ** 2, dim=-1)) * w_lin_vel_xy
    )
    lin_vel_z_rew = torch.exp(-8.0 * (base_lin_vel_w[:, 2] - ref_lin_vel[:, 2]) ** 2) * w_lin_vel_z
    ang_vel_xy_rew = torch.exp(-2.0 * torch.sum((base_ang_vel_w[:, :2] - ref_ang_vel[:, :2]) ** 2, dim=-1)) * w_ang_vel_xy
    ang_vel_z_rew = torch.exp(-2.0 * (base_ang_vel_w[:, 2] - ref_ang_vel[:, 2]) ** 2) * w_ang_vel_z

    # 2026-07-28: joint_pos was the ONE unbounded term here — every other
    # tracking term above is exp(-err), bounded to [0, 1], while this was a
    # raw negative quadratic that grows without limit. reward_breakdown_v2.py
    # measured the consequence on imitation_v8's final policy: imitation
    # contributed -1.01/step on average (vs alive's +0.40, the largest
    # positive term), driving the summed reward negative on 81.6% of steps —
    # and `_get_rewards` clamps to [0, ...], so those steps delivered a
    # constant 0 and no gradient. The policy was effectively learning from
    # under a fifth of its experience. `bounded_joint_pos` switches this to
    # the same exp form as the velocity terms so it can't dominate; the old
    # quadratic stays reachable for direct comparison.
    if bounded_joint_pos:
        # w_joint_pos is the exp SHARPNESS; w_joint_pos_amp is how much the
        # bounded term is worth relative to the other six imitation terms.
        # v15 proved these are not interchangeable: dropping sharpness 4.0->1.5
        # raised joint_pos_rew 0.148->0.425 at matched checkpoints while the
        # actual joint error stayed put (14.0deg -> 14.3deg). Pose tracking was
        # structurally underpriced -- halving the error is worth only ~+0.02/step
        # against the fall risk of moving precisely -- and no sharpness setting
        # can fix that, because the term is bounded to [0,1] either way.
        joint_pos_rew = torch.exp(-w_joint_pos * torch.sum((joint_pos - ref_joint_pos) ** 2, dim=-1)) * w_joint_pos_amp
    else:
        joint_pos_rew = -torch.sum((joint_pos - ref_joint_pos) ** 2, dim=-1) * w_joint_pos
    joint_vel_rew = -torch.sum((joint_vel - ref_joint_vel) ** 2, dim=-1) * w_joint_vel

    ref_contacts_bool = (ref_foot_contacts > 0.5).float()
    if swing_only_contact and w_stance_violation > 0.0:
        # 2026-07-28 (v13): the plain swing_only form below pays for lifting a
        # foot the reference wants lifted, but costs NOTHING for lifting one the
        # reference wants planted. Flickering both feet therefore raises the
        # chance of overlapping a swing phase and gets paid for it -- and that is
        # exactly what imitation_v12 produced: the user watched it and reported
        # the feet "진동하는것마냥" chattering against the ground, with the
        # measured contact toggle rate at 144-319/10s against v6's 29.4 best.
        # Adding the stance-violation penalty keeps standing at 0 (feet planted
        # during stance costs nothing, and the swing term pays nothing) while
        # making chatter strictly negative.
        swing = torch.sum((1.0 - ref_contacts_bool) * (1.0 - contacts), dim=-1)
        stance_violation = torch.sum(ref_contacts_bool * (1.0 - contacts), dim=-1)
        contact_rew = (swing - w_stance_violation * stance_violation) * w_contact
    elif swing_only_contact:
        # Credit only for lifting a foot the reference says should be lifted.
        # The plain agreement form below counts a planted foot as "matching"
        # whenever the reference also has it planted -- and scripts/ref_stats.py
        # measured the reference's stance duty at 0.692/0.652, so a robot that
        # simply keeps both feet on the ground collects ~1.34 of the available
        # 2.0 for doing nothing at all. That is exactly the behavior observed on
        # imitation_v10 ("발이 붙여진 상태" -- trembling in place, feet planted,
        # never stepping). Scoring swing agreement instead makes standing worth
        # 0 here and forces actual foot alternation to earn the term.
        contact_rew = torch.sum((1.0 - ref_contacts_bool) * (1.0 - contacts), dim=-1) * w_contact
    else:
        contact_rew = torch.sum((contacts == ref_contacts_bool).float(), dim=-1) * w_contact

    reward = lin_vel_xy_rew + lin_vel_z_rew + ang_vel_xy_rew + ang_vel_z_rew + joint_pos_rew + joint_vel_rew + contact_rew
    reward = reward * (cmd_norm > 0.01).float()
    return torch.nan_to_num(reward)


def reward_path_tracking(
    path_err: torch.Tensor,  # [N,3] — lateral, cos(yaw_err), sin(yaw_err)
    k_lateral: float,
    k_yaw: float,
    w_yaw: float,
) -> torch.Tensor:
    """경로에서 벗어난 정도를 벌한다 (Disney BD-X의 path frame).

    속도 명령은 순수 rate라 yaw_rate=0이 "원래 방향으로 돌아와라"를 뜻하지
    않는다. 그래서 한 번 휘면 정책이 그 사실 자체를 관측하지 못하고 되돌릴
    이유도 없다. 적분된 경로 기준의 횡방향·방향 오차를 관측에 넣고 여기서
    보상해야 비로소 "일자로 걷기"가 학습 목표가 된다.
    """
    lateral = path_err[:, 0]
    yaw_err = torch.atan2(path_err[:, 2], path_err[:, 1])
    return torch.exp(-k_lateral * lateral**2) + torch.exp(-k_yaw * yaw_err**2) * w_yaw
