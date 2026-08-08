"""Open Duck Mini V2 Isaac Lab extension — registers the Joystick task.

Registration is best-effort: this package also contains pure-torch modules
(reference_motion/, tasks/velocity/rewards.py, joint_order.py) that are
imported directly by tests/*.py on machines with no Isaac Sim/gymnasium
install (see tests/README or docs/decisions.md — Mac-runnable verification
is a deliberate design goal, not an afterthought). Since any submodule
import runs this file first, a hard ImportError here would break those
tests too. On a real Isaac Lab machine, gymnasium and isaaclab_rl are
always present and registration proceeds normally.
"""

try:
    import gymnasium as gym

    from . import agents

    gym.register(
        id="Isaac-OpenDuckMini-Joystick-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg",
        },
    )

    # 살아있는 변형만 등록한다 (2026-07-29 정리). 실패로 판정된 16개 설정은
    # joystick_env_cfg.py에서 삭제했고, 값과 실패 이유는 git 이력과
    # docs/training_log.md에 남아 있다.
    #
    #   Walk3    imitation_v13 — 명령 추종 최고 (전진 0.117 m/s)
    #   Walk6    imitation_v17 — 육안 판정 최고 보행
    #   Upstream imitation_v14 — Open_Duck_Playground 리워드 기준선
    for _variant_suffix, _variant_cfg_cls in [
        ("Walk3", "JoystickEnvCfg_Walk3"),
        ("Walk6", "JoystickEnvCfg_Walk6"),
        ("Upstream", "JoystickEnvCfg_Upstream"),
    ]:
        gym.register(
            id=f"Isaac-OpenDuckMini-Joystick-{_variant_suffix}-v0",
            entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:{_variant_cfg_cls}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg",
            },
        )

    # imitation_v20 — 비대칭 크리틱 + upstream 네트워크/PPO. 러너 설정까지
    # 바꾸므로 따로 등록한다.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Walk9-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Walk9",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )




    # imitation_v24 — v22에서 gamma만 0.99 -> 0.97.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Walk9G97-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Walk9",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v30 — v28 의 키를 한 단계 더 (walk_com_height 0.19).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Taller-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Taller",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v31 — v28 + 고관절 roll/yaw 이탈 상한 (실기 이식용, 접촉 = 파손).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-HipLimit-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_HipLimit",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v32 — v28 + 고관절이 안쪽으로 벗어나는 것만 벌점 (v31 재설계).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-HipInward-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_HipInward",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v33 — v32 + Z 자 목 자세로 학습 (자세만 바꾸면 넘어진다).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-ZNeck-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_ZNeck",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v33 에서 CAD 모델(2.7140 kg)과 레퍼런스(ref_g115)만 새것으로 바꾼 것.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V33N-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V33N",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v33n + 정지 시 보행 위상 고정.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V34C-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V34C",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V34C10-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V34C10",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v34c10 + 정지 목표 자세를 몸통 수직으로.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V34U-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V34U",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v34c10 + 정지 시 몸통 수직을 중력벡터로 직접 요구. 로봇은 새 CAD(2.7430 kg).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V35-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V35",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v35 + 정지 좌우 대칭 + 액션 저역통과를 학습에 포함.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V36-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V36",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v36 에서 저역통과를 정지에만 거는 변형.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V37-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V37",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v37 + hip_inward 를 보행 중에만 (정지 좌우대칭과의 충돌 제거).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V38-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V38",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # v38 + 저역통과 제거 + 액션 2차차분(진동) 벌점.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V39-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V39",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V34C20-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V34C20",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    gym.register(
        id="Isaac-OpenDuckMini-Joystick-V34C2-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_V34C2",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # 안전 필터를 켠 변형 (학습용이 아니라 평가용).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-TallSafe-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_TallSafe",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    gym.register(
        id="Isaac-OpenDuckMini-Joystick-HipInwardSafe-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_HipInwardSafe",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v29 — v28 과 환경은 같고 러너에 좌우 미러 손실만 더한다.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Sym-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Tall",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Symmetry",
        },
    )

    # imitation_v28 — v27 + 로봇을 더 세운다 (레퍼런스 walk_com_height 0.175).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Tall-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Tall",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v27 — v26(Path) + 정책 관측에 중력 방향 3차원.
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Grav-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Grav",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

    # imitation_v25 — v24 + path frame (경로 추종으로 직진성 확보).
    gym.register(
        id="Isaac-OpenDuckMini-Joystick-Path-v0",
        entry_point=f"{__name__}.tasks.velocity.joystick_env:JoystickEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.tasks.velocity.joystick_env_cfg:JoystickEnvCfg_Path",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Gamma097",
        },
    )

except ImportError:
    pass
