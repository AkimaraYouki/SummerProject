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
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JoystickPPORunnerCfg_Upstream",
        },
    )

except ImportError:
    pass
