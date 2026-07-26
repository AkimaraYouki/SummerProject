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

    # alive_scale sweep variants — see joystick_env_cfg.py's comment above
    # JoystickEnvCfg_Alive5/_Alive10 for why. The base task above now runs
    # with alive_scale=2.0.
    for _variant_suffix, _variant_cfg_cls in [
        ("Alive5", "JoystickEnvCfg_Alive5"),
        ("Alive10", "JoystickEnvCfg_Alive10"),
        # alive_scale x w_joint_pos sweep (2026-07-27) — see
        # joystick_env_cfg.py's comment above JoystickEnvCfg_A10J10 for why.
        ("A10J10", "JoystickEnvCfg_A10J10"),
        ("A5J15", "JoystickEnvCfg_A5J15"),
        ("A20J5", "JoystickEnvCfg_A20J5"),
        ("A5J5", "JoystickEnvCfg_A5J5"),
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
except ImportError:
    pass
