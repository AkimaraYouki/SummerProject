"""태스크 id -> (환경 설정, 러너 설정) 한 곳.

같은 매핑이 scripts/ 아래 여섯 군데에 복제돼 있었다. 새 태스크를 등록하면
여섯 곳을 다 고쳐야 했고, 실제로 `reward_breakdown_v2.py`는 Path 태스크를
모른 채 남아 있다가 `KeyError: 'Isaac-OpenDuckMini-Joystick-Path-v0'` 로
죽었다 (2026-07-30에 발견).

러너 설정을 태스크와 함께 묶는 이유: 체크포인트는 **학습에 쓴 네트워크 크기**로
열어야 한다. Walk9 계열은 (512,256,128)이고 나머지는 (256,128,64)라, 잘못
고르면 `actor.0.weight` 크기 불일치로 로드가 실패한다.
"""

from __future__ import annotations

#: 태스크 id -> joystick_env_cfg 안의 설정 클래스 이름
ENV_CFG_CLASS = {
    "Isaac-OpenDuckMini-Joystick-v0": "JoystickEnvCfg",
    "Isaac-OpenDuckMini-Joystick-Walk3-v0": "JoystickEnvCfg_Walk3",
    "Isaac-OpenDuckMini-Joystick-Walk6-v0": "JoystickEnvCfg_Walk6",
    "Isaac-OpenDuckMini-Joystick-Walk9-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0": "JoystickEnvCfg_Walk9",
    "Isaac-OpenDuckMini-Joystick-Path-v0": "JoystickEnvCfg_Path",
    "Isaac-OpenDuckMini-Joystick-Grav-v0": "JoystickEnvCfg_Grav",
    "Isaac-OpenDuckMini-Joystick-Tall-v0": "JoystickEnvCfg_Tall",
    "Isaac-OpenDuckMini-Joystick-Sym-v0": "JoystickEnvCfg_Tall",  # v29: 환경은 v28 과 동일
    "Isaac-OpenDuckMini-Joystick-Taller-v0": "JoystickEnvCfg_Taller",
    "Isaac-OpenDuckMini-Joystick-HipLimit-v0": "JoystickEnvCfg_HipLimit",
    "Isaac-OpenDuckMini-Joystick-HipInward-v0": "JoystickEnvCfg_HipInward",
    "Isaac-OpenDuckMini-Joystick-ZNeck-v0": "JoystickEnvCfg_ZNeck",
    "Isaac-OpenDuckMini-Joystick-V33N-v0": "JoystickEnvCfg_V33N",
    "Isaac-OpenDuckMini-Joystick-V34C-v0": "JoystickEnvCfg_V34C",
    "Isaac-OpenDuckMini-Joystick-V34C2-v0": "JoystickEnvCfg_V34C2",
    "Isaac-OpenDuckMini-Joystick-V34C20-v0": "JoystickEnvCfg_V34C20",
    "Isaac-OpenDuckMini-Joystick-V34C10-v0": "JoystickEnvCfg_V34C10",
    "Isaac-OpenDuckMini-Joystick-V34U-v0": "JoystickEnvCfg_V34U",
    "Isaac-OpenDuckMini-Joystick-V35-v0": "JoystickEnvCfg_V35",
    "Isaac-OpenDuckMini-Joystick-V36-v0": "JoystickEnvCfg_V36",
    "Isaac-OpenDuckMini-Joystick-V37-v0": "JoystickEnvCfg_V37",
    "Isaac-OpenDuckMini-Joystick-V38-v0": "JoystickEnvCfg_V38",
    "Isaac-OpenDuckMini-Joystick-TallSafe-v0": "JoystickEnvCfg_TallSafe",
    "Isaac-OpenDuckMini-Joystick-HipInwardSafe-v0": "JoystickEnvCfg_HipInwardSafe",
    "Isaac-OpenDuckMini-Joystick-Upstream-v0": "JoystickEnvCfg_Upstream",
}

#: 큰 네트워크(512,256,128) + gamma 0.97 을 쓰는 태스크들.
#: 여기 없으면 기본 JoystickPPORunnerCfg (256,128,64).
_BIG_NET_TASKS = {
    "Isaac-OpenDuckMini-Joystick-Walk9-v0",
    "Isaac-OpenDuckMini-Joystick-Walk9G97-v0",
    "Isaac-OpenDuckMini-Joystick-Path-v0",
    "Isaac-OpenDuckMini-Joystick-Grav-v0",
    "Isaac-OpenDuckMini-Joystick-Tall-v0",
    "Isaac-OpenDuckMini-Joystick-Sym-v0",
    "Isaac-OpenDuckMini-Joystick-Taller-v0",
    "Isaac-OpenDuckMini-Joystick-HipLimit-v0",
    "Isaac-OpenDuckMini-Joystick-HipInward-v0",
    "Isaac-OpenDuckMini-Joystick-ZNeck-v0",
    "Isaac-OpenDuckMini-Joystick-V33N-v0",
    "Isaac-OpenDuckMini-Joystick-V34C-v0",
    "Isaac-OpenDuckMini-Joystick-V34C2-v0",
    "Isaac-OpenDuckMini-Joystick-V34C20-v0",
    "Isaac-OpenDuckMini-Joystick-V34C10-v0",
    "Isaac-OpenDuckMini-Joystick-V34U-v0",
    "Isaac-OpenDuckMini-Joystick-V35-v0",
    "Isaac-OpenDuckMini-Joystick-V36-v0",
    "Isaac-OpenDuckMini-Joystick-V37-v0",
    "Isaac-OpenDuckMini-Joystick-V38-v0",
    "Isaac-OpenDuckMini-Joystick-TallSafe-v0",
    "Isaac-OpenDuckMini-Joystick-HipInwardSafe-v0",
}


def env_cfg_for(task: str):
    """태스크 id 에 맞는 환경 설정 **인스턴스**를 만들어 돌려준다."""
    from open_duck_mini_isaaclab.tasks.velocity import joystick_env_cfg as _cm

    try:
        name = ENV_CFG_CLASS[task]
    except KeyError:
        raise KeyError(
            f"모르는 태스크: {task!r}. task_registry.ENV_CFG_CLASS 에 추가하세요. "
            f"알려진 것: {sorted(ENV_CFG_CLASS)}"
        ) from None
    return getattr(_cm, name)()


def runner_cfg_for(task: str):
    """체크포인트를 열 때 쓸 러너 설정 **인스턴스**."""
    from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import (
        JoystickPPORunnerCfg,
        JoystickPPORunnerCfg_Gamma097,
    )

    if task not in ENV_CFG_CLASS:
        raise KeyError(
            f"모르는 태스크: {task!r}. task_registry.ENV_CFG_CLASS 에 추가하세요."
        )
    # v29 만 러너가 다르다(미러 손실). 환경은 v28 과 같아 ENV_CFG_CLASS 로는
    # 구분되지 않으므로 여기서 따로 잡는다.
    if task == "Isaac-OpenDuckMini-Joystick-Sym-v0":
        from open_duck_mini_isaaclab.agents.rsl_rl_ppo_cfg import JoystickPPORunnerCfg_Symmetry
        return JoystickPPORunnerCfg_Symmetry()
    return JoystickPPORunnerCfg_Gamma097() if task in _BIG_NET_TASKS else JoystickPPORunnerCfg()
