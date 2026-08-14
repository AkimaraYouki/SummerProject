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
    "Isaac-OpenDuckMini-Joystick-V39-v0": "JoystickEnvCfg_V39",
    "Isaac-OpenDuckMini-Joystick-V40-v0": "JoystickEnvCfg_V40",
    "Isaac-OpenDuckMini-Joystick-V41-v0": "JoystickEnvCfg_V41",
    "Isaac-OpenDuckMini-Joystick-V42-v0": "JoystickEnvCfg_V42",
    "Isaac-OpenDuckMini-Joystick-V43-v0": "JoystickEnvCfg_V43",
    "Isaac-OpenDuckMini-Joystick-V44-v0": "JoystickEnvCfg_V44",
    "Isaac-OpenDuckMini-Joystick-V46-v0": "JoystickEnvCfg_V46",
    "Isaac-OpenDuckMini-Joystick-V47-v0": "JoystickEnvCfg_V47",
    "Isaac-OpenDuckMini-Joystick-V48-v0": "JoystickEnvCfg_V48",
    "Isaac-OpenDuckMini-Joystick-V49-v0": "JoystickEnvCfg_V49",
    "Isaac-OpenDuckMini-Joystick-V50-v0": "JoystickEnvCfg_V50",
    "Isaac-OpenDuckMini-Joystick-V51-v0": "JoystickEnvCfg_V51",
    "Isaac-OpenDuckMini-Joystick-V52-v0": "JoystickEnvCfg_V52",
    "Isaac-OpenDuckMini-Joystick-V54-v0": "JoystickEnvCfg_V54",
    "Isaac-OpenDuckMini-Joystick-V53-v0": "JoystickEnvCfg_V53",
    "Isaac-OpenDuckMini-Joystick-V55-v0": "JoystickEnvCfg_V55",
    "Isaac-OpenDuckMini-Joystick-V56-v0": "JoystickEnvCfg_V56",
    "Isaac-OpenDuckMini-Joystick-V57-v0": "JoystickEnvCfg_V57",
    "Isaac-OpenDuckMini-Joystick-V58-v0": "JoystickEnvCfg_V58",
    "Isaac-OpenDuckMini-Joystick-V59-v0": "JoystickEnvCfg_V59",
    "Isaac-OpenDuckMini-Joystick-V60-v0": "JoystickEnvCfg_V60",
    "Isaac-OpenDuckMini-Joystick-V61-v0": "JoystickEnvCfg_V61",
    "Isaac-OpenDuckMini-Joystick-V62-v0": "JoystickEnvCfg_V62",
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
    "Isaac-OpenDuckMini-Joystick-V39-v0",
    "Isaac-OpenDuckMini-Joystick-V40-v0",
    "Isaac-OpenDuckMini-Joystick-V41-v0",
    "Isaac-OpenDuckMini-Joystick-V42-v0",
    "Isaac-OpenDuckMini-Joystick-V43-v0",
    "Isaac-OpenDuckMini-Joystick-V44-v0",
    "Isaac-OpenDuckMini-Joystick-V46-v0",
    "Isaac-OpenDuckMini-Joystick-V47-v0",
    "Isaac-OpenDuckMini-Joystick-V48-v0",
    "Isaac-OpenDuckMini-Joystick-V49-v0",
    "Isaac-OpenDuckMini-Joystick-V50-v0",
    "Isaac-OpenDuckMini-Joystick-V51-v0",
    "Isaac-OpenDuckMini-Joystick-V52-v0",
    "Isaac-OpenDuckMini-Joystick-V54-v0",
    "Isaac-OpenDuckMini-Joystick-V53-v0",
    "Isaac-OpenDuckMini-Joystick-V55-v0",
    "Isaac-OpenDuckMini-Joystick-V56-v0",
    "Isaac-OpenDuckMini-Joystick-V57-v0",
    "Isaac-OpenDuckMini-Joystick-V58-v0",
    "Isaac-OpenDuckMini-Joystick-V59-v0",
    "Isaac-OpenDuckMini-Joystick-V60-v0",
    "Isaac-OpenDuckMini-Joystick-V61-v0",
    "Isaac-OpenDuckMini-Joystick-V62-v0",
    "Isaac-OpenDuckMini-Joystick-TallSafe-v0",
    "Isaac-OpenDuckMini-Joystick-HipInwardSafe-v0",
}


def _apply_lpf_override(cfg) -> None:
    """`ODM_LPF` 가 있으면 액션 저역통과 alpha 를 덮어쓴다 (odm 의 `--lpf`).

    왜 환경변수 하나인가: 필터를 걸어보고 싶은 곳이 play·measure·diag 일곱 군데다.
    스크립트마다 `--smooth` 를 심으면 이 파일 맨 위 주석이 경고하는 **"여섯 군데
    복제"** 를 그대로 반복하게 된다. 설정 인스턴스를 만드는 길목이 여기 하나뿐이니
    여기서 한 번만 덮어쓴다.

    보행용과 정지용 alpha 를 **함께** 덮는 게 핵심이다. v37 에서 정지용이
    `action_lowpass_alpha_standstill` 로 분리됐기 때문에, 보행용만 바꾸면
    v39·v40 처럼 정지 alpha 가 0.0 인 정책에서는 **정지 중에 필터가 아예 안 걸린다**
    — 정작 떨림을 보고 싶은 구간이 정지인데.

    형식:  ODM_LPF=0.5        보행·정지 둘 다 0.5
           ODM_LPF=0.0,0.7    보행 0.0 / 정지 0.7 (쉼표로 따로)
    """
    import os

    raw = os.environ.get("ODM_LPF", "").strip()
    if not raw:
        return

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2:
        raise SystemExit(f"ODM_LPF 형식이 잘못됐다: {raw!r} (예: 0.5 또는 0.0,0.7)")
    try:
        a_move, a_still = (float(p) for p in parts)
    except ValueError:
        raise SystemExit(f"ODM_LPF 를 숫자로 못 읽었다: {raw!r}") from None
    for a in (a_move, a_still):
        # a=1 이면 액션이 영원히 갱신되지 않는다 (y = 1*y_prev + 0*x).
        if not 0.0 <= a < 1.0:
            raise SystemExit(f"ODM_LPF alpha 는 0 이상 1 미만이어야 한다: {a}")

    cfg.action_lowpass_alpha = a_move
    cfg.action_lowpass_alpha_standstill = a_still

    import math

    fs = 1.0 / (cfg.sim.dt * cfg.decimation)
    def _fc(a: float) -> str:
        if a <= 0.0:
            return "끔"
        c = 1.0 - (1.0 - a) ** 2 / (2.0 * a)
        f = fs / (2 * math.pi) * math.acos(max(-1.0, min(1.0, c)))
        return f"{f:.1f} Hz, 군지연 {a / (1 - a) * 1000 / fs:.0f} ms"

    print(f"[ODM_LPF] 액션 저역통과 — 보행 a={a_move} ({_fc(a_move)}) · "
          f"정지 a={a_still} ({_fc(a_still)})  [제어 {fs:.0f} Hz, 보행 1.85 Hz]",
          flush=True)
    if max(a_move, a_still) >= 0.8:
        print("[ODM_LPF] ⚠️ a>=0.8 은 차단이 1.8 Hz 로 보행 주파수(1.85 Hz)보다 낮다 "
              "— 보행 자체가 감쇠된다", flush=True)


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
    cfg = getattr(_cm, name)()
    _apply_lpf_override(cfg)
    return cfg


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
