"""평지가 아닌 지형 프리셋. 측정(`scripts/diag/terrain_test.py`)과
재생(`scripts/play_fixed_cmd.py --terrain ...`)이 **같은 정의**를 쓰게 한 곳.

이 프로젝트의 학습은 전부 `terrain_type="plane"`(완벽한 평면)에서 돌았고,
관측에는 지형을 보는 수단이 하나도 없다 — height scan 도 ray caster 도 없다.
즉 정책은 **완전한 blind** 다. 발밑이 달라져도 자기가 평지에 있다고 믿고 IMU 와
관절 되먹임만으로 반응한다.

**스케일이 핵심이다.** IsaacLab 의 `ROUGH_TERRAINS_CFG` 기본값은 계단 5~23 cm,
요철 2~10 cm 인데 그건 ANYmal(선 키 ~55 cm) 기준이다. 우리 로봇은 서 있는 높이가
125~140 mm 이고 발 들어올림이 4 cm 다 — **23 cm 계단은 로봇 키보다 높다.**
그래서 같은 지형을 IsaacLab 기본값(`*_isaac`)과 로봇 스케일(`*_s`, 대략 1/4)
두 벌로 둔다. 기본값에서 넘어지는 것은 정책 탓이 아니라 지형이 로봇에 안 맞는
것이므로, 둘을 섞어 읽지 말 것.

`vertical_scale` 은 요철 최소 단위(2 mm)보다 작아야 한다. IsaacLab 기본값 5 mm
를 그대로 쓰면 로봇 스케일 지형이 통째로 뭉개져 평지가 된다.
"""

# 이름 -> 설명. CLI 의 choices 와 도움말이 여기서 나온다.
TERRAIN_CHOICES = {
    "plane": "완벽한 평면 (기준선 — 학습 환경과 동일)",
    "rough_s": "요철 2~12 mm  (로봇 스케일)",
    "rough_m": "요철 5~25 mm  (로봇 스케일, 거침)",
    "rough_isaac": "요철 2~10 cm  (IsaacLab 기본값 — ANYmal 기준)",
    "slope_s": "경사 0~6도    (로봇 스케일)",
    "slope_isaac": "경사 0~22도   (IsaacLab 기본값)",
    "stairs_s": "계단 5~15 mm  (로봇 스케일)",
    "stairs_isaac": "계단 5~23 cm  (IsaacLab 기본값 — 로봇 키보다 높다)",
    "grid_s": "격자 요철 5~20 mm (로봇 스케일)",
    # 언덕 = 물결 지형. IsaacLab 에 hills 라는 이름은 없고 HfWaveTerrainCfg 가
    # 그 역할이다 (사인파 능선 num_waves 개). 요철과 달리 **파장이 길어서**
    # 로봇이 한 걸음 안에 다 못 넘고 오르내리게 된다 — 경사와 요철의 중간.
    "hills_s": "언덕(물결) 진폭 5~20 mm, 4파  (로봇 스케일)",
    "hills_m": "언덕(물결) 진폭 20~50 mm, 3파  (로봇 스케일, 큼)",
    "hills_isaac": "언덕(물결) 진폭 5~15 cm, 4파 (ANYmal 스케일)",
    "stones_s": "징검다리 돌 폭 6~12 cm, 틈 2 cm  (로봇 스케일)",
    "obstacles_s": "흩뿌린 장애물 높이 5~20 mm  (로봇 스케일)",
    # IsaacSim 이 기본 제공하는 USD 지형. 절차적 생성이 아니라 **고정 크기 에셋**
    # 이라 로봇에 맞게 줄일 수 없다 — ANYmal/Carter 급 기준이다. 참고용.
    # (2026-08-08 확인: hills.usd 는 없다. 이 네 개가 전부다.)
    "usd_rough": "IsaacSim 기본 rough_plane.usd (고정 크기, 네트워크 필요)",
    "usd_slope": "IsaacSim 기본 slope.usd     (고정 크기, 네트워크 필요)",
    "usd_stairs": "IsaacSim 기본 stairs.usd    (고정 크기, 네트워크 필요)",
}

# terrain_type="usd" 로 붙이는 것들 -> 에셋 파일명
_USD_TERRAINS = {
    "usd_rough": "rough_plane.usd",
    "usd_slope": "slope.usd",
    "usd_stairs": "stairs.usd",
}


def _sub(name: str):
    """지형 하나짜리 sub-terrain 설정. IsaacLab 임포트는 지연시킨다 —
    이 모듈이 AppLauncher 보다 먼저 임포트돼도 깨지지 않게."""
    import isaaclab.terrains as tg

    if name == "rough_s":
        return tg.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.002, 0.012), noise_step=0.002, border_width=0.25)
    if name == "rough_m":
        return tg.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.005, 0.025), noise_step=0.005, border_width=0.25)
    if name == "rough_isaac":
        return tg.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25)
    if name == "slope_s":
        return tg.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.10), platform_width=1.0, border_width=0.25)
    if name == "slope_isaac":
        return tg.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.40), platform_width=1.0, border_width=0.25)
    if name == "stairs_s":
        return tg.HfPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.005, 0.015), step_width=0.15,
            platform_width=1.0, border_width=0.25)
    if name == "stairs_isaac":
        return tg.HfPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3,
            platform_width=1.0, border_width=0.25)
    if name == "grid_s":
        return tg.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.10, grid_height_range=(0.005, 0.020), platform_width=1.0)
    if name == "hills_s":
        return tg.HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.005, 0.020), num_waves=4, border_width=0.25)
    if name == "hills_m":
        return tg.HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.020, 0.050), num_waves=3, border_width=0.25)
    if name == "hills_isaac":
        return tg.HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.05, 0.15), num_waves=4, border_width=0.25)
    if name == "stones_s":
        return tg.HfSteppingStonesTerrainCfg(
            proportion=1.0, stone_height_max=0.005, stone_width_range=(0.06, 0.12),
            stone_distance_range=(0.005, 0.02), holes_depth=-0.05,
            platform_width=1.0, border_width=0.25)
    if name == "obstacles_s":
        return tg.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0, obstacle_height_mode="choice", obstacle_width_range=(0.05, 0.15),
            obstacle_height_range=(0.005, 0.020), num_obstacles=40,
            platform_width=1.0, border_width=0.25)
    raise ValueError(f"모르는 지형: {name}")


def apply_terrain(env_cfg, name: str, num_rows: int = 4, num_cols: int = 4):
    """`env_cfg.terrain` 을 지형 프리셋으로 바꾼다. "plane" 이면 그대로 둔다.

    지면 마찰은 학습 때 값(1.0/1.0, multiply)을 그대로 유지한다 — 바꾸는 것은
    **지형 형상 하나**여야 결과를 읽을 수 있다.
    """
    if name == "plane":
        return env_cfg

    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

    _mat = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
        static_friction=1.0, dynamic_friction=1.0,
    )

    if name in _USD_TERRAINS:
        # IsaacSim 기본 에셋. 클라우드 Nucleus 에서 받아오므로 네트워크가 필요하고,
        # 크기가 고정이라 로봇에 맞게 줄일 수 없다.
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        env_cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/{_USD_TERRAINS[name]}",
            collision_group=-1, physics_material=_mat, debug_vis=False,
        )
        return env_cfg

    env_cfg.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(4.0, 4.0), border_width=1.0, num_rows=num_rows, num_cols=num_cols,
            horizontal_scale=0.05, vertical_scale=0.001, slope_threshold=0.75,
            use_cache=False, curriculum=False,
            sub_terrains={"t": _sub(name)},
        ),
        collision_group=-1, physics_material=_mat, debug_vis=False,
    )
    return env_cfg
