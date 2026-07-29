"""랩PC(rsl-rl 2.x)에서 학습한 것을 이 PC(rsl-rl 5.0.1)에서 그대로 여는 이식 계층.

2026-07-30, 학습 환경을 랩PC에서 사용자 PC로 옮기면서 IsaacLab 체크아웃이 함께
바뀌었다. SSD는 그대로 옮겨 꽂아 로그와 체크포인트가 전부 따라왔지만, IsaacLab은
각 기계의 홈에 따로 설치돼 있어 따라오지 않았다:

    랩PC     isaaclab_rl 0.2.0   +  rsl-rl 2.x   (학습에 쓴 것)
    이 PC    isaaclab_rl 0.5.1   +  rsl-rl 5.0.1

그 사이에 rsl-rl의 API가 두 군데서 끊겼고, 둘 다 재생/측정 스크립트를 죽였다.

1. **설정 스키마** — `policy: RslRlPpoActorCriticCfg` 하나가 `actor`/`critic`
   두 개의 `RslRlMLPModelCfg`로 갈라졌다. 구 스키마를 그대로 넘기면
   `PPO.construct_algorithm`이 `cfg["actor"].pop("class_name")`에서
   `KeyError: 'class_name'`으로 죽는다 (인수인계 문서의 "아직 안 되는 것" 2번).
   상류가 `handle_deprecated_rsl_rl_cfg`로 변환 경로를 제공하므로 직접 다시
   짜지 않고 그것을 부른다 — 상류 train.py/play.py도 같은 함수를 부른다.

2. **체크포인트 레이아웃** — 저장 포맷이 통째로 바뀌었다. 구버전은 액터·크리틱을
   `ActorCritic` 모듈 하나에 담아 `model_state_dict` 한 덩어리로 저장했고,
   관측 정규화는 러너가 모델 **바깥**에서 들고 있었다. 신버전은 액터와 크리틱이
   독립한 `MLPModel`이고 정규화기가 모델 **안**에 들어갔다. 그래서 키 이름이
   전부 어긋난다. `logs/.../imitation_v25/model_1500.pt`를 실제로 열어 확인한
   대응은 이렇다:

       구 (rsl-rl 2.x)                       신 (rsl-rl 5.0.1)
       ------------------------------------  --------------------------------
       model_state_dict["actor.{N}.*"]       actor_state_dict["mlp.{N}.*"]
       model_state_dict["critic.{N}.*"]      critic_state_dict["mlp.{N}.*"]
       model_state_dict["std"]               actor_state_dict["distribution.std_param"]
       obs_norm_state_dict                   actor_state_dict["obs_normalizer.*"]
       privileged_obs_norm_state_dict        critic_state_dict["obs_normalizer.*"]

   레이어 인덱스는 손대지 않아도 된다. 구 `ActorCritic`의 액터는
   Linear-ELU-Linear-ELU-Linear-ELU-Linear를 `nn.Sequential`에 그대로 쌓아
   0/2/4/6번이 선형층이었고, 신 `MLP`도 `nn.Sequential`을 같은 순서로 쌓아
   0/2/4/6번이 선형층이다. v25 체크포인트의 실제 shape
   (actor.0 = 512x104, .2 = 256x512, .4 = 128x256, .6 = 14x128)이 신 모델의
   hidden_dims=(512,256,128), 액션 14와 정확히 맞는다.

   정규화기 버퍼 이름(`_mean`/`_var`/`_std`/`count`)은 양쪽이 같아서 접두사만
   붙이면 된다.

주의 — **이 파일은 rsl_rl도 isaaclab도 import 하지 않는다.** 변환 자체는 dict를
옮겨 담는 일뿐이라 Isaac Sim 없이 검증할 수 있어야 하고, 실제로
`tests/test_rsl_rl_checkpoint_conversion.py`가 그렇게 돈다. 무거운 import는
전부 함수 안에 둔다.
"""

from __future__ import annotations

from importlib import metadata

#: 구버전 체크포인트를 알아보는 표식. 신버전은 액터/크리틱을 따로 저장하므로
#: 이 키가 없다.
LEGACY_MODEL_KEY = "model_state_dict"

#: 구버전 러너가 모델 바깥에 들고 있던 정규화기 -> 신버전 모델 안의 정규화기.
_NORMALIZER_KEYS = {
    "obs_norm_state_dict": "actor",
    "privileged_obs_norm_state_dict": "critic",
}


def installed_rsl_rl_version() -> str:
    """설치된 rsl-rl 버전. 상류 train.py와 같은 방식으로 읽는다."""
    return metadata.version("rsl-rl-lib")


def prepare_agent_cfg(agent_cfg):
    """구 스키마(`policy`)를 설치된 rsl-rl이 이해하는 모양으로 옮긴다.

    `handle_deprecated_rsl_rl_cfg`는 agent_cfg를 제자리에서 고치고 같은 객체를
    돌려준다. rsl-rl >= 4.0에서는 `policy`로부터 `actor`/`critic`을 유추한 뒤
    `policy`를 지우고, `empirical_normalization`을 모델별
    `obs_normalization`으로 옮긴다. rsl-rl < 4.0에서는 반대로 `actor`/`critic`을
    지우고 `policy`를 남긴다 — 그래서 이 함수는 양쪽 어디서도 안전하다.
    """
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    return handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version())


def build_runner(env, agent_cfg, log_dir=None):
    """스키마를 맞춘 뒤 OnPolicyRunner를 만든다.

    스크립트들이 `OnPolicyRunner(env, agent_cfg.to_dict(), ...)`를 직접 부르던
    자리를 그대로 대신한다.
    """
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = prepare_agent_cfg(agent_cfg)
    return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)


def is_legacy_checkpoint(loaded: dict) -> bool:
    """구버전(rsl-rl < 4.0) 저장 포맷인지."""
    return LEGACY_MODEL_KEY in loaded


def convert_legacy_checkpoint(loaded: dict) -> dict:
    """구버전 체크포인트 dict를 신버전이 읽는 모양으로 옮긴다.

    텐서는 건드리지 않고 어느 dict의 어느 키에 놓일지만 바꾼다 — 그래서 torch
    없이도 돌고, 값이 조용히 바뀔 여지가 없다.

    옵티마이저 상태는 인덱스를 다시 매길 필요가 없다. 구 `ActorCritic`은
    `std`, `actor.*`, `critic.*` 순으로 파라미터를 등록했고, 신 PPO는
    `chain(actor.parameters(), critic.parameters())`로 옵티마이저를 만드는데
    신 액터의 등록 순서가 `distribution.std_param`, `mlp.*`라 이어 붙이면
    구버전과 같은 순서가 된다. 다만 이건 두 구현이 우연히 맞아떨어진 것이라
    가정으로 남기지 않고, 실제로 쓸 때 `load_checkpoint`가 shape로 검증한다.
    """
    model = loaded[LEGACY_MODEL_KEY]
    states: dict[str, dict] = {"actor": {}, "critic": {}}

    for key, value in model.items():
        if key.startswith("actor."):
            states["actor"]["mlp." + key[len("actor.") :]] = value
        elif key.startswith("critic."):
            states["critic"]["mlp." + key[len("critic.") :]] = value
        elif key == "std":
            # noise_std_type="scalar" — 신버전은 표준편차를 분포 모듈이 들고 있다.
            states["actor"]["distribution.std_param"] = value
        elif key == "log_std":
            states["actor"]["distribution.log_std_param"] = value
        else:
            raise KeyError(
                f"구버전 체크포인트에 모르는 키가 있습니다: {key!r}. "
                "액터/크리틱 MLP와 표준편차 말고 다른 것이 저장돼 있다면 "
                "여기서 조용히 버리는 대신 대응을 명시적으로 추가해야 합니다."
            )

    for legacy_key, target in _NORMALIZER_KEYS.items():
        normalizer = loaded.get(legacy_key)
        if not normalizer:
            continue
        for key, value in normalizer.items():
            states[target]["obs_normalizer." + key] = value

    converted = {
        "actor_state_dict": states["actor"],
        "critic_state_dict": states["critic"],
        "iter": loaded.get("iter", 0),
        "infos": loaded.get("infos"),
    }
    if "optimizer_state_dict" in loaded:
        converted["optimizer_state_dict"] = loaded["optimizer_state_dict"]
    return converted


def load_checkpoint(runner, path: str, *, load_optimizer: bool = False, map_location: str | None = None):
    """체크포인트를 러너에 싣는다. 구버전 포맷이면 자동으로 변환한다.

    `runner.load(path)`를 대신한다. 옵티마이저는 기본으로 싣지 않는다 — 재생과
    측정에는 필요 없고, 구버전 상태를 신버전 옵티마이저에 얹는 것은 파라미터
    순서 가정에 기대는 유일한 부분이라 학습을 이어갈 때만 켜서 검증까지 하고
    쓰는 편이 안전하다.
    """
    import torch

    loaded = torch.load(path, weights_only=False, map_location=map_location)
    legacy = is_legacy_checkpoint(loaded)
    if legacy:
        loaded = convert_legacy_checkpoint(loaded)
        if load_optimizer:
            _verify_optimizer_layout(runner.alg.optimizer, loaded.get("optimizer_state_dict"))

    load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": load_optimizer,
        "iteration": True,
        "rnd": False,
    }
    runner.alg.load(loaded, load_cfg, strict=True)
    runner.current_learning_iteration = int(loaded.get("iter", 0))
    print(
        f"[compat] 체크포인트 로드: {path} (iter {runner.current_learning_iteration}"
        f"{', 구버전 포맷 변환' if legacy else ''})",
        flush=True,
    )
    return loaded.get("infos")


def _verify_optimizer_layout(optimizer, optimizer_state) -> None:
    """구버전 옵티마이저 상태의 파라미터 순서가 신버전과 같은지 shape로 확인한다.

    `convert_legacy_checkpoint`의 docstring이 설명한 "순서가 우연히 같다"는
    관찰이 이 체크포인트에도 실제로 성립하는지 보는 곳. 어긋나면 조용히 엉뚱한
    모멘텀을 얹는 대신 멈춘다.
    """
    if not optimizer_state:
        raise ValueError("구버전 체크포인트에 옵티마이저 상태가 없습니다.")

    params = [p for group in optimizer.param_groups for p in group["params"]]
    for index, entry in optimizer_state.get("state", {}).items():
        moment = entry.get("exp_avg")
        if moment is None:
            continue
        if index >= len(params) or tuple(params[index].shape) != tuple(moment.shape):
            saved_shape = tuple(moment.shape)
            live_shape = tuple(params[index].shape) if index < len(params) else None
            raise ValueError(
                "구/신 옵티마이저의 파라미터 순서가 다릅니다 "
                f"(index {index}: 저장 {saved_shape} vs 현재 {live_shape}). "
                "옵티마이저 상태 없이(load_optimizer=False) 이어가거나, "
                "인덱스 대응을 명시적으로 짜야 합니다."
            )
