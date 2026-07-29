"""구버전(rsl-rl 2.x) 체크포인트 -> 신버전(rsl-rl 5.x) 변환 검증.

torch도 Isaac Sim도 없이 도는 순수 테스트다. 변환은 텐서를 만지지 않고 dict를
옮겨 담기만 하므로, 텐서 자리에 아무 객체나 넣어도 대응이 맞는지 확인할 수 있다.
그래서 여기서는 문자열을 표식으로 쓴다 — 값이 엉뚱한 자리로 가면 바로 드러난다.

레이아웃은 상상해서 만든 것이 아니라
`logs/rsl_rl/open_duck_mini_v2_joystick/2026-07-29_17-22-59_imitation_v25/model_1500.pt`
를 실제로 열어 본 것을 그대로 옮겼다 (v25 = path frame, 512/256/128, 액션 14,
관측 104 / 크리틱 208).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from open_duck_mini_isaaclab.agents.rsl_rl_compat import (  # noqa: E402
    convert_legacy_checkpoint,
    is_legacy_checkpoint,
)

HIDDEN_LAYER_INDICES = (0, 2, 4, 6)


def _legacy_checkpoint(with_privileged=True):
    """v25의 model_1500.pt와 같은 키 구성."""
    model = {"std": "std"}
    for net in ("actor", "critic"):
        for i in HIDDEN_LAYER_INDICES:
            model[f"{net}.{i}.weight"] = f"{net}.{i}.weight"
            model[f"{net}.{i}.bias"] = f"{net}.{i}.bias"
    ckpt = {
        "model_state_dict": model,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "iter": 1500,
        "infos": None,
        "obs_norm_state_dict": {"_mean": "a_mean", "_var": "a_var", "_std": "a_std", "count": "a_count"},
    }
    if with_privileged:
        ckpt["privileged_obs_norm_state_dict"] = {
            "_mean": "c_mean", "_var": "c_var", "_std": "c_std", "count": "c_count",
        }
    return ckpt


def test_detects_legacy_format():
    assert is_legacy_checkpoint(_legacy_checkpoint())
    assert not is_legacy_checkpoint({"actor_state_dict": {}, "critic_state_dict": {}})


def test_key_layout_matches_new_models():
    converted = convert_legacy_checkpoint(_legacy_checkpoint())

    expected_actor = {"distribution.std_param", "obs_normalizer._mean", "obs_normalizer._var",
                      "obs_normalizer._std", "obs_normalizer.count"}
    expected_critic = {"obs_normalizer._mean", "obs_normalizer._var",
                       "obs_normalizer._std", "obs_normalizer.count"}
    for i in HIDDEN_LAYER_INDICES:
        expected_actor |= {f"mlp.{i}.weight", f"mlp.{i}.bias"}
        expected_critic |= {f"mlp.{i}.weight", f"mlp.{i}.bias"}

    assert set(converted["actor_state_dict"]) == expected_actor
    assert set(converted["critic_state_dict"]) == expected_critic
    # 크리틱에는 분포가 없다. 액션 표준편차가 크리틱으로 새면 strict 로드가
    # "unexpected key"로 죽는다.
    assert not any(k.startswith("distribution.") for k in converted["critic_state_dict"])


def test_values_land_in_the_right_slots():
    converted = convert_legacy_checkpoint(_legacy_checkpoint())
    actor, critic = converted["actor_state_dict"], converted["critic_state_dict"]

    # 레이어 인덱스는 보존된다 (구 Sequential과 신 MLP가 같은 순서로 쌓인다).
    for i in HIDDEN_LAYER_INDICES:
        assert actor[f"mlp.{i}.weight"] == f"actor.{i}.weight"
        assert critic[f"mlp.{i}.bias"] == f"critic.{i}.bias"
    assert actor["distribution.std_param"] == "std"
    # 정규화기가 서로 바뀌면 액터가 208차원 통계로 104차원을 정규화하게 된다.
    assert actor["obs_normalizer._mean"] == "a_mean"
    assert critic["obs_normalizer._mean"] == "c_mean"

    assert converted["iter"] == 1500
    assert "optimizer_state_dict" in converted


def test_log_std_parameterization():
    legacy = _legacy_checkpoint()
    del legacy["model_state_dict"]["std"]
    legacy["model_state_dict"]["log_std"] = "log_std"
    actor = convert_legacy_checkpoint(legacy)["actor_state_dict"]
    assert actor["distribution.log_std_param"] == "log_std"
    assert "distribution.std_param" not in actor


def test_symmetric_critic_run_without_privileged_normalizer():
    """v13/v17처럼 state_space=0인 런에도 크리틱 정규화기는 저장돼 있지만,
    없는 체크포인트를 만나도 조용히 틀린 값을 넣지 않고 그냥 비워 둔다."""
    converted = convert_legacy_checkpoint(_legacy_checkpoint(with_privileged=False))
    assert not any(k.startswith("obs_normalizer.") for k in converted["critic_state_dict"])


def test_unknown_key_is_not_silently_dropped():
    legacy = _legacy_checkpoint()
    legacy["model_state_dict"]["memory_a.rnn.weight_ih_l0"] = "rnn"
    try:
        convert_legacy_checkpoint(legacy)
    except KeyError as exc:
        assert "memory_a" in str(exc)
    else:
        raise AssertionError("모르는 키를 그냥 버렸습니다 (순환 정책 체크포인트 등)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
