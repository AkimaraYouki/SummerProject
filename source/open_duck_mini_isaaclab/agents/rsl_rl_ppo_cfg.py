"""rsl_rl PPO runner config for the Joystick task.

Uses `isaaclab_rl.rsl_rl.rl_cfg`'s ACTUAL API as installed on the lab PC
(isaaclab_rl==0.2.0, confirmed 2026-07-25 by reading rl_cfg.py directly and
by a real smoke-test crash): a single `policy: RslRlPpoActorCriticCfg` field
with `actor_hidden_dims`/`critic_hidden_dims`, not the `actor`/`critic`:
`RslRlMLPModelCfg` + `obs_groups` shape a previous pass at this file assumed
("the modern rsl_rl >= 4.0 API") — that assumption was wrong for this
checkout: `RslRlMLPModelCfg` doesn't exist in this version of
isaaclab_rl.rsl_rl at all, and the smoke test failed with
`ImportError: cannot import name 'RslRlMLPModelCfg'` until this was fixed.
`RslRlPpoActorCriticCfg` is not "deprecated" here — it's the only shape this
version has. Re-verify against source (`cat` the actual installed rl_cfg.py)
if this ever moves to a different IsaacLab checkout, rather than assuming
either API shape.

`empirical_normalization = True` here (opposite of the previous pass's
`False`) because this API version has no per-model `obs_normalization` knob
to fall back on — `empirical_normalization` is this checkout's only
observation-normalization switch, so it must be on to get any normalization
at all.

Network sizing (GPU-memory-conscious: modest hidden layers, small
mini-batch count) carried over in spirit from the abandoned WIP's own
`agents/rsl_rl_ppo_cfg.py`, whose comments note an RTX-class /
VRAM-constrained target GPU — this repo's actual training GPU (RTX 5080,
16GB, confirmed on the lab PC) has more headroom than that WIP assumed, so
these sizes are conservative rather than tight; fine as a smoke-test/first-
pass config, revisit once real training starts.

TODO before trusting real training results: cross-check num_timesteps /
network sizes against Playground's own
`mujoco_playground.config.locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`
if that package is available — not verified against Playground's exact
recipe, just reasonable PPO defaults for a 14-action-dim biped.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class JoystickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "open_duck_mini_v2_joystick"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Pre-approved fallback (imitation_v6, paired with
# joystick_env_cfg.py::JoystickEnvCfg_A30J25Im2) if imitation_v5 (A30J25)
# also fails to escape the observed local-minimum ("joint locks near a
# limit angle then pops/flings outward") behavior. Doubles the initial
# action-noise std to widen early exploration.
@configclass
class JoystickPPORunnerCfg_Gamma097(JoystickPPORunnerCfg):
    """imitation_v24 이후의 표준 설정. 비대칭 크리틱과 함께 쓴다.

    2026-07-29 정리. v20~v24를 거치며 러너 클래스가 7개까지 늘었는데
    (N2 / Upstream / BigNet / BigNetLowEnt / BigNetMB16 / Gamma097) 대부분은
    실패로 판정된 경로였다. 상속도 3단계라 실제 값을 알려면 사슬을 타야 했다.
    여기서는 값을 직접 적고, 지운 것들의 결론만 남긴다:

      N2          init_noise_std 2.0 — v6 시절 탐색 확대 시도, 이후 미사용
      Upstream    brax 하이퍼파라미터 그대로 이식(minibatch 32, entropy 0.005,
                  epochs 4, gamma 0.97, steps 20). num_minibatches 32는 rsl_rl
                  에서 iteration당 128회 업데이트를 뜻해(v17은 20회) KL이 초과했고
                  adaptive 스케줄이 lr을 최저 한계 1e-5까지 깎아 iter 150부터
                  학습이 동결됐다. brax는 lr 고정에 KL 적응이 없다 —
                  **하이퍼파라미터는 PPO 구현 간에 이식되지 않는다.**
      BigNet      네트워크만 키우고 PPO는 v17 값. entropy 0.01이 노이즈 std를
                  0.75로 밀어올려(액션 잡음 약 10°, 관절 추종 오차 7~8°보다 큼)
                  iter 100부터 0.145에 정체.
      BigNetLowEnt entropy 0.005로 되돌려 0.222까지 회복. Gamma097의 직전 단계.
      BigNetMB16  minibatch 16. 4→16으로 네 배 올렸는데 곡선이 BigNetLowEnt와
                  완전히 겹쳤다(iter 160에서 둘 다 0.192) — minibatch는 이
                  태스크에서 무관하며, 앞서 이를 원인으로 지목했던 것은 오귀속이었다.

    여기서 base 대비 실제로 다른 것은 두 가지뿐이다:
      네트워크  (256,128,64) -> (512,256,128)
      gamma     0.99 -> 0.97
    gamma 0.97의 유효 시간지평은 약 33스텝으로 이 로봇의 보행 주기 27스텝과
    거의 같다. 0.99는 100스텝, 세 주기 반이다. 주기적 과제에서 크리틱이
    추정해야 할 범위를 한 주기로 좁힌 것이 +53%를 만들었다.
    """

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
