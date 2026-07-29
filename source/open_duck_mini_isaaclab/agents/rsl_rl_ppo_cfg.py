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
class JoystickPPORunnerCfg_N2(JoystickPPORunnerCfg):
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=2.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class JoystickPPORunnerCfg_Upstream(JoystickPPORunnerCfg):
    """PPO hyperparameters aligned to what upstream actually runs.

    Upstream calls `locomotion_params.brax_ppo_config(
    "BerkeleyHumanoidJoystickFlatTerrain")` -- with a literal `# TODO` -- so
    these are Berkeley Humanoid's numbers, borrowed untuned. Recorded here so a
    later reader does not mistake them for values validated on this robot.

      network         (512, 256, 128)   was (256, 128, 64)
      learning_rate   3e-4              was 1e-3
      entropy_coef    0.005             was 0.01
      num_mini_batches 32               was 4
      gamma           0.97              was 0.99
      num_updates     4                 was 5   (upstream num_updates_per_batch)
      rollout         20                was 24  (upstream unroll_length)
    """

    num_steps_per_env = 20

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_mini_batches = 32
        self.algorithm.num_learning_epochs = 4
        self.algorithm.gamma = 0.97


@configclass
class JoystickPPORunnerCfg_BigNet(JoystickPPORunnerCfg):
    """imitation_v21 — 비대칭 크리틱의 이득만 남기고 나머지는 v17로 복귀.

    v20에서 upstream 하이퍼파라미터를 통째로 옮긴 결과를 분리해보면:

      효과 있었음  네트워크 (512,256,128) + 비대칭 크리틱
                   → 리워드 동일 조건에서 스텝당 3배, 에피소드 2.6배
      역효과       num_mini_batches 32 (= 32x4 = iteration당 128 업데이트,
                   v17은 4x5 = 20). 정책이 6.4배 크게 움직여 KL이 초과하고,
                   rsl_rl의 adaptive 스케줄이 lr을 최저 한계 1e-5까지 깎아
                   iter 200부터 학습이 사실상 동결됐다.
      역효과       entropy_coef 0.005 → mean_noise_std가 0.99에서 0.27로 붕괴
                   (v17은 0.8 유지). action_scale 0.25를 곱하면 탐색 폭이
                   3.8도밖에 안 된다. 게다가 가우시안 KL은 sigma^2에 반비례해
                   std가 줄수록 KL이 커지는 되먹임까지 걸린다.

    brax의 num_minibatches는 데이터 분할 정의가 rsl_rl과 다르고 brax는 lr
    고정에 KL 적응이 없다 — 숫자를 그대로 옮기면 의미가 달라진다는 게
    이 런의 교훈이다.

    그래서 여기서는 네트워크 크기만 유지하고 PPO 파라미터는 전부 상속받은
    기본값(v17과 동일)을 쓴다. __post_init__ 대신 policy를 통째로 다시 적어
    클래스 본문만 봐도 값이 보이게 했다.
    """

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class JoystickPPORunnerCfg_BigNetLowEnt(JoystickPPORunnerCfg_BigNet):
    """imitation_v22 — v20b와 v21이 각각 반쪽씩 맞았던 것을 합친다.

    v20b는 upstream 파라미터 두 개를 한꺼번에 바꿔서 좋은 쪽과 나쁜 쪽이
    섞여 있었고, v21에서 둘 다 되돌렸더니 이번엔 반대로 나빠졌다:

      v20b  minibatch 32, entropy 0.005  →  스텝당 0.345, 그러나 lr이
            iter 150에 1e-5 바닥에 붙어 학습이 동결
      v21   minibatch  4, entropy 0.01   →  lr은 정상 범위(2.3e-3~7.6e-5)에서
            진동하지만 스텝당이 iter 180에 0.141로 v20b의 절반

    v21이 뒤처진 이유는 탐색 잡음 자체다. std 0.69 x action_scale 0.25 =
    관절당 약 10도의 액션 잡음인데, 이 정책의 관절 추종 오차가 7~8도
    수준이라 잡음이 신호보다 크다. 정밀한 자세 추종이 곧 리워드인 과제에서는
    넓은 탐색이 그대로 손해가 된다.

    그래서 minibatch만 4로 되돌려 KL 초과와 lr 붕괴를 막고, entropy는 v20b의
    0.005를 유지해 탐색 잡음을 낮게 둔다.
    """

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
