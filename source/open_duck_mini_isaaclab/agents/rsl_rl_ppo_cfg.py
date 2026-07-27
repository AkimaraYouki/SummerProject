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
