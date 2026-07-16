"""rsl_rl PPO runner config for the Joystick task.

Uses the current (non-deprecated) rsl_rl >= 4.0 API from this repo's
targeted IsaacLab checkout (`isaaclab_rl.rsl_rl.rl_cfg`): `actor`/`critic`
as `RslRlMLPModelCfg` + `obs_groups`, rather than the older deprecated
`policy: RslRlPpoActorCriticCfg` field the abandoned WIP used (that field
still exists but is marked deprecated in this checkout's rl_cfg.py, and
`actor`/`critic`/`obs_groups`/`empirical_normalization` are MISSING/required
regardless of which path is used — the modern path was verified directly
against source rather than assumed from the WIP's possibly-stale example).

Network sizing (GPU-memory-conscious: modest hidden layers, small
mini-batch count) carried over in spirit from the WIP's own
`agents/rsl_rl_ppo_cfg.py`, whose comments note an RTX-class /
VRAM-constrained target GPU — this repo's actual training GPU is still
unknown, so kept conservative.

TODO before Stage 4's Ubuntu gate: cross-check num_timesteps / network
sizes against Playground's own
`mujoco_playground.config.locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`
if that package is available — not found in any locally-read file during
planning, so the values below are reasonable PPO defaults for a
14-action-dim biped, not a verified match to Playground's exact recipe.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# No asymmetric critic in this v1 port (see joystick_env_cfg.py's
# docstring) — actor and critic both consume the single "policy" obs group.
_OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


@configclass
class JoystickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "open_duck_mini_v2_joystick"
    empirical_normalization = False  # deprecated flag; obs_normalization below is the modern equivalent
    obs_groups = _OBS_GROUPS

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
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
