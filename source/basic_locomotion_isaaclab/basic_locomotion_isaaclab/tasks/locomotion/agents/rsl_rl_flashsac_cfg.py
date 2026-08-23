# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license),
# with modifications by Holiday Robotics (BSD-3-Clause license).

"""Runner/algorithm configuration dataclasses for FlashSAC on Isaac Lab.

There is a single unified :class:`FlashSACRunnerCfg` carrying the full FlashSAC benchmark
hyperparameter set; per-task configs subclass it and
override ``task_name`` only (registered at the bottom of this module).
"""

from __future__ import annotations

import re
from dataclasses import field

from isaaclab.utils import configclass


@configclass
class FlashSACActorCfg:
    """Configuration for the FlashSAC actor model."""

    class_name: str = "FlashSACActor"
    """The model class name (resolved from rsl_rl_flashsac.models)."""

    num_blocks: int = 2
    """The number of residual FlashSAC blocks in the trunk."""

    hidden_dim: int = 128
    """The hidden dimension of the trunk."""

    log_std_min: float = -10.0
    """Lower bound of the Tanh-normalized log standard deviation."""

    log_std_max: float = 2.0
    """Upper bound of the Tanh-normalized log standard deviation."""




@configclass
class FlashSACCriticCfg:
    """Configuration for the FlashSAC critic model."""

    class_name: str = "FlashSACCritic"
    """The model class name (resolved from rsl_rl_flashsac.models)."""

    num_blocks: int = 2
    """The number of residual FlashSAC blocks in the trunk."""

    hidden_dim: int = 256
    """The hidden dimension of the trunk."""

    num_bins: int = 101
    """The number of bins of the categorical value distribution."""

    min_v: float = -5.0
    """Minimum value of the categorical support. Keep at -normalized_G_max."""

    max_v: float = 5.0
    """Maximum value of the categorical support. Keep at +normalized_G_max."""


@configclass
class FlashSACAlgorithmCfg:
    """Configuration for the FlashSAC algorithm (official IsaacLab benchmark defaults)."""

    class_name: str = "FlashSAC"
    """The algorithm class name (resolved from rsl_rl_flashsac.algorithms)."""

    replay_buffer_size: int = 10_000_00
    """Maximum number of transitions in the replay buffer (total across all environments)."""

    buffer_min_length: int = 100_000
    """Minimum number of transitions in the replay buffer before updates start."""

    buffer_optimize_memory_usage: bool = True
    """Store observations once and reconstruct next observations by index."""

    buffer_device: str | None = "cpu"
    """Device for the replay buffer storage. None uses the training device."""

    buffer_obs_dtype: str | None = None
    """Optional torch dtype name for observation storage (e.g. "bfloat16")."""

    num_learning_epochs: int = 1
    """Number of gradient epochs per update step."""

    num_mini_batches: int = 2
    """Gradient updates per iteration (with num_steps_per_env=1: updates per env step)."""

    mini_batch_size: int = 2048
    """Mini-batch size drawn from the replay buffer."""

    learning_rate_init: float = 3.0e-4
    """Learning rate at the start of the warmup."""

    learning_rate_peak: float = 3.0e-4
    """Learning rate after warmup (optimizer base learning rate)."""

    learning_rate_end: float = 1.5e-4
    """Learning rate at the end of the cosine decay."""

    learning_rate_warmup_steps: int = 0
    """Number of update steps for the linear warmup."""

    learning_rate_decay_steps: int | None = None
    """Total schedule length in update steps. None resolves it from max_iterations."""

    actor_bc_alpha: float = 0.0
    """BC regularization coefficient. 0 disables BC regularization."""

    actor_noise_zeta_mu: float = 2.0
    """Zeta distribution exponent for exploration noise repeat lengths."""

    actor_noise_zeta_max: int = 16
    """Maximum exploration noise repeat length."""

    actor_update_period: int = 2
    """Actor/temperature update period relative to critic updates."""

    critic_target_update_tau: float = 0.01
    """EMA coefficient for the target critic."""

    temp_initial_value: float = 0.01
    """Initial temperature value."""

    temp_target_sigma: float = 0.15
    """Target per-dimension physical action std used to derive the target entropy (fixed in
    physical action space regardless of the action range)."""

    temp_target_entropy: float | None = None
    """Explicit target entropy. None derives it from temp_target_sigma."""

    gamma: float = 0.99
    """The discount factor."""

    n_steps: int = 3
    """Number of steps for n-step returns."""

    normalize_reward: bool = True
    """Whether to normalize rewards with the running return scale."""

    normalized_G_max: float = 5.0
    """Maximum magnitude of the normalized return (match the critic min_v/max_v support)."""

    use_compile: bool = True
    """Whether to torch.compile the network forward passes and update helpers."""

    compile_mode: str = "auto"
    """torch.compile mode. 'auto' picks autotuned kernels without CUDA graphs (see FlashSAC docs)."""

    use_amp: bool = True
    """Whether to use fp16 automatic mixed precision for actor/critic updates."""

    rnd_cfg: dict | None = None
    """Not supported by FlashSAC; must stay None."""

    symmetry_cfg: dict | None = None
    """Symmetry data augmentation config (``rsl_rl.extensions.Symmetry`` kwargs), or None to
    disable. ``use_mirror_loss=True`` is not supported. See ``G1DreamwaqSymmetryAlgorithmCfg``
    for the G1 DreamWaQ left-right mirror used by the tasks below."""




@configclass
class FlashSACRunnerCfg:
    """Unified runner configuration for FlashSAC training on Isaac Lab tasks.

    Per-task configs subclass this and set ``task_name`` only; ``experiment_name`` is derived
    automatically. Hyperparameters follow the official FlashSAC IsaacLab benchmark settings
    (~50M env steps with 1024 environments, two gradient updates per environment step).
    """

    task_name: str = ""
    """The Isaac Lab gym task id this configuration belongs to (set by per-task subclasses)."""

    class_name: str = "OffPolicyRunner"
    """The runner class name."""

    seed: int = 0
    """The seed for the experiment (0 = first seed of the official FlashSAC benchmark)."""

    device: str = "cuda:0"
    """The device for the rl-agent."""

    num_steps_per_env: int = 1
    """The number of environment steps per iteration."""

    max_iterations: int = 50_000
    """The maximum number of iterations. With num_steps_per_env=1 and 1024 environments this is
    ~51M env steps, matching the official FlashSAC benchmark budget (50,000,896 env steps)."""

    save_interval: int = -1
    """The number of iterations between checkpoint saves. Values <= 0 derive
    ``max_iterations // 10`` at runtime, so a run saves 10 checkpoints in total
    (9 periodic + the final model)."""

    log_interval: int = 20
    """The number of iterations between logging the training statistics."""

    start_training: int = 0
    """Extra update-free iterations; updates are already gated by algorithm.buffer_min_length."""

    experiment_name: str = ""
    """The experiment name. Empty derives ``flashsac_<task>`` from ``task_name``."""

    run_name: str = ""
    """The run name suffix."""

    logger: str | dict = field(
        default_factory=lambda: {"class_name": "WandbLogWriter", "project_name": "basic-locomotion"}
    )
    """The logging backend: "tensorboard" or a LogWriter dict, e.g.
    ``{"class_name": "WandbLogWriter", "project_name": "flashsac"}`` (entity via
    ``WANDB_USERNAME``). With W&B, ``flashsac-eval`` appends its metrics to the same run."""

    obs_groups: dict = field(default_factory=lambda: {"actor": ["policy"], "critic": ["policy"]})
    """Observation set mapping. Override in a task cfg when the env provides privileged
    critic observations (e.g. ``{"actor": ["policy"], "critic": ["policy", "critic"]}``)."""

    clip_actions: float | None = None
    """Optional action clipping applied inside the env wrapper."""

    action_bound: str = "scalar"
    """How the Tanh policy's affine action scaling is computed: "joint_limit" (zero bias with a
    symmetric per-joint range of the max distance from the default pose to the soft joint
    position limits) or "scalar" (±action_bound_scale bounds)."""

    action_bound_scale: float = 1.0
    """Half-width of the scalar action bounds (actions span ±action_bound_scale). Matches the
    official FlashSAC per-task ACTION_BOUNDS table (1.0 for most tasks, 3.0 for the Franka
    manipulation tasks). Also used as the fallback when joint-limit scaling is unavailable."""

    check_for_nan: bool = True
    """Whether to check environment outputs for NaN values."""

    actor: FlashSACActorCfg = field(default_factory=FlashSACActorCfg)
    """The actor model configuration."""

    critic: FlashSACCriticCfg = field(default_factory=FlashSACCriticCfg)
    """The critic model configuration."""

    algorithm: FlashSACAlgorithmCfg = field(default_factory=FlashSACAlgorithmCfg)
    """The algorithm configuration."""

    def __post_init__(self) -> None:
        """Derive the experiment name from the task name when not explicitly set."""
        if not self.experiment_name and self.task_name:
            slug = re.sub(r"-v\d+$", "", self.task_name)
            slug = re.sub(r"^Isaac-", "", slug).replace("-", "_").lower()
            self.experiment_name = f"flashsac_{slug}"
        # The run directory basename doubles as the W&B run name; include the experiment
        # slug so runs are distinguishable across tasks.
        if not self.run_name:
            self.run_name = self.experiment_name


###############################
# Task registry               #
###############################

TASK_CFG_REGISTRY: dict[str, type[FlashSACRunnerCfg]] = {}


def register_task(cfg_class: type[FlashSACRunnerCfg]) -> type[FlashSACRunnerCfg]:
    """Class decorator registering a runner cfg under its task id and the -Play variant."""
    # Read via an instance: isaaclab's configclass does not keep plain class attributes.
    task_name = cfg_class().task_name
    if not task_name:
        raise ValueError(f"{cfg_class.__name__} must set 'task_name'.")
    TASK_CFG_REGISTRY[task_name] = cfg_class
    TASK_CFG_REGISTRY[re.sub(r"-v(\d+)$", r"-Play-v\1", task_name)] = cfg_class
    return cfg_class


def get_task_cfg(task_name: str) -> FlashSACRunnerCfg:
    """Instantiate the registered runner configuration for the given task id."""
    if task_name not in TASK_CFG_REGISTRY:
        available = sorted(TASK_CFG_REGISTRY)
        raise KeyError(f"No FlashSAC config registered for task '{task_name}'. Available: {available}")
    return TASK_CFG_REGISTRY[task_name]()


###############################
# Velocity locomotion tasks   #
###############################
# FlashSAC uses a single hyperparameter set across tasks, so every config below is the
# unified FlashSACRunnerCfg with only ``task_name`` overridden.


@register_task
@configclass
class LocomotionGo2FlatFlashSACCfg(FlashSACRunnerCfg):
    task_name: str = "Locomotion-Go2-Flat"
    action_bound_scale: float = 3.0

@register_task
@configclass
class LocomotionGo2RoughBlindFlashSACCfg(FlashSACRunnerCfg):
    task_name: str = "Locomotion-Go2-Rough-Blind"
    action_bound_scale: float = 3.0

@register_task
@configclass
class LocomotionGo2RoughVisionFlashSACCfg(FlashSACRunnerCfg):
    task_name: str = "Locomotion-Go2-Rough-Vision"
    action_bound_scale: float = 3.0
    
