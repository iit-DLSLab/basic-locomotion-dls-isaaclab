# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from copy import deepcopy
from . import amp_cfg
from . import morphosymm_cfg


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "flat_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP Related Stuff
    amp_data_path = amp_cfg.amp_data_path
    dataset_names = list(amp_cfg.dataset_names)
    dataset_weights = list(amp_cfg.dataset_weights)
    slow_down_factor = amp_cfg.slow_down_factor
    discriminator = deepcopy(amp_cfg.discriminator)

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg


@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 50
    experiment_name = "rough_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
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

     # AMP Related Stuff
    amp_data_path = amp_cfg.amp_data_path
    dataset_names = list(amp_cfg.dataset_names)
    dataset_weights = list(amp_cfg.dataset_weights)
    slow_down_factor = amp_cfg.slow_down_factor
    discriminator = deepcopy(amp_cfg.discriminator)

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg