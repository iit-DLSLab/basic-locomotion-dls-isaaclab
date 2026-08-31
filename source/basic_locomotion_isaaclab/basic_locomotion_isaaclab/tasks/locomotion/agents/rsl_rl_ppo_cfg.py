# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from copy import deepcopy
from . import amp_cfg
from . import morphosymm_cfg


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "flat_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}  # rsl_rl 5.4 port
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
    dataset = deepcopy(amp_cfg.dataset)
    discriminator = deepcopy(amp_cfg.discriminator)

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg


@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 50
    experiment_name = "rough_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}  # rsl_rl 5.4 port
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.97,
        lam=0.99,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # AMP Related Stuff
    dataset = deepcopy(amp_cfg.dataset)
    discriminator = deepcopy(amp_cfg.discriminator)

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg
