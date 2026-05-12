# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from pathlib import Path
from dataclasses import MISSING

@configclass
class DiscriminatorCfg:
    """Configuration for the discriminator network."""

    class_name: str = "Discriminator"
    """The discriminator class name. Default is Discriminator."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""

    reward_scale: float = MISSING
    """The reward coefficient."""

    loss_type: str = "BCEWithLogits"
    """The type of loss to use for training the discriminator. Default is BCEWithLogits."""
    
    empirical_normalization: bool = False
    """Whether to use empirical normalization for the discriminator inputs. Default is False."""


@configclass
class MorphologycalSymmetriesCfg:
    """Configuration for using morphosymm-rl."""

    class_name: str = "MorphologycalSymmetries"
    """The class name."""

    obs_space_names_actor =  None
    """The observation space names for the actor network."""

    obs_space_names_critic = None
    """The observation space names for the critic network."""

    action_space_names = None
    """The action space names."""

    joints_order = None
    """The order of the joints in the robot."""

    robot_name = None
    """The name of the robot to use inside Morphosymm."""


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
        schedule="fixed", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    #AMP Related Stuff
    amp_data_path = "./../../../../../../scripts/amp_rl/amp_dataset/"
    dataset_names = ["flat"]
    dataset_weights = [1.0, 1.0]
    slow_down_factor = 1.0
    discriminator = DiscriminatorCfg(
        hidden_dims=[128, 128],
        reward_scale=0.1,
        loss_type="BCEWithLogits",
        empirical_normalization= False
    )

    # Symmetry Related Stuff - Actor Critic
    history_length = 5
    obs_space_names_actor = [
            "base_lin_vel:base",
            "base_ang_vel:base",
            "gravity:base",
            "ctrl_commands",
            "default_qpos_js_error",
            "qvel_js",
            "actions",
            "clock_data",
        ]*int(history_length)
    obs_space_names_critic = obs_space_names_actor
    
    # Symmetry Related Stuff -  Asymmetric Critic
    """obs_space_names_critic += ["position_gains", 
            "velocity_gains",
            "friction_static",
            "friction_dynamic",
            "armature"
        ]"""

    morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        action_space_names = ["actions"],
        joints_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ],
        robot_name = "a1",
    )


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
        #actor_hidden_dims=[512, 256, 128],
        #critic_hidden_dims=[512, 256, 128],
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

    #AMP Related Stuff
    amp_data_path = "./../../../../../../scripts/amp_rl/amp_dataset/"
    dataset_names = ["flat", "boxes", "stairs"]
    dataset_weights = [1.0, 1.0, 1.0, 1.0]
    slow_down_factor = 1.0
    discriminator = DiscriminatorCfg(
        hidden_dims=[1024, 512],
        reward_scale=1.0,
        loss_type="BCEWithLogits",
        empirical_normalization= False
    )

    # Symmetry Related Stuff - Actor Critic
    history_length = 5
    obs_space_names_actor = [
            "base_lin_vel:base",
            "base_ang_vel:base",
            "gravity:base",
            "ctrl_commands",
            "default_qpos_js_error",
            "qvel_js",
            "actions",
            "clock_data",
        ]*int(history_length)

    #obs_space_names_actor += ["heightmap:rows4xcols4"]
    #obs_space_names_actor += ["position_gains"]
    #obs_space_names_actor += ["velocity_gains"]


    # Symmetry Related Stuff -  Asymmetric Critic
    obs_space_names_critic = [
            "base_lin_vel:base",
            "base_ang_vel:base",
            "gravity:base",
            "ctrl_commands",
            "default_qpos_js_error",
            "qvel_js",
            "actions",
            "clock_data",
        ]*int(history_length)
    obs_space_names_critic += ["base_pos_z", "base_pos_z"]

    morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        action_space_names = ["actions"],
        joints_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ],
        robot_name = "a1",
    )