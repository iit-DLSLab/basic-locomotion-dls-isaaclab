# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hyperparameter sweep definitions for the dedicated PPO tuner."""

from __future__ import annotations

from ray import tune


class PPOJobCfg:
    """Common PPO search space with independently selectable parameter groups."""

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        vary_env_count: bool = False,
        vary_mlp: bool = False,
        vary_network_type: bool = False,
        vary_algorithm: bool = False,
    ) -> None:
        cfg = {} if cfg is None else cfg
        cfg.setdefault("runner_args", {})
        cfg.setdefault("hydra_args", {})
        cfg["runner_args"]["--logger"] = "wandb"
        cfg["runner_args"]["--log_project_name"] = "basic-locomotion-ppo-tuning"
        cfg["hydra_args"]["agent.max_iterations"] = 2_000
        cfg["runner_args"]["--num_envs"] = 11264

        if vary_mlp:
            mlp_options = [
                [512, 256, 128],
                [256, 256, 256],
                [128, 128, 128],
            ]

            cfg["hydra_args"]["agent.policy.actor_hidden_dims"] = tune.choice(mlp_options)
            cfg["hydra_args"]["agent.policy.critic_hidden_dims"] = tune.choice(mlp_options)
            cfg["hydra_args"]["agent.policy.activation"] = tune.choice(["relu", "tanh", "sigmoid", "elu"])

        if vary_network_type:
            cfg["hydra_args"]["agent.policy.class_name"] = tune.choice(["ActorCriticRecurrent", "ActorCritic"])

        if vary_algorithm:
            cfg["hydra_args"]["agent.algorithm.clip_param"] = tune.choice([0.1, 0.15, 0.2, 0.25, 0.3])
            cfg["hydra_args"]["agent.algorithm.entropy_coef"] = tune.choice([0.005, 0.01, 0.015])
            cfg["hydra_args"]["agent.algorithm.num_learning_epochs"] = 5
            cfg["hydra_args"]["agent.algorithm.num_mini_batches"] = 4
            cfg["hydra_args"]["agent.algorithm.learning_rate"] = 1.0e-3
            cfg["hydra_args"]["agent.algorithm.gamma"] = tune.choice([0.97, 0.99, 0.999])
            cfg["hydra_args"]["agent.algorithm.lam"] = tune.choice([0.93, 0.95, 0.97])
            cfg["hydra_args"]["agent.algorithm.desired_kl"] = tune.choice([0.005, 0.01, 0.02])
            cfg["hydra_args"]["agent.algorithm.value_loss_coef"] = tune.choice([0.8, 1.0, 1.2])
            cfg["hydra_args"]["agent.num_steps_per_env"] = 24

        if "--task" not in cfg["runner_args"]:
            raise ValueError("No PPO task specified.")
        self.cfg = cfg


class LocomotionAliengoFlatPPOTuner(PPOJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Aliengo-Flat"}}
        super().__init__(cfg, vary_mlp=True, vary_algorithm=True, vary_network_type=True)


class LocomotionAliengoRoughBlindPPOTuner(PPOJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Aliengo-Rough-Blind"}}
        super().__init__(cfg, vary_mlp=True, vary_algorithm=True, vary_network_type=True)


class LocomotionAliengoRoughVisionPPOTuner(PPOJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Aliengo-Rough-Vision"}}
        super().__init__(cfg, vary_mlp=True, vary_algorithm=True, vary_network_type=True)


class LocomotionGo2FlatPPOTuner(PPOJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Go2-Flat"}}
        super().__init__(cfg, vary_mlp=True, vary_algorithm=True, vary_network_type=True)


class LocomotionGo2RoughBlindPPOTuner(PPOJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Go2-Rough-Blind"}}
        super().__init__(cfg, vary_mlp=True, vary_algorithm=True, vary_network_type=True)


class LocomotionGo2RoughVisionPPOTuner(PPOJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Go2-Rough-Vision"}}
        super().__init__(cfg, vary_mlp=True, vary_algorithm=True, vary_network_type=True)
