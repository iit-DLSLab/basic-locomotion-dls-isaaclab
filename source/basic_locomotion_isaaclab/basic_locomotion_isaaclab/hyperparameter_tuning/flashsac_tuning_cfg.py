# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hyperparameter sweep definitions for the dedicated FlashSAC tuner."""

from __future__ import annotations

from ray import tune


class FlashSACJobCfg:
    """Common FlashSAC search space with independently selectable groups."""

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        vary_env_count: bool = True,
        vary_network: bool = True,
        vary_algorithm: bool = True,
    ):
        cfg = {} if cfg is None else cfg
        cfg.setdefault("runner_args", {})
        cfg.setdefault("agent_args", {})
        cfg["runner_args"]["headless_singleton"] = "--headless"

        # WandbLogWriter mirrors metrics to local TensorBoard event files for Ray.
        cfg["agent_args"]["agent.logger"] = {
            "class_name": "WandbLogWriter",
            "project_name": "basic-locomotion-flashsac-tuning",
        }
        cfg["agent_args"]["agent.max_iterations"] = 10_000
        cfg["agent_args"]["agent.log_interval"] = 20
        # Keep only the runner's final checkpoint during short-lived tuning trials.
        cfg["agent_args"]["agent.save_interval"] = 10_000

        if vary_env_count:
            cfg["runner_args"]["--num_envs"] = tune.choice([2048])

        if vary_network:
            cfg["agent_args"]["agent.actor.num_blocks"] = tune.choice([1, 2, 3])
            cfg["agent_args"]["agent.actor.hidden_dim"] = tune.choice([128, 256, 512])
            cfg["agent_args"]["agent.critic.num_blocks"] = tune.choice([1, 2, 3])
            cfg["agent_args"]["agent.critic.hidden_dim"] = tune.choice([128, 256, 512])

        if vary_algorithm:
            cfg["agent_args"]["agent.algorithm.mini_batch_size"] = tune.choice([1024, 2048, 4096])
            cfg["agent_args"]["agent.algorithm.num_mini_batches"] = tune.choice([1, 2, 4])
            cfg["agent_args"]["agent.algorithm.learning_rate_peak"] = tune.choice([1.5e-4, 3.0e-4, 6.0e-4])
            cfg["agent_args"]["agent.algorithm.actor_update_period"] = tune.choice([1, 2, 4])
            cfg["agent_args"]["agent.algorithm.critic_target_update_tau"] = tune.choice([0.005, 0.01, 0.02])
            cfg["agent_args"]["agent.algorithm.temp_initial_value"] = tune.choice([0.001, 0.01, 0.1])
            cfg["agent_args"]["agent.algorithm.temp_target_sigma"] = tune.choice([0.1, 0.15, 0.2])
            cfg["agent_args"]["agent.algorithm.gamma"] = tune.choice([0.97, 0.99, 0.995])
            cfg["agent_args"]["agent.algorithm.n_steps"] = tune.choice([1, 3, 5])

        if "--task" not in cfg["runner_args"]:
            raise ValueError("No FlashSAC task specified.")
        self.cfg = cfg


class LocomotionGo2FlatFlashSACTuner(FlashSACJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Go2-Flat"}}
        super().__init__(cfg)


class LocomotionGo2RoughBlindFlashSACTuner(FlashSACJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Go2-Rough-Blind"}}
        super().__init__(cfg)


class LocomotionGo2RoughVisionFlashSACTuner(FlashSACJobCfg):
    def __init__(self):
        cfg = {"runner_args": {"--task": "Locomotion-Go2-Rough-Vision"}}
        super().__init__(cfg)
