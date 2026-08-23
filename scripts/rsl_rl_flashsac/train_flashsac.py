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

"""Train FlashSAC on an Isaac Lab task.

Usage (from an environment with Isaac Lab installed):
    flashsac-train --task Isaac-Velocity-Rough-G1-v0 --num_envs 1024 --headless
"""

from __future__ import annotations

import argparse
import ast
import os
from datetime import datetime


def _parse_agent_override_value(raw_value: str):
    """Parse a command-line agent override while preserving unquoted strings."""
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def apply_agent_cfg_overrides(agent_cfg, overrides: list[str]) -> None:
    """Apply dotted ``agent.<path>=<value>`` overrides to an agent config.

    This lets Ray Tune vary nested actor, critic, runner, and algorithm settings.
    Unknown paths are rejected so a misspelled sweep parameter cannot be ignored.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid agent override '{override}': expected agent.<path>=<value>.")

        key, raw_value = override.split("=", maxsplit=1)
        if not key.startswith("agent."):
            raise ValueError(f"Invalid agent override '{override}': keys must start with 'agent.'.")

        path = key.removeprefix("agent.").split(".")
        if not all(path):
            raise ValueError(f"Invalid agent override '{override}': the configuration path is empty.")

        target = agent_cfg
        for part in path[:-1]:
            if isinstance(target, dict):
                if part not in target:
                    raise ValueError(f"Unknown agent configuration path '{key}'.")
                target = target[part]
            else:
                if not hasattr(target, part):
                    raise ValueError(f"Unknown agent configuration path '{key}'.")
                target = getattr(target, part)

        leaf = path[-1]
        value = _parse_agent_override_value(raw_value)
        if isinstance(target, dict):
            if leaf not in target:
                raise ValueError(f"Unknown agent configuration path '{key}'.")
            target[leaf] = value
        else:
            if not hasattr(target, leaf):
                raise ValueError(f"Unknown agent configuration path '{key}'.")
            setattr(target, leaf, value)


def main() -> None:
    """Launch Isaac Sim and train FlashSAC on the requested task."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train FlashSAC on an Isaac Lab task.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task (Isaac Lab gym id).")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override for the run.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override the number of iterations.")
    parser.add_argument("--run_name", type=str, default=None, help="Suffix appended to the run name.")
    parser.add_argument(
        "--action_bound", type=str, default=None, choices=["joint_limit", "scalar"], help="Action bound override."
    )
    parser.add_argument(
        "--action_bound_scale", type=float, default=None, help="Half-width of the scalar action bounds override."
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="Enable left-right symmetry data augmentation (tracking tasks only; DreamWaQ tasks have it built in).",
    )
    parser.add_argument(
        "--mini_batch_size", type=int, default=None, help="Replay mini-batch size override (algorithm.mini_batch_size)."
    )
    parser.add_argument(
        "--motion_files",
        type=str,
        nargs="+",
        default=None,
        help="Motion clip .npz path(s) or a directory of clips (required for tracking tasks).",
    )
    # AppLauncher consumes this to bind each rank to cuda:{LOCAL_RANK} (launch via torchrun)
    parser.add_argument("--distributed", action="store_true", help="Run distributed data-parallel training.")
    parser.add_argument(
        "agent_overrides",
        nargs="*",
        metavar="agent.<path>=<value>",
        help="Nested FlashSAC agent overrides used by hyperparameter sweeps.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Late imports: Isaac Lab modules require the simulation app to be running
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the Isaac Lab gym tasks)
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_flashsac.envs  # noqa: F401  (registers the out-of-tree G1 DreamWaQ velocity + tracking tasks)
    from isaaclab_flashsac.envs.g1_wbt import apply_motion_files
    #from isaaclab_flashsac.rl_cfg import G1_TRACKING_SYMMETRY_CFG, get_task_cfg
    from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
    from rsl_rl_flashsac.runners import OffPolicyRunner

    from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_flashsac_cfg import get_task_cfg

    # Resolve the agent configuration and apply CLI overrides
    agent_cfg = get_task_cfg(args_cli.task)
    try:
        apply_agent_cfg_overrides(agent_cfg, args_cli.agent_overrides)
    except ValueError as exc:
        parser.error(str(exc))
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if getattr(args_cli, "device", None) is not None:
        agent_cfg.device = args_cli.device
    if args_cli.run_name is not None:
        agent_cfg.run_name = f"{agent_cfg.run_name}_{args_cli.run_name}" if agent_cfg.run_name else args_cli.run_name
    if args_cli.action_bound is not None:
        agent_cfg.action_bound = args_cli.action_bound
    if args_cli.action_bound_scale is not None:
        agent_cfg.action_bound_scale = args_cli.action_bound_scale
    if args_cli.symmetry:
        if "Tracking" not in args_cli.task:
            parser.error(f"--symmetry is only supported for tracking tasks, got task '{args_cli.task}'.")
        agent_cfg.algorithm.symmetry_cfg = dict(G1_TRACKING_SYMMETRY_CFG)
    if args_cli.mini_batch_size is not None:
        agent_cfg.algorithm.mini_batch_size = args_cli.mini_batch_size

    # Multi-GPU (torchrun) launch: AppLauncher resolved the sim device to cuda:{local_rank};
    # mirror it for the agent and offset the seed per rank to decorrelate rollouts (the
    # runner broadcasts rank-0 model parameters at learn start, so init seeds may differ).
    if args_cli.distributed:
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.seed += app_launcher.global_rank

    # Seeding and matmul settings (matching the official FlashSAC training setup)
    torch.manual_seed(agent_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(agent_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Build the environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=agent_cfg.device, num_envs=args_cli.num_envs)
    env_cfg.seed = agent_cfg.seed
    apply_motion_files(env_cfg, args_cli.motion_files)

    # Logging directory and config dumps (reproducibility) — rank 0 only in distributed
    # runs; the rsl_rl Logger disables all writers and checkpoint saves on other ranks.
    log_dir = None
    if not args_cli.distributed or app_launcher.global_rank == 0:
        # Include microseconds because Ray can launch multiple trials in the same second.
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        if agent_cfg.run_name:
            run_stamp += f"_{agent_cfg.run_name}"
        # Seed in the dir name: same-second launches of one task must not share a dir
        run_stamp += f"_s{agent_cfg.seed}"
        log_root_path = os.path.abspath(os.path.join("logs", "flashsac", agent_cfg.experiment_name))
        log_dir = os.path.join(log_root_path, run_stamp)
        # These two lines form the log-discovery interface used by the Ray tuner.
        print(f"[INFO] Logging experiment in directory: {log_root_path}", flush=True)
        print(f"Exact experiment name requested from command line: {run_stamp}", flush=True)
        os.makedirs(log_dir, exist_ok=False)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Build the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = FlashSACVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        action_bound=agent_cfg.action_bound,
        action_bound_scale=agent_cfg.action_bound_scale,
    )

    # Train
    train_cfg = agent_cfg.to_dict()  # type: ignore[attr-defined]
    runner = OffPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    # Randomize initial episode lengths to decorrelate resets across environments
    # (training only — eval/play start from fresh resets), as in the official FlashSAC.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
