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

"""Evaluate trained FlashSAC policies on an Isaac Lab task.

Follows the official FlashSAC evaluation protocol: episodes are evaluated in rounds of
``num_envs`` parallel environments; within a round every environment runs until its FIRST
episode end (terminated or truncated) and is masked afterwards, so each round contributes
exactly ``num_envs`` full episodes.

Usage (from an environment with Isaac Lab installed):
    flashsac-eval --task Isaac-Velocity-Rough-G1-v0   # 1024 envs x 1024 episodes, as in the paper
    flashsac-eval --task Isaac-Velocity-Rough-G1-v0 --all_checkpoints   # eval curve over a run
"""

from __future__ import annotations

import argparse
import json
import os
import re


def _checkpoint_iteration(checkpoint_path: str) -> int:
    """Parse the training iteration from a ``model_<it>.pt`` checkpoint filename."""
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else -1


def log_eval_to_wandb(project_name: str, run_dir: str, eval_infos: list[dict]) -> None:
    """Append eval metrics to the W&B run of the training that produced the checkpoints.

    The training run is located by name: the upstream rsl_rl ``WandbLogWriter`` names its W&B run
    after the log directory basename (= ``run_dir``). Eval metrics use their own step axis
    (``Eval/checkpoint_iter``) so logging works even after the training run has finished (W&B
    rejects steps below the run's last global step).

    .. note::
        Resuming a run that a live training process is still writing to is not officially
        supported by W&B; prefer evaluating after training finishes (or accept interleaved logs).
    """
    try:
        import wandb
    except ModuleNotFoundError:
        print("[WARN] wandb is not installed; skipping W&B logging.")
        return

    run_name = os.path.basename(os.path.abspath(run_dir))
    entity = os.environ.get("WANDB_USERNAME")
    try:
        api = wandb.Api()
        entity = entity or api.default_entity
        runs = list(api.runs(f"{entity}/{project_name}", filters={"display_name": run_name}))
    except Exception as err:  # noqa: BLE001 — eval results must survive logging failures
        print(f"[WARN] W&B lookup failed ({err}); skipping W&B logging.")
        return
    if len(runs) != 1:
        print(
            f"[WARN] Found {len(runs)} W&B runs named '{run_name}' in project '{project_name}'; skipping W&B logging."
        )
        return

    print(f"[INFO] Logging {len(eval_infos)} eval point(s) to W&B run '{run_name}' (id: {runs[0].id}).")
    run = wandb.init(
        project=project_name,
        entity=entity,
        id=runs[0].id,
        resume="allow",
        settings=wandb.Settings(start_method="thread"),
    )
    run.define_metric("Eval/checkpoint_iter")
    run.define_metric("Eval/*", step_metric="Eval/checkpoint_iter")
    for eval_info in eval_infos:
        payload = {f"Eval/{key}": value for key, value in eval_info.items() if isinstance(value, (int, float))}
        payload["Eval/checkpoint_iter"] = eval_info["iteration"]
        run.log(payload)
    run.finish()


def main() -> None:
    """Launch Isaac Sim and evaluate trained FlashSAC checkpoints."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate trained FlashSAC policies on an Isaac Lab task.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task (Isaac Lab gym id).")
    parser.add_argument(
        "--num_envs", type=int, default=1024, help="Number of parallel environments (official benchmark: 1024)."
    )
    parser.add_argument("--num_episodes", type=int, default=None, help="Episodes per checkpoint (default: num_envs).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Default: latest run.")
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="Evaluate every model_*.pt in the run directory (eval curve) instead of a single checkpoint.",
    )
    # A checkpoint trained with a train.py --action_bound override MUST be evaluated with the
    # same bound - the policy's normalized [-1, 1] actions only mean what the training-time
    # wrapper scaling made them mean.
    parser.add_argument(
        "--action_bound", type=str, default=None, choices=["joint_limit", "scalar"], help="Action bound override."
    )
    parser.add_argument("--action_bound_scale", type=float, default=None, help="Scalar action bound half-width.")
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate checkpoints even when their eval JSON already exists."
    )
    parser.add_argument(
        "--motion_files",
        type=str,
        nargs="+",
        default=None,
        help="Motion clip .npz path(s) or a directory of clips (required for tracking tasks).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Late imports: Isaac Lab modules require the simulation app to be running
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the Isaac Lab gym tasks)
    import torch
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import isaaclab_flashsac.envs  # noqa: F401  (registers the out-of-tree G1 DreamWaQ velocity + tracking tasks)
    from isaaclab_flashsac.envs.g1_wbt import apply_motion_files
    from isaaclab_flashsac.rl_cfg import get_task_cfg
    from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
    from rsl_rl_flashsac.runners import OffPolicyRunner

    agent_cfg = get_task_cfg(args_cli.task)
    if getattr(args_cli, "device", None) is not None:
        agent_cfg.device = args_cli.device
    if args_cli.action_bound is not None:
        agent_cfg.action_bound = args_cli.action_bound
    if args_cli.action_bound_scale is not None:
        agent_cfg.action_bound_scale = args_cli.action_bound_scale
    device = agent_cfg.device

    num_episodes = args_cli.num_episodes if args_cli.num_episodes is not None else args_cli.num_envs
    if num_episodes % args_cli.num_envs != 0:
        raise ValueError(f"num_episodes ({num_episodes}) must be divisible by num_envs ({args_cli.num_envs}).")
    num_rounds = num_episodes // args_cli.num_envs

    # Resolve the checkpoint(s)
    if args_cli.checkpoint is not None:
        anchor_checkpoint = args_cli.checkpoint
    else:
        log_root = os.path.abspath(os.path.join("logs", "flashsac", agent_cfg.experiment_name))
        anchor_checkpoint = get_checkpoint_path(log_root, run_dir=".*", checkpoint="model_.*.pt")
    run_dir = os.path.dirname(os.path.abspath(anchor_checkpoint))
    if args_cli.all_checkpoints:
        import glob

        checkpoints = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=_checkpoint_iteration)
        if not checkpoints:
            raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")
    else:
        checkpoints = [anchor_checkpoint]
    print(f"[INFO] Evaluating {len(checkpoints)} checkpoint(s) from: {run_dir}")

    # Build the environment once for all checkpoints
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    apply_motion_files(env_cfg, args_cli.motion_files)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = FlashSACVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        action_bound=agent_cfg.action_bound,
        action_bound_scale=agent_cfg.action_bound_scale,
    )

    train_cfg = agent_cfg.to_dict()  # type: ignore[attr-defined]
    runner = OffPolicyRunner(env, train_cfg, log_dir=None, device=device)

    # Anchor for the file-mtime wall-time fallback (checkpoints from before wall_time was stored):
    # the config dump happens right before the runner/training loop starts.
    anchor = os.path.join(run_dir, "params", "agent.yaml")
    mtime_anchor = os.path.getmtime(anchor) if os.path.exists(anchor) else None

    eval_infos: list[dict] = []
    for checkpoint_path in checkpoints:
        # Idempotent in --all_checkpoints mode: skip checkpoints another eval already covered
        ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        out_path = os.path.join(run_dir, f"eval_{ckpt_name}.json")
        if args_cli.all_checkpoints and not args_cli.force and os.path.exists(out_path):
            print(f"[INFO] {ckpt_name}: eval JSON exists, skipping.")
            continue
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=device)
        wall_time = runner.loaded_wall_time
        if wall_time is None and mtime_anchor is not None:
            wall_time = os.path.getmtime(checkpoint_path) - mtime_anchor

        total_returns: list[float] = []
        total_lengths: list[float] = []
        with torch.no_grad():
            for _ in range(num_rounds):
                obs, _ = env.reset()
                obs = obs.to(device)
                returns = torch.zeros(args_cli.num_envs, device=device)
                lengths = torch.zeros(args_cli.num_envs, device=device)
                finished = torch.zeros(args_cli.num_envs, device=device)

                # Every environment runs until its first episode end, then is masked out
                while finished.sum() < args_cli.num_envs:
                    actions = policy(obs)
                    obs, rewards, dones, _ = env.step(actions.to(env.device))
                    obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)

                    active = 1.0 - finished
                    returns += rewards * active
                    lengths += active
                    # terminated or truncated both end the episode during evaluation
                    finished = torch.maximum(finished, dones.float().reshape(-1))

                total_returns.extend(returns.cpu().tolist())
                total_lengths.extend(lengths.cpu().tolist())

        returns_t = torch.tensor(total_returns)
        lengths_t = torch.tensor(total_lengths)
        eval_info = {
            "task": args_cli.task,
            "checkpoint": checkpoint_path,
            "iteration": runner.current_learning_iteration,
            "num_episodes": num_episodes,
            "wall_time": wall_time,
            "avg_return": returns_t.mean().item(),
            "std_return": returns_t.std().item(),
            "avg_length": lengths_t.mean().item(),
        }
        eval_infos.append(eval_info)
        print(
            f"[INFO] {os.path.basename(checkpoint_path)} (it {eval_info['iteration']}):"
            f" avg_return {eval_info['avg_return']:.3f} ± {eval_info['std_return']:.3f},"
            f" avg_length {eval_info['avg_length']:.1f}"
        )

        # Save the metrics next to the checkpoint
        with open(out_path, "w") as f:
            json.dump(eval_info, f, indent=2)

    print(f"[INFO] Saved {len(eval_infos)} evaluation JSON file(s) to: {run_dir}")

    # When training logged to W&B, append the eval metrics to the same run
    logger_cfg = agent_cfg.logger
    if eval_infos and isinstance(logger_cfg, dict) and logger_cfg.get("class_name") == "WandbLogWriter":
        log_eval_to_wandb(project_name=logger_cfg["project_name"], run_dir=run_dir, eval_infos=eval_infos)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
