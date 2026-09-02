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

"""Play a trained FlashSAC policy on an Isaac Lab task.

Usage (from an environment with Isaac Lab installed):
    flashsac-play --task Isaac-Velocity-Rough-G1-Play-v0 --num_envs 32
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """Launch Isaac Sim and roll out a trained FlashSAC policy."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play a trained FlashSAC policy on an Isaac Lab task.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task (Isaac Lab gym id).")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Default: latest run.")
    parser.add_argument("--export_policy", action="store_true", help="Export the policy as JIT and ONNX.")
    parser.add_argument("--video", action="store_true", help="Record a video of the rollout, then exit.")
    parser.add_argument("--video_length", type=int, default=600, help="Video length in env steps.")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory the video is written to.")
    parser.add_argument("--viewer_eye", type=str, default=None, help="Viewer camera eye as 'x,y,z'.")
    parser.add_argument("--viewer_lookat", type=str, default=None, help="Viewer camera target as 'x,y,z'.")
    parser.add_argument(
        "--viewer_follow_env",
        type=int,
        default=None,
        help="Track the robot of this env index; eye/lookat become offsets relative to the robot.",
    )
    # A checkpoint trained with a train.py --action_bound override MUST be played with the
    # same bound - the policy's normalized [-1, 1] actions only mean what the training-time
    # wrapper scaling made them mean.
    parser.add_argument(
        "--action_bound", type=str, default=None, choices=["joint_limit", "scalar"], help="Action bound override."
    )
    parser.add_argument("--action_bound_scale", type=float, default=None, help="Scalar action bound half-width.")
    parser.add_argument(
        "--motion_files",
        type=str,
        nargs="+",
        default=None,
        help="Motion clip .npz path(s) or a directory of clips (required for tracking tasks).",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.video:
        args_cli.enable_cameras = True  # offscreen rendering is required for video capture

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Late imports: Isaac Lab modules require the simulation app to be running
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the Isaac Lab gym tasks)
    import torch
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import isaaclab_flashsac.envs  # noqa: F401  (registers the out-of-tree G1 DreamWaQ velocity + tracking tasks)
    from isaaclab_flashsac.deploy_export import verify_exported_pair, write_deploy_config
    from isaaclab_flashsac.envs.g1_wbt import apply_motion_files
    #from isaaclab_flashsac.rl_cfg import get_task_cfg
    from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
    from rsl_rl_flashsac.runners import OffPolicyRunner

    from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_flashsac_cfg import get_task_cfg

    agent_cfg = get_task_cfg(args_cli.task)
    if getattr(args_cli, "device", None) is not None:
        agent_cfg.device = args_cli.device
    if args_cli.action_bound is not None:
        agent_cfg.action_bound = args_cli.action_bound
    if args_cli.action_bound_scale is not None:
        agent_cfg.action_bound_scale = args_cli.action_bound_scale

    # Resolve the checkpoint
    if args_cli.checkpoint is not None:
        checkpoint_path = args_cli.checkpoint
    else:
        log_root = os.path.abspath(os.path.join("logs", "flashsac", agent_cfg.experiment_name))
        checkpoint_path = get_checkpoint_path(log_root, run_dir=".*", checkpoint="model_.*.pt")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # Build the environment
    env_cfg = parse_env_cfg(args_cli.task, device=agent_cfg.device, num_envs=args_cli.num_envs)
    apply_motion_files(env_cfg, args_cli.motion_files)
    if args_cli.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.viewer_eye.split(","))
    if args_cli.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.viewer_lookat.split(","))
    if args_cli.viewer_follow_env is not None:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.viewer_follow_env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = FlashSACVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        action_bound=agent_cfg.action_bound,
        action_bound_scale=agent_cfg.action_bound_scale,
    )

    # Load the policy
    train_cfg = agent_cfg.to_dict()  # type: ignore[attr-defined]
    runner = OffPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # Optionally export the deterministic policy for deployment
    if args_cli.export_policy:
        export_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
        runner.export_policy_to_jit(export_dir)
        runner.export_policy_to_onnx(export_dir)
        print(f"[INFO] Exported policy to: {export_dir}")

        # DreamWaQ variants additionally export the CENet estimator and the sim2real deploy
        # contract (config.yaml + joint order + provenance), then self-verify that the exported
        # policy.pt + cenet.pt compose to reproduce the live policy's action.
        if hasattr(policy, "cenet_as_jit"):
            runner.export_cenet_to_jit(export_dir)
            write_deploy_config(env, agent_cfg, export_dir)
            verify_exported_pair(policy, export_dir, env.get_observations())
            print(f"[INFO] Exported + verified cenet.pt and config.yaml to: {export_dir}")

    # Roll out (a finite clip when recording, otherwise until the app is closed)
    obs = env.get_observations().to(agent_cfg.device)
    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions.to(env.device))
            obs = obs.to(agent_cfg.device)
        steps += 1
        if args_cli.video and steps > args_cli.video_length:
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
