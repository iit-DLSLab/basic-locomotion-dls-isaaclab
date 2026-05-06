# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a depth-to-heightmap dataset using a trained locomotion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect a terrain reconstruction dataset with a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="Where to save the collected dataset. Defaults to <checkpoint_dir>/terrain_reconstruction_dataset.pt.",
)
parser.add_argument(
    "--num_collection_rollouts",
    type=int,
    default=25,
    help="Number of rollout windows to collect before saving the dataset.",
)
parser.add_argument(
    "--rollout_horizon",
    type=int,
    default=None,
    help="Optional fixed rollout length. Defaults to the environment max episode length when available.",
)
parser.add_argument(
    "--depth_history_length",
    type=int,
    default=5,
    help="Number of depth frames per saved training sample.",
)
parser.add_argument(
    "--proprio_history_length",
    type=int,
    default=50,
    help="Number of robot-info frames per saved training sample.",
)
parser.add_argument(
    "--max_dataset_samples",
    type=int,
    default=5000,
    help="Maximum number of training samples to keep in the saved dataset.",
)
parser.add_argument(
    "--save_every_rollouts",
    type=int,
    default=5,
    help="Save an intermediate dataset checkpoint every N rollout windows.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import basic_locomotion_isaaclab.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class TerrainReconstructionDatasetBuilder:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.depth_batches: list[torch.Tensor] = []
        self.robot_info_batches: list[torch.Tensor] = []
        self.heightmap_batches: list[torch.Tensor] = []
        self.num_samples = 0

    def add_batch(self, depth_data: torch.Tensor, robot_info: torch.Tensor, heightmaps: torch.Tensor) -> int:
        if self.num_samples >= self.max_samples:
            return 0

        remaining = self.max_samples - self.num_samples
        batch_size = depth_data.shape[0]
        if batch_size > remaining:
            selected_indices = torch.randperm(batch_size, device=depth_data.device)[:remaining]
            depth_data = depth_data[selected_indices]
            robot_info = robot_info[selected_indices]
            heightmaps = heightmaps[selected_indices]
            batch_size = remaining

        self.depth_batches.append(depth_data.detach().cpu())
        self.robot_info_batches.append(robot_info.detach().cpu())
        self.heightmap_batches.append(heightmaps.detach().cpu())
        self.num_samples += batch_size
        return batch_size

    def save(self, dataset_path: str, metadata: dict) -> None:
        if self.num_samples == 0:
            raise RuntimeError("No dataset samples were collected, so nothing can be saved.")

        dataset_dir = os.path.dirname(dataset_path)
        if dataset_dir:
            os.makedirs(dataset_dir, exist_ok=True)
        dataset = {
            "depth_data": torch.cat(self.depth_batches, dim=0),
            "robot_info": torch.cat(self.robot_info_batches, dim=0),
            "heightmaps": torch.cat(self.heightmap_batches, dim=0),
            "metadata": {
                **metadata,
                "num_samples": self.num_samples,
            },
        }
        torch.save(dataset, dataset_path)
        print(f"[INFO] Saved terrain reconstruction dataset to: {dataset_path}")


def _sanitize_depth_data(env: RslRlVecEnvWrapper) -> torch.Tensor:
    depth_data = env.unwrapped._depth_camera.data.output["distance_to_image_plane"]
    depth_data = torch.nan_to_num(depth_data, nan=0.0, posinf=1.0, neginf=-1.0)
    depth_data = depth_data.clip(-2.0, 2.0)
    depth_data = depth_data.permute(0, 3, 1, 2)
    return depth_data


def _get_heightmap_grid_shape(env: RslRlVecEnvWrapper, num_rays: int) -> tuple[int, int]:
    pattern_cfg = env.unwrapped.cfg.height_scanner2.pattern_cfg
    heightmap_cols = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
    if num_rays % heightmap_cols != 0:
        heightmap_rows = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
    else:
        heightmap_rows = num_rays // heightmap_cols

    if heightmap_rows * heightmap_cols != num_rays:
        raise ValueError(
            f"Could not infer heightmap grid shape from {num_rays} rays and config "
            f"(rows={heightmap_rows}, cols={heightmap_cols})."
        )
    return heightmap_rows, heightmap_cols


def _get_heightmap_targets(env: RslRlVecEnvWrapper) -> tuple[torch.Tensor, tuple[int, int]]:
    height_data = env.unwrapped._height_scanner2.data.pos_w[:, 2].unsqueeze(1) - env.unwrapped._height_scanner2.data.ray_hits_w[..., 2] - 0.5
    height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data = height_data.clip(-1.0, 1.0)

    heightmap_rows, heightmap_cols = _get_heightmap_grid_shape(env=env, num_rays=height_data.shape[1])
    heightmaps = height_data.view(height_data.shape[0], 1, heightmap_rows, heightmap_cols)
    return heightmaps, (heightmap_rows, heightmap_cols)


def _default_dataset_path(log_dir: str) -> str:
    return os.path.join(log_dir, "terrain_reconstruction_dataset.pt")


def _maybe_save_checkpoint(
    dataset_builder: TerrainReconstructionDatasetBuilder,
    dataset_path: str,
    metadata: dict,
    collected_rollouts: int,
) -> None:
    if dataset_builder.num_samples == 0:
        return

    checkpoint_path = dataset_path.replace(".pt", f"_rollouts_{collected_rollouts}.pt")
    dataset_builder.save(checkpoint_path, metadata)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect depth/robot-state/heightmap tuples for terrain reconstruction training."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    dataset_path = os.path.abspath(args_cli.dataset_path) if args_cli.dataset_path else _default_dataset_path(log_dir)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "collect_depth_to_heightmap"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during dataset collection.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    current_depth = _sanitize_depth_data(env)
    current_heightmaps, heightmap_size = _get_heightmap_targets(env)

    num_envs = current_depth.shape[0]
    rollout_horizon = args_cli.rollout_horizon
    if rollout_horizon is None:
        rollout_horizon = int(getattr(env.unwrapped, "max_episode_length", 200))

    max_history_length = max(args_cli.depth_history_length, args_cli.proprio_history_length)
    valid_history_lengths = torch.ones(num_envs, dtype=torch.long, device=env.unwrapped.device)

    depth_history: deque[torch.Tensor] = deque(maxlen=args_cli.depth_history_length)
    robot_history: deque[torch.Tensor] = deque(maxlen=args_cli.proprio_history_length)
    depth_history.append(current_depth.clone())
    robot_history.append(obs["common"].clone())

    dataset_builder = TerrainReconstructionDatasetBuilder(max_samples=args_cli.max_dataset_samples)
    collected_rollouts = 0
    rollout_step = 0

    metadata = {
        "task": args_cli.task,
        "checkpoint_path": resume_path,
        "robot_obs_key": "common",
        "depth_history_length": args_cli.depth_history_length,
        "proprio_history_length": args_cli.proprio_history_length,
        "depth_image_size": tuple(current_depth.shape[-2:]),
        "heightmap_size": heightmap_size,
        "num_envs": num_envs,
    }

    while (
        simulation_app.is_running()
        and collected_rollouts < args_cli.num_collection_rollouts
        and dataset_builder.num_samples < args_cli.max_dataset_samples
    ):
        with torch.no_grad():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()

            current_depth = _sanitize_depth_data(env)
            current_heightmaps, _ = _get_heightmap_targets(env)
            current_robot_info = obs["common"].clone()

        valid_history_lengths[dones] = 0
        depth_history.append(current_depth.clone())
        robot_history.append(current_robot_info)
        valid_history_lengths = torch.clamp(valid_history_lengths + 1, max=max_history_length)

        if len(depth_history) >= args_cli.depth_history_length and len(robot_history) >= args_cli.proprio_history_length:
            valid_mask = valid_history_lengths >= max_history_length
            if valid_mask.any():
                depth_sequence = torch.stack(list(depth_history), dim=1)
                robot_sequence = torch.stack(list(robot_history), dim=1)
                added_samples = dataset_builder.add_batch(
                    depth_data=depth_sequence[valid_mask],
                    robot_info=robot_sequence[valid_mask],
                    heightmaps=current_heightmaps[valid_mask],
                )
                if added_samples > 0 and dataset_builder.num_samples % 1000 < added_samples:
                    print(f"[INFO] Collected {dataset_builder.num_samples} / {args_cli.max_dataset_samples} samples.")

        rollout_step += 1
        if rollout_step >= rollout_horizon:
            collected_rollouts += 1
            rollout_step = 0
            print(
                f"[INFO] Completed rollout {collected_rollouts}/{args_cli.num_collection_rollouts} "
                f"with {dataset_builder.num_samples} saved samples."
            )

            if args_cli.save_every_rollouts > 0 and collected_rollouts % args_cli.save_every_rollouts == 0:
                _maybe_save_checkpoint(
                    dataset_builder=dataset_builder,
                    dataset_path=dataset_path,
                    metadata=metadata,
                    collected_rollouts=collected_rollouts,
                )

    if dataset_builder.num_samples == 0:
        raise RuntimeError("No terrain reconstruction samples were collected. Try increasing rollout count or lowering history lengths.")

    dataset_builder.save(dataset_path, metadata)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
