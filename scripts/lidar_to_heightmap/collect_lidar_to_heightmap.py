# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a Unitree L2 LiDAR-to-heightmap dataset using a trained locomotion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from collections import deque

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect a LiDAR terrain reconstruction dataset with a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="Where to save the dataset. Defaults to <checkpoint_dir>/lidar_terrain_reconstruction_dataset.pt.",
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
    help="Optional fixed rollout length. Defaults to the environment max episode length.",
)
parser.add_argument(
    "--lidar_history_length",
    type=int,
    default=5,
    help="Number of LiDAR frames per saved training sample.",
)
parser.add_argument(
    "--proprio_history_length",
    type=int,
    default=50,
    help="Number of robot-info frames per saved training sample.",
)
parser.add_argument(
    "--lidar_clip_range",
    type=float,
    default=2.0,
    help="Absolute clipping range in meters for LiDAR points expressed in the sensor frame.",
)
parser.add_argument(
    "--lidar_update_hz_min",
    type=float,
    default=5.0,
    help="Minimum simulated LiDAR arrival frequency in Hz for sample-and-hold collection.",
)
parser.add_argument(
    "--lidar_update_hz_max",
    type=float,
    default=10.0,
    help="Maximum simulated LiDAR arrival frequency in Hz for sample-and-hold collection.",
)
parser.add_argument(
    "--no_lidar_valid_mask",
    action="store_true",
    default=False,
    help="Do not append a validity channel to each LiDAR point.",
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.lidar_update_hz_min <= 0.0 or args_cli.lidar_update_hz_max <= 0.0:
    raise ValueError("--lidar_update_hz_min and --lidar_update_hz_max must be positive.")
if args_cli.lidar_update_hz_min > args_cli.lidar_update_hz_max:
    raise ValueError("--lidar_update_hz_min cannot be greater than --lidar_update_hz_max.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import basic_locomotion_isaaclab.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class LidarTerrainReconstructionDatasetBuilder:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.lidar_batches: list[torch.Tensor] = []
        self.robot_info_batches: list[torch.Tensor] = []
        self.heightmap_batches: list[torch.Tensor] = []
        self.num_samples = 0

    def add_batch(self, lidar_data: torch.Tensor, robot_info: torch.Tensor, heightmaps: torch.Tensor) -> int:
        if self.num_samples >= self.max_samples:
            return 0

        remaining = self.max_samples - self.num_samples
        batch_size = lidar_data.shape[0]
        if batch_size > remaining:
            selected_indices = torch.randperm(batch_size, device=lidar_data.device)[:remaining]
            lidar_data = lidar_data[selected_indices]
            robot_info = robot_info[selected_indices]
            heightmaps = heightmaps[selected_indices]
            batch_size = remaining

        self.lidar_batches.append(lidar_data.detach().cpu())
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
            "lidar_data": torch.cat(self.lidar_batches, dim=0),
            "robot_info": torch.cat(self.robot_info_batches, dim=0),
            "heightmaps": torch.cat(self.heightmap_batches, dim=0),
            "metadata": {
                **metadata,
                "num_samples": self.num_samples,
            },
        }
        torch.save(dataset, dataset_path)
        print(f"[INFO] Saved LiDAR terrain reconstruction dataset to: {dataset_path}")


def _sanitize_lidar_data(env: RslRlVecEnvWrapper) -> torch.Tensor:
    if not hasattr(env.unwrapped, "_unitree_l2_lidar"):
        raise AttributeError(
            "The LiDAR-to-heightmap collector needs env.unwrapped._unitree_l2_lidar. "
            "Run it with a task/config where use_unitree_l2_lidar=True."
        )

    lidar = env.unwrapped._unitree_l2_lidar
    ray_hits_w = lidar.data.ray_hits_w
    sensor_pos_w = lidar.data.pos_w
    sensor_quat_w = lidar.data.quat_w

    valid = torch.isfinite(ray_hits_w).all(dim=-1, keepdim=True)
    relative_hits_w = torch.nan_to_num(ray_hits_w - sensor_pos_w.unsqueeze(1), nan=0.0, posinf=0.0, neginf=0.0)

    batch_size, num_points, _ = relative_hits_w.shape
    quat = sensor_quat_w.unsqueeze(1).expand(batch_size, num_points, 4).reshape(-1, 4)
    points_l = math_utils.quat_apply_inverse(quat, relative_hits_w.reshape(-1, 3)).reshape(batch_size, num_points, 3)
    points_l = points_l.clip(-args_cli.lidar_clip_range, args_cli.lidar_clip_range)
    points_l = points_l * valid.to(points_l.dtype)

    if args_cli.no_lidar_valid_mask:
        return points_l

    return torch.cat((points_l, valid.to(points_l.dtype)), dim=-1)


class LidarSampleAndHold:
    """Hold the latest LiDAR frame while the policy/proprioception runs faster."""

    def __init__(self, env: RslRlVecEnvWrapper, update_hz_range: tuple[float, float], step_dt: float):
        self.env = env
        self.update_hz_range = update_hz_range
        self.step_dt = step_dt
        self.elapsed_s = 0.0
        self.next_update_period_s = self._sample_update_period()
        self.current_lidar = _sanitize_lidar_data(env).clone()

    def _sample_update_period(self) -> float:
        min_hz, max_hz = self.update_hz_range
        if min_hz == max_hz:
            return 1.0 / min_hz
        update_hz = min_hz + (max_hz - min_hz) * torch.rand((), device=self.env.unwrapped.device).item()
        return 1.0 / update_hz

    def step(self, dones: torch.Tensor | None = None) -> torch.Tensor:
        self.elapsed_s += self.step_dt
        force_refresh = bool(dones is not None and dones.any().item())
        if force_refresh or self.elapsed_s + 1e-9 >= self.next_update_period_s:
            self.current_lidar = _sanitize_lidar_data(self.env).clone()
            self.elapsed_s = 0.0
            self.next_update_period_s = self._sample_update_period()
        return self.current_lidar


def _get_heightmap_grid_shape(env: RslRlVecEnvWrapper, num_rays: int) -> tuple[int, int]:
    pattern_cfg = env.unwrapped.cfg.perceptive_height_scanner.pattern_cfg
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
    height_data = (
        env.unwrapped._perceptive_height_scanner.data.pos_w[:, 2].unsqueeze(1)
        - env.unwrapped._perceptive_height_scanner.data.ray_hits_w[..., 2]
        - 0.5
    )
    height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data = height_data.clip(-1.0, 1.0)

    heightmap_rows, heightmap_cols = _get_heightmap_grid_shape(env=env, num_rays=height_data.shape[1])
    heightmaps = height_data.view(height_data.shape[0], 1, heightmap_rows, heightmap_cols)
    return heightmaps, (heightmap_rows, heightmap_cols)


def _default_dataset_path(log_dir: str) -> str:
    return os.path.join(log_dir, "lidar_terrain_reconstruction_dataset.pt")


def _maybe_save_checkpoint(
    dataset_builder: LidarTerrainReconstructionDatasetBuilder,
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
    """Collect LiDAR/robot-state/heightmap tuples for terrain reconstruction training."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "unitree_l2_lidar"):
        fastest_lidar_period = 1.0 / args_cli.lidar_update_hz_max
        current_update_period = getattr(env_cfg.unitree_l2_lidar, "update_period", 0.0)
        if current_update_period > fastest_lidar_period:
            env_cfg.unitree_l2_lidar.update_period = fastest_lidar_period

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
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
            "video_folder": os.path.join(log_dir, "videos", "collect_lidar_to_heightmap"),
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
    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 50.0))
    lidar_sampler = LidarSampleAndHold(
        env=env,
        update_hz_range=(args_cli.lidar_update_hz_min, args_cli.lidar_update_hz_max),
        step_dt=step_dt,
    )
    current_lidar = lidar_sampler.current_lidar
    current_heightmaps, heightmap_size = _get_heightmap_targets(env)

    num_envs = current_lidar.shape[0]
    rollout_horizon = args_cli.rollout_horizon
    if rollout_horizon is None:
        rollout_horizon = int(getattr(env.unwrapped, "max_episode_length", 200))

    max_history_length = max(args_cli.lidar_history_length, args_cli.proprio_history_length)
    valid_history_lengths = torch.ones(num_envs, dtype=torch.long, device=env.unwrapped.device)

    lidar_history: deque[torch.Tensor] = deque(maxlen=args_cli.lidar_history_length)
    robot_history: deque[torch.Tensor] = deque(maxlen=args_cli.proprio_history_length)
    lidar_history.append(current_lidar.clone())
    robot_history.append(obs["common"].clone())

    dataset_builder = LidarTerrainReconstructionDatasetBuilder(max_samples=args_cli.max_dataset_samples)
    collected_rollouts = 0
    rollout_step = 0

    metadata = {
        "task": args_cli.task,
        "checkpoint_path": resume_path,
        "robot_obs_key": "common",
        "lidar_history_length": args_cli.lidar_history_length,
        "proprio_history_length": args_cli.proprio_history_length,
        "lidar_num_points": current_lidar.shape[-2],
        "lidar_feature_dim": current_lidar.shape[-1],
        "lidar_frame": "unitree_l2_lidar",
        "lidar_valid_mask_channel": not args_cli.no_lidar_valid_mask,
        "lidar_clip_range": args_cli.lidar_clip_range,
        "lidar_sample_and_hold": True,
        "lidar_update_hz_range": (args_cli.lidar_update_hz_min, args_cli.lidar_update_hz_max),
        "collector_step_dt": step_dt,
        "collector_inference_hz": 1.0 / step_dt,
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

            current_lidar = lidar_sampler.step(dones=dones)
            current_heightmaps, _ = _get_heightmap_targets(env)
            current_robot_info = obs["common"].clone()

        valid_history_lengths[dones] = 0
        lidar_history.append(current_lidar.clone())
        robot_history.append(current_robot_info)
        valid_history_lengths = torch.clamp(valid_history_lengths + 1, max=max_history_length)

        if len(lidar_history) >= args_cli.lidar_history_length and len(robot_history) >= args_cli.proprio_history_length:
            valid_mask = valid_history_lengths >= max_history_length
            if valid_mask.any():
                lidar_sequence = torch.stack(list(lidar_history), dim=1)
                robot_sequence = torch.stack(list(robot_history), dim=1)
                added_samples = dataset_builder.add_batch(
                    lidar_data=lidar_sequence[valid_mask],
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
        raise RuntimeError("No LiDAR reconstruction samples were collected. Try increasing rollout count.")

    dataset_builder.save(dataset_path, metadata)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
