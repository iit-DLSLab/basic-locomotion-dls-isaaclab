# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained LiDAR terrain reconstructor online and display its heightmap with Isaac Lab markers."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from collections import deque

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--terrain_model_path",
    "--model_path",
    dest="terrain_model_path",
    required=True,
    help="Trained LiDAR-to-heightmap checkpoint produced by terrain_reconstruction_networks.py.",
)
parser.add_argument("--task", type=str, default="IEKF-Go2-Rough-Vision", help="Isaac Lab task to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulated environments.")
parser.add_argument("--visualized_env", type=int, default=0, help="Environment whose prediction is displayed.")
parser.add_argument(
    "--lidar_history_length",
    type=int,
    default=None,
    help="Override LiDAR history length. Defaults to the model's dataset metadata.",
)
parser.add_argument(
    "--proprio_history_length",
    type=int,
    default=None,
    help="Override proprioceptive history length. Defaults to the model's dataset metadata.",
)
parser.add_argument(
    "--lidar_clip_range",
    type=float,
    default=None,
    help="Override LiDAR clipping range in metres. Defaults to the model's dataset metadata.",
)
parser.add_argument(
    "--lidar_update_hz_min",
    type=float,
    default=None,
    help="Override minimum sample-and-hold LiDAR frequency.",
)
parser.add_argument(
    "--lidar_update_hz_max",
    type=float,
    default=None,
    help="Override maximum sample-and-hold LiDAR frequency.",
)
parser.add_argument(
    "--max_lidar_points",
    type=int,
    default=None,
    help="Optional inference-time LiDAR point cap. Defaults to the model checkpoint configuration.",
)
parser.add_argument("--marker_radius", type=float, default=0.025, help="Prediction marker sphere radius in metres.")
parser.add_argument(
    "--marker_height_offset",
    type=float,
    default=0.02,
    help="Vertical display offset applied to prediction and target markers.",
)
parser.add_argument(
    "--error_threshold",
    type=float,
    default=0.05,
    help="Absolute height error in metres below which prediction markers are green.",
)
parser.add_argument(
    "--show_ground_truth",
    action="store_true",
    help="Also display smaller white markers at the height-scanner target positions.",
)
parser.add_argument(
    "--show_lidar_rays",
    action="store_true",
    help="Keep the Unitree L2 ray-caster debug visualization enabled.",
)
parser.add_argument(
    "--metrics_interval",
    type=int,
    default=50,
    help="Print live MAE/RMSE every N simulation steps. Use 0 to disable.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after N simulation steps. Use 0 to run indefinitely.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration entry point.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the published locomotion-policy checkpoint instead of the one recorded in model metadata.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    raise ValueError("--num_envs must be positive.")
if not 0 <= args_cli.visualized_env < args_cli.num_envs:
    raise ValueError("--visualized_env must be in [0, num_envs).")
if args_cli.marker_radius <= 0.0:
    raise ValueError("--marker_radius must be positive.")
if args_cli.error_threshold < 0.0:
    raise ValueError("--error_threshold cannot be negative.")
if args_cli.max_lidar_points is not None and args_cli.max_lidar_points <= 0:
    raise ValueError("--max_lidar_points must be positive when provided.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import iekf_quadruped_isaaclab.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from terrain_reconstruction_networks import load_terrain_reconstructor_checkpoint


def _as_torch(value: Any) -> torch.Tensor:
    """Return a torch view for both classic tensors and Isaac Lab ProxyArray values."""
    torch_view = getattr(value, "torch", None)
    return torch_view if torch_view is not None else value


def _sanitize_lidar_data(
    env: RslRlVecEnvWrapper,
    lidar_clip_range: float,
    include_valid_mask: bool,
) -> torch.Tensor:
    if not hasattr(env.unwrapped, "_unitree_l2_lidar"):
        raise AttributeError(
            "Online terrain reconstruction needs env.unwrapped._unitree_l2_lidar. "
            "Use a task where use_unitree_l2_lidar=True."
        )

    lidar = env.unwrapped._unitree_l2_lidar
    ray_hits_w = _as_torch(lidar.data.ray_hits_w)
    sensor_pos_w = _as_torch(lidar.data.pos_w)
    sensor_quat_w = _as_torch(lidar.data.quat_w)

    valid = torch.isfinite(ray_hits_w).all(dim=-1, keepdim=True)
    relative_hits_w = torch.nan_to_num(ray_hits_w - sensor_pos_w.unsqueeze(1), nan=0.0, posinf=0.0, neginf=0.0)

    batch_size, num_points, _ = relative_hits_w.shape
    quat = sensor_quat_w.unsqueeze(1).expand(batch_size, num_points, 4).reshape(-1, 4)
    points_l = math_utils.quat_apply_inverse(quat, relative_hits_w.reshape(-1, 3)).reshape(batch_size, num_points, 3)
    points_l = points_l.clip(-lidar_clip_range, lidar_clip_range)
    points_l = points_l * valid.to(points_l.dtype)
    if include_valid_mask:
        return torch.cat((points_l, valid.to(points_l.dtype)), dim=-1)
    return points_l


class LidarSampleAndHold:
    """Match the asynchronous LiDAR sampling used by the dataset collector."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        update_hz_range: tuple[float, float],
        step_dt: float,
        lidar_clip_range: float,
        include_valid_mask: bool,
    ):
        self.env = env
        self.update_hz_range = update_hz_range
        self.step_dt = step_dt
        self.lidar_clip_range = lidar_clip_range
        self.include_valid_mask = include_valid_mask
        self.elapsed_s = 0.0
        self.next_update_period_s = self._sample_update_period()
        self.current_lidar = self._read_lidar().clone()

    def _read_lidar(self) -> torch.Tensor:
        return _sanitize_lidar_data(
            self.env,
            lidar_clip_range=self.lidar_clip_range,
            include_valid_mask=self.include_valid_mask,
        )

    def _sample_update_period(self) -> float:
        min_hz, max_hz = self.update_hz_range
        if min_hz == max_hz:
            return 1.0 / min_hz
        update_hz = min_hz + (max_hz - min_hz) * torch.rand((), device=self.env.unwrapped.device).item()
        return 1.0 / update_hz

    def step(self, dones: torch.Tensor | None = None) -> torch.Tensor:
        self.elapsed_s += self.step_dt
        force_refresh = bool(dones is not None and dones.any().item())
        if force_refresh or self.elapsed_s + 1.0e-9 >= self.next_update_period_s:
            self.current_lidar = self._read_lidar().clone()
            self.elapsed_s = 0.0
            self.next_update_period_s = self._sample_update_period()
        return self.current_lidar


def _get_heightmap_targets(env: RslRlVecEnvWrapper, heightmap_size: tuple[int, int]) -> torch.Tensor:
    scanner = env.unwrapped._perceptive_height_scanner
    sensor_pos_w = _as_torch(scanner.data.pos_w)
    ray_hits_w = _as_torch(scanner.data.ray_hits_w)
    height_data = sensor_pos_w[:, 2].unsqueeze(1) - ray_hits_w[..., 2] - 0.5
    height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0).clip(-1.0, 1.0)
    expected_cells = heightmap_size[0] * heightmap_size[1]
    if height_data.shape[1] != expected_cells:
        raise ValueError(
            f"Model expects a {heightmap_size} heightmap ({expected_cells} cells), "
            f"but the task height scanner has {height_data.shape[1]} rays."
        )
    return height_data.view(height_data.shape[0], 1, *heightmap_size)


def _create_prediction_markers(marker_radius: float) -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LidarHeightmapPrediction",
        markers={
            "close": sim_utils.SphereCfg(
                radius=marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
            ),
            "above_target": sim_utils.SphereCfg(
                radius=marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.05)),
            ),
            "below_target": sim_utils.SphereCfg(
                radius=marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.35, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _create_target_markers(marker_radius: float) -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LidarHeightmapTarget",
        markers={
            "target": sim_utils.SphereCfg(
                radius=max(marker_radius * 0.45, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def _update_markers(
    env: RslRlVecEnvWrapper,
    predicted_heightmap: torch.Tensor,
    target_heightmap: torch.Tensor,
    visualized_env: int,
    prediction_markers: VisualizationMarkers,
    target_markers: VisualizationMarkers | None,
) -> tuple[float, float, float]:
    scanner = env.unwrapped._perceptive_height_scanner
    sensor_pos_w = _as_torch(scanner.data.pos_w)
    ray_hits_w = _as_torch(scanner.data.ray_hits_w)

    predicted = predicted_heightmap[0, 0].reshape(-1)
    target = target_heightmap[visualized_env, 0].reshape(-1)
    ray_xy = ray_hits_w[visualized_env, :, :2]
    predicted_z = sensor_pos_w[visualized_env, 2] - predicted - 0.5 + args_cli.marker_height_offset
    target_z = sensor_pos_w[visualized_env, 2] - target - 0.5 + args_cli.marker_height_offset
    predicted_positions = torch.cat((ray_xy, predicted_z.unsqueeze(1)), dim=1)
    target_positions = torch.cat((ray_xy, target_z.unsqueeze(1)), dim=1)

    valid = torch.isfinite(predicted_positions).all(dim=1) & torch.isfinite(target_positions).all(dim=1)
    world_height_error = predicted_z - target_z
    marker_indices = torch.zeros_like(world_height_error, dtype=torch.long)
    marker_indices[world_height_error > args_cli.error_threshold] = 1
    marker_indices[world_height_error < -args_cli.error_threshold] = 2
    prediction_markers.visualize(
        translations=predicted_positions[valid],
        marker_indices=marker_indices[valid],
    )
    if target_markers is not None:
        target_markers.visualize(translations=target_positions[valid])

    value_error = predicted - target
    mae = float(value_error.abs().mean())
    rmse = float(value_error.square().mean().sqrt())
    max_abs_error = float(value_error.abs().max())
    return mae, rmse, max_abs_error


def _metadata_value(
    metadata: Mapping[str, Any],
    key: str,
    override: Any,
    fallback: Any,
) -> Any:
    if override is not None:
        return override
    return metadata.get(key, fallback)


def _resolve_policy_checkpoint(
    agent_cfg: RslRlBaseRunnerCfg,
    train_task_name: str,
    model_metadata: Mapping[str, Any],
) -> str:
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not checkpoint_path:
            raise RuntimeError(f"No published checkpoint is available for task {train_task_name}.")
        return checkpoint_path
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)

    recorded_checkpoint = model_metadata.get("checkpoint_path")
    if recorded_checkpoint and Path(str(recorded_checkpoint)).expanduser().is_file():
        return str(Path(str(recorded_checkpoint)).expanduser().resolve())

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run locomotion, terrain inference, and marker visualization."""
    terrain_model, terrain_checkpoint = load_terrain_reconstructor_checkpoint(
        args_cli.terrain_model_path,
        device="cpu",
        max_lidar_points=args_cli.max_lidar_points,
    )
    model_config = terrain_checkpoint["model_config"]
    model_metadata = terrain_checkpoint.get("dataset_metadata", {})
    if not isinstance(model_metadata, Mapping):
        model_metadata = {}

    lidar_history_length = int(
        _metadata_value(model_metadata, "lidar_history_length", args_cli.lidar_history_length, 5)
    )
    proprio_history_length = int(
        _metadata_value(model_metadata, "proprio_history_length", args_cli.proprio_history_length, 50)
    )
    lidar_clip_range = float(_metadata_value(model_metadata, "lidar_clip_range", args_cli.lidar_clip_range, 2.0))
    metadata_hz_range = model_metadata.get("lidar_update_hz_range", (5.0, 10.0))
    lidar_update_hz_min = float(
        args_cli.lidar_update_hz_min if args_cli.lidar_update_hz_min is not None else metadata_hz_range[0]
    )
    lidar_update_hz_max = float(
        args_cli.lidar_update_hz_max if args_cli.lidar_update_hz_max is not None else metadata_hz_range[1]
    )
    if lidar_history_length <= 0 or proprio_history_length <= 0:
        raise ValueError("History lengths must be positive.")
    if lidar_clip_range <= 0.0:
        raise ValueError("LiDAR clip range must be positive.")
    if lidar_update_hz_min <= 0.0 or lidar_update_hz_max < lidar_update_hz_min:
        raise ValueError("LiDAR update frequencies must be positive and ordered min <= max.")

    point_feature_dim = int(model_config["point_feature_dim"])
    if point_feature_dim not in (3, 4):
        raise ValueError(
            f"Online LiDAR preprocessing supports 3 (xyz) or 4 (xyz+valid) features, got {point_feature_dim}."
        )
    include_valid_mask = point_feature_dim == 4
    heightmap_size = tuple(model_config["heightmap_size"])
    expected_proprio_dim = int(model_config["proprio_dim"])

    recorded_task = model_metadata.get("task")
    if recorded_task and recorded_task != args_cli.task:
        print(f"[WARN] Model was collected with task {recorded_task!r}, but this run uses {args_cli.task!r}.")

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "randomize_far_goal_yaw"):
        env_cfg.randomize_far_goal_yaw = False
    if hasattr(env_cfg, "visualize_goal"):
        env_cfg.visualize_goal = False
    if hasattr(env_cfg, "unitree_l2_lidar"):
        env_cfg.unitree_l2_lidar.update_period = min(
            env_cfg.unitree_l2_lidar.update_period,
            1.0 / lidar_update_hz_max,
        )
        env_cfg.unitree_l2_lidar.debug_vis = args_cli.show_lidar_rays
    if hasattr(env_cfg, "perceptive_height_scanner"):
        env_cfg.perceptive_height_scanner.debug_vis = False

    policy_checkpoint = _resolve_policy_checkpoint(agent_cfg, train_task_name, model_metadata)
    env_cfg.log_dir = os.path.dirname(policy_checkpoint)
    print(f"[INFO] Loading terrain model: {Path(args_cli.terrain_model_path).expanduser().resolve()}")
    print(f"[INFO] Loading locomotion policy: {policy_checkpoint}")
    print(
        "[INFO] Online preprocessing: "
        f"lidar_history={lidar_history_length}, proprio_history={proprio_history_length}, "
        f"lidar_hz=({lidar_update_hz_min:g}, {lidar_update_hz_max:g}), clip={lidar_clip_range:g} m"
    )
    print("[INFO] Marker colors: green=close, orange=above target, blue=below target.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(policy_checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    terrain_model.to(env.unwrapped.device)
    terrain_model.eval()
    obs = env.get_observations()
    if "common" not in obs:
        raise KeyError("The task observation dictionary has no 'common' proprioceptive observation.")
    if obs["common"].shape[-1] != expected_proprio_dim:
        raise ValueError(
            f"Model expects proprio_dim={expected_proprio_dim}, but task observation 'common' has "
            f"{obs['common'].shape[-1]} features."
        )

    step_dt = float(env.unwrapped.step_dt)
    lidar_sampler = LidarSampleAndHold(
        env=env,
        update_hz_range=(lidar_update_hz_min, lidar_update_hz_max),
        step_dt=step_dt,
        lidar_clip_range=lidar_clip_range,
        include_valid_mask=include_valid_mask,
    )
    if lidar_sampler.current_lidar.shape[-1] != point_feature_dim:
        raise ValueError(
            f"Model expects {point_feature_dim} LiDAR features, but online preprocessing produced "
            f"{lidar_sampler.current_lidar.shape[-1]}."
        )
    _get_heightmap_targets(env, heightmap_size)

    lidar_history: deque[torch.Tensor] = deque(maxlen=lidar_history_length)
    proprio_history: deque[torch.Tensor] = deque(maxlen=proprio_history_length)
    lidar_history.append(lidar_sampler.current_lidar.clone())
    proprio_history.append(obs["common"].clone())
    max_history_length = max(lidar_history_length, proprio_history_length)
    valid_history_lengths = torch.ones(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)

    prediction_markers = _create_prediction_markers(args_cli.marker_radius)
    target_markers = _create_target_markers(args_cli.marker_radius) if args_cli.show_ground_truth else None
    prediction_markers.set_visibility(False)
    if target_markers is not None:
        target_markers.set_visibility(False)

    timestep = 0
    warmup_announced = False
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()
            if hasattr(policy, "reset"):
                policy.reset(dones)

            current_lidar = lidar_sampler.step(dones=dones)
            valid_history_lengths[dones] = 0
            lidar_history.append(current_lidar.clone())
            proprio_history.append(obs["common"].clone())
            valid_history_lengths = torch.clamp(valid_history_lengths + 1, max=max_history_length)

            histories_ready = (
                len(lidar_history) == lidar_history_length
                and len(proprio_history) == proprio_history_length
                and valid_history_lengths[args_cli.visualized_env] >= max_history_length
            )
            if histories_ready:
                env_slice = slice(args_cli.visualized_env, args_cli.visualized_env + 1)
                lidar_sequence = torch.stack(tuple(lidar_history), dim=1)[env_slice]
                proprio_sequence = torch.stack(tuple(proprio_history), dim=1)[env_slice]
                prediction = terrain_model(lidar_data=lidar_sequence, robot_info=proprio_sequence)
                target_heightmap = _get_heightmap_targets(env, heightmap_size)

                prediction_markers.set_visibility(True)
                if target_markers is not None:
                    target_markers.set_visibility(True)
                mae, rmse, max_abs_error = _update_markers(
                    env=env,
                    predicted_heightmap=prediction.refined_heightmap,
                    target_heightmap=target_heightmap,
                    visualized_env=args_cli.visualized_env,
                    prediction_markers=prediction_markers,
                    target_markers=target_markers,
                )
                if args_cli.metrics_interval > 0 and timestep % args_cli.metrics_interval == 0:
                    print(
                        f"[METRICS] step={timestep} env={args_cli.visualized_env} "
                        f"MAE={mae:.4f} m RMSE={rmse:.4f} m max_abs_error={max_abs_error:.4f} m"
                    )
                warmup_announced = False
            else:
                prediction_markers.set_visibility(False)
                if target_markers is not None:
                    target_markers.set_visibility(False)
                if not warmup_announced:
                    remaining = max(
                        max_history_length - int(valid_history_lengths[args_cli.visualized_env].item()),
                        0,
                    )
                    print(f"[INFO] Warming up network histories for env {args_cli.visualized_env}: {remaining} steps.")
                    warmup_announced = True

        timestep += 1
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break
        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
