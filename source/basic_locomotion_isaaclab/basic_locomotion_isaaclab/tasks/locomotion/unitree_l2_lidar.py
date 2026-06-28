"""Unitree L2 LiDAR ray-cast pattern."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from isaaclab.utils import configclass


def unitree_l2_non_repetitive_pattern(
    cfg: "UnitreeL2PatternCfg", device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one Unitree L2-style non-repetitive scan."""
    full_points_per_scan = int(round(cfg.effective_points_per_second / cfg.circumferential_scan_frequency))
    if full_points_per_scan <= 0:
        raise ValueError(f"points_per_scan must be positive. Received: {full_points_per_scan}.")
    if not 0.0 < cfg.keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1]. Received: {cfg.keep_ratio}.")

    points_per_scan = full_points_per_scan
    if cfg.enable_downsample:
        points_per_scan = max(1, int(round(full_points_per_scan * cfg.keep_ratio)))

    sample_ids = torch.arange(points_per_scan, dtype=torch.float32, device=device)
    yaw_min = math.radians(cfg.horizontal_fov_range[0])
    yaw_span = math.radians(cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0])
    yaw_fraction = torch.remainder(sample_ids * cfg.horizontal_stride + cfg.horizontal_phase, 1.0)
    yaw = yaw_min + yaw_span * yaw_fraction

    vertical_min_deg, vertical_max_deg = cfg.vertical_fov_range
    if cfg.vertical_fov_orientation == "down":
        vertical_min_deg, vertical_max_deg = -vertical_max_deg, -vertical_min_deg
    elif cfg.vertical_fov_orientation != "up":
        raise ValueError(
            f"vertical_fov_orientation must be 'down' or 'up'. Received: {cfg.vertical_fov_orientation}."
        )
    vertical_min = math.radians(vertical_min_deg)
    vertical_span = math.radians(vertical_max_deg - vertical_min_deg)
    vertical_fraction = torch.remainder(sample_ids * cfg.vertical_stride + cfg.vertical_phase, 1.0)
    elevation = vertical_min + vertical_span * vertical_fraction

    cos_elevation = torch.cos(elevation)
    ray_directions = torch.stack(
        (
            cos_elevation * torch.cos(yaw),
            cos_elevation * torch.sin(yaw),
            torch.sin(elevation),
        ),
        dim=-1,
    )
    ray_directions = ray_directions / torch.linalg.norm(ray_directions, dim=-1, keepdim=True)

    ray_starts = cfg.near_blind_spot * ray_directions
    return ray_starts, ray_directions


@configclass
class UnitreeL2PatternCfg:
    """Configuration for the Unitree L2 non-repetitive LiDAR scan pattern."""

    func: Callable = unitree_l2_non_repetitive_pattern

    horizontal_fov_range: tuple[float, float] = (0.0, 360.0)
    """Horizontal field of view range in degrees."""

    vertical_fov_range: tuple[float, float] = (-6.0, 90.0)
    """Vertical field of view range in degrees, using the L2 negative-angle mode."""

    vertical_fov_orientation: Literal["down", "up"] = "down"
    """Whether the vertical FOV is projected mostly below or above the horizon in Isaac's z-up frame."""

    effective_points_per_second: float = 64000.0
    """Effective point rate in points per second."""

    circumferential_scan_frequency: float = 5.55
    """Circumferential scan frequency in Hz."""

    vertical_scan_frequency: float = 216.0
    """Vertical scan frequency in Hz."""

    near_blind_spot: float = 0.05
    """Minimum detection distance in meters."""

    vertical_phase: float = 0.0
    """Initial phase of the vertical scanner as a fraction of a cycle."""

    horizontal_phase: float = 0.0
    """Initial phase of the circumferential scanner as a fraction of a turn."""

    horizontal_stride: float = 0.6180339887498949
    """Low-discrepancy horizontal stride for the non-repetitive accumulated frame."""

    vertical_stride: float = 0.4142135623730951
    """Low-discrepancy vertical stride for the non-repetitive accumulated frame."""

    enable_downsample: bool = True
    """Whether to downsample the simulated ray pattern."""

    keep_ratio: float = 0.9
    """Fraction of rays to keep when downsampling."""

    distance_resolution: float = 0.0045
    """Distance resolution in meters. Stored for downstream post-processing."""

    measurement_accuracy: float = 0.02
    """Range accuracy in meters. Stored for downstream post-processing."""
