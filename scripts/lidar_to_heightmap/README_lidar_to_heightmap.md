## Overview

This module trains a supervised network that predicts the local heightmap from the Unitree L2 LiDAR point cloud and robot proprioception.

The LiDAR input is saved as a point-cloud sequence, not as a depth image:

```text
lidar_data: (num_samples, lidar_history, num_points, point_feature_dim)
robot_info: (num_samples, proprio_history, proprio_dim)
heightmaps: (num_samples, 1, heightmap_rows, heightmap_cols)
```

By default, `point_feature_dim=4`: `(x, y, z, valid)` in the LiDAR sensor frame. Use `--no_lidar_valid_mask` during collection to save only `(x, y, z)`.

## How To Use

1. Train a locomotion policy with the LiDAR-enabled task/config.

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Go2-Rough-Vision --num_envs=4096 --headless
```

2. Collect LiDAR-to-heightmap supervision data.

```bash
python scripts/lidar_to_heightmap/collect_lidar_to_heightmap.py \
  --task=Locomotion-Go2-Rough-Vision \
  --num_envs=8192 \
  --headless
```

3. Train one of the contained LiDAR networks.

```bash
python scripts/lidar_to_heightmap/terrain_reconstruction_pointnet.py \
  --dataset_path /path/to/lidar_terrain_reconstruction_dataset.pt
```

```bash
python scripts/lidar_to_heightmap/terrain_reconstruction_transformer.py \
  --dataset_path /path/to/lidar_terrain_reconstruction_dataset.pt
```

You can also use the combined entry point:

```bash
python scripts/lidar_to_heightmap/terrain_reconstruction_networks.py \
  --model pointnet_gru \
  --dataset_path /path/to/lidar_terrain_reconstruction_dataset.pt
```

The PointNet-GRU model is lighter and is the recommended first check. The transformer model uses cross-attention from proprioceptive tokens to LiDAR point tokens, so it can be more expensive when many points are kept. Use `--max_lidar_points` to cap the number of LiDAR points consumed per frame.

## Smoke Tests

Run without `--dataset_path` to train briefly on fake data.

```bash
python scripts/lidar_to_heightmap/terrain_reconstruction_pointnet.py --epochs 1 --num_samples 32
python scripts/lidar_to_heightmap/terrain_reconstruction_transformer.py --epochs 1 --num_samples 32 --max_lidar_points 256
```
