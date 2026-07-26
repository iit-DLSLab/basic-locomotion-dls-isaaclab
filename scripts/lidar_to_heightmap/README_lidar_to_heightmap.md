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

The collector runs one policy/environment step per sample, typically 50 Hz for this task, but simulates a slower LiDAR stream with sample-and-hold. By default, the LiDAR frame is refreshed at a random frequency between 5 Hz and 10 Hz, while proprioception and policy inputs stay at the latest 50 Hz values:

```bash
python scripts/lidar_to_heightmap/collect_lidar_to_heightmap.py \
  --task=Locomotion-Go2-Rough-Vision \
  --num_envs=8192 \
  --lidar_update_hz_min=5.0 \
  --lidar_update_hz_max=10.0 \
  --headless
```

3. Train one of the contained LiDAR networks.

```bash
python scripts/lidar_to_heightmap/train_reconstruction_pointnet.py \
  --dataset_path /path/to/lidar_terrain_reconstruction_dataset.pt
```

```bash
python scripts/lidar_to_heightmap/train_reconstruction_transformer.py \
  --dataset_path /path/to/lidar_terrain_reconstruction_dataset.pt
```

You can also use the combined entry point:

```bash
python scripts/lidar_to_heightmap/terrain_reconstruction_networks.py \
  --model pointnet_gru \
  --dataset_path /path/to/lidar_terrain_reconstruction_dataset.pt
```

The PointNet-GRU model is lighter and is the recommended first check. The transformer model uses cross-attention from proprioceptive tokens to LiDAR point tokens, so it can be more expensive when many points are kept. Use `--max_lidar_points` to cap the number of LiDAR points consumed per frame.


## Visualize Live Predictions in Isaac Lab

Run the trained network on the live Unitree L2 LiDAR and display the generated
heightmap as markers in the Isaac Lab GUI:

```bash
python scripts/lidar_to_heightmap/visualize_heightmap_prediction_isaaclab.py \
  --terrain_model_path /path/to/transformer_lidar_terrain_reconstructor.pt \
  --task IEKF-Go2-Rough-Vision \
  --num_envs 1
```

The script reads the training metadata from the terrain-model checkpoint,
including the locomotion-policy checkpoint, LiDAR/proprioception history lengths,
LiDAR clipping, and sample-and-hold frequency. Pass `--checkpoint` to explicitly
select a different locomotion policy.

Each predicted heightmap cell is rendered as a sphere:

- Green: prediction is within `--error_threshold` of the height-scanner target.
- Orange: predicted world height is above the target.
- Blue: predicted world height is below the target.

Useful options:

- `--show_ground_truth` overlays smaller white target markers.
- `--show_lidar_rays` enables the Unitree L2 ray-caster debug visualization.
- `--visualized_env 3` selects one environment when running multiple environments.
- `--marker_radius 0.025` and `--marker_height_offset 0.02` control marker appearance.
- `--real-time` throttles the simulation to wall-clock speed.
- `--max_steps 500` stops automatically, which is useful for smoke tests or recordings.

The markers appear after the LiDAR and proprioceptive histories have warmed up.
Live MAE, RMSE, and maximum absolute error are printed periodically.



## Smoke Tests

Run without `--dataset_path` to train briefly on fake data.

```bash
python scripts/lidar_to_heightmap/terrain_reconstruction_pointnet.py --epochs 1 --num_samples 32
python scripts/lidar_to_heightmap/terrain_reconstruction_transformer.py --epochs 1 --num_samples 32 --max_lidar_points 256
```
