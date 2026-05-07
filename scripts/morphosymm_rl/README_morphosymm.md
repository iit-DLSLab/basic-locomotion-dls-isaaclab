## Overview

These script integrate with morphosymm-rl to train a policy with symmetries.


## How to use Depth to Heightmap

1. First, you need to train a locomotion policy

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Go2-Rough-Vision --num_envs=4096 --headless
```

2. Launch the train_dagger.py script. This will load the latest policy you trained in step 1.

```bash
python scripts/depth_to_heightmap/collect_depth_to_heightmap.py --task=Locomotion-Go2-Rough-Vision --num_envs=8192 --headless
```

3. Train the network
```bash
python scripts/depth_to_heightmap/terrain_reconstruction_cnn.py
python scripts/depth_to_heightmap/terrain_reconstruction_transformer.py
```

## Deploy

Currently, the deploy folder does not contain any utils for this.