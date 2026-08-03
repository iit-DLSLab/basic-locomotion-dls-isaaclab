## Overview

These script integrate with amp-rsl-rl to train a policy with AMP.

## How to use AMP

- To train with AMO, modify the related [amp_cfg.py](./../../source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/locomotion/agents/amp_cfg.py)

```bash
python scripts/amp_rl/train_amp.py --task=Locomotion-Aliengo-Flat --num_envs=4096 --headless
python scripts/amp_rl/train_amp.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=4096 --headless
python scripts/amp_rl/train_amp.py --task=Locomotion-Aliengo-Rough-Vision --num_envs=4096 --headless
```


TODO

