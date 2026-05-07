## Overview

These script integrate with amp-rsl-rl to train a policy with AMP.

## How to use

- To train with Symmetries, modify the related [rsl_rl_ppo_cfg.py](https://github.com/iit-DLSLab/basic-locomotion-dls-isaaclab/blob/devel/source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py) setting *class_name = PPOSymmDataAugmented*

```bash
python scripts/morphosymm_rl/train_symm.py --task=Locomotion-Aliengo-Flat --num_envs=4096 --headless
python scripts/morphosymm_rl/train_symm.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=4096 --headless
```

