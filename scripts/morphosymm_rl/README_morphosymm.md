## Overview

These script integrate with morphosymm-rl to train a policy with symmetries.


## How to use Morphologycal Symmetries

- To train a locomotion policy modify the related cfg file [morphosymm_cfg](./../../source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/locomotion/agents/morphosymm_cfg.py) with the needed observation.

- Check the [how to](https://github.com/iit-DLSLab/morphosymm-rl/blob/main/README_how_to.md) first!!!

- Train with

```bash
python scripts/morphosymm_rl/train_symm.py --task=Locomotion-Go2-Flat --num_envs=4096 --headless
python scripts/morphosymm_rl/train_symm.py --task=Locomotion-Go2-Rough-Blind --num_envs=4096 --headless
python scripts/morphosymm_rl/train_symm.py --task=Locomotion-Go2-Rough-Vision --num_envs=4096 --headless
```

- Play with

```bash
python scripts/morphosymm_rl/play_symm.py --task=Locomotion-Go2-Flat --num_envs=4096 --headless
python scripts/morphosymm_rl/play_symm.py --task=Locomotion-Go2-Rough-Blind --num_envs=4096 --headless
python scripts/morphosymm_rl/play_symm.py --task=Locomotion-Go2-Rough-Vision --num_envs=4096 --headless
```