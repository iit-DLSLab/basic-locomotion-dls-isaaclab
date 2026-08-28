## Overview

The current implementation of DAGGER is tailored for distill a policy that has heightmap as input, to a policy that use depth. 

## How to use DAGGER

1. First, you need to train a locomotion policy

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Go2-Rough-Vision --num_envs=4096
```

2. Then, launch the train_dagger.py script. This will load the latest policy you trained in step 1.

```bash
python scripts/dagger/train_dagger.py --task=Locomotion-Go2-Rough-Vision --num_envs=8192
```

3. Other hyperparameter can be found the train_dagger.py script


## Deploy

Currently, the deploy folder does not contain any utils for this.