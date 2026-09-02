## Installation Train

1. Install Isaac Lab by following the [installation guide](https://github.com/isaac-sim/IsaacLab). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

2. Install git for very large file
```bash
sudo apt install git-lfs
```

3. Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)


4. Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/basic_locomotion_isaaclab
```

5. If you want to play with [Morphologycal Symmetries](https://arxiv.org/pdf/2403.17320), install the repo [morphosymm-rl](https://github.com/iit-DLSLab/morphosymm-rl)

6. If you want to play with [Adversarial Motion Priors](https://arxiv.org/pdf/2104.02180), install the repo [amp-rsl-rl](https://github.com/ami-iit/amp-rsl-rl) from the [AMI](https://github.com/ami-iit) research lab.

## Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Flat --num_envs=4096
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=4096
```

- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=Locomotion-Aliengo-Flat --num_envs=16 --visualizer newton
python scripts/rsl_rl/play.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=16 --visualizer newton
```



## Use AMP, Morphological Symmetries, DAGGER or Depth to Heightmap
Each of these modules has a specific README in its own script folder.


## Run Hyperparameter Search

Both tuners log each trial to W&B while retaining local TensorBoard event files
for Ray Tune metric collection. Authenticate once with `wandb login` before
starting a sweep.

Before proceeding, install this dependencies in your isaaclab env:
```bash
pip install pyarrow
pip install optuna

```

### PPO

In local mode, the tuner starts and owns the local Ray runtime; no separate Ray
process is needed.

```bash
python source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/ppo_tuner.py \
  --run_mode local \
  --cfg_file source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/ppo_tuning_cfg.py \
  --cfg_class LocomotionAliengoFlatPPOTuner
```

### FlashSAC

FlashSAC uses a separate tuner and sweep configuration so its off-policy search
space and process handling remain independent of PPO:

```bash
python source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/flashsac_tuner.py \
  --run_mode local \
  --cfg_file source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/flashsac_tuning_cfg.py \
  --cfg_class LocomotionGo2RoughVisionFlashSACTuner \
  --num_samples 20
```

Pass `--ray_address auto` (or an explicit Ray address) only when connecting to
an already-running cluster.
