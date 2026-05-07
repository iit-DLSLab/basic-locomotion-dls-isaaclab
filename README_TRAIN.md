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
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Flat --num_envs=4096 --headless
python scripts/rsl_rl/train.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=4096 --headless
```

- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=Locomotion-Aliengo-Flat --num_envs=16
python scripts/rsl_rl/play.py --task=Locomotion-Aliengo-Rough-Blind --num_envs=16
```



## Use AMP, Morphological Symmetried, DAGGER or Depth to Heightmap
Each of these modules as a specific README in its own script folder.


## Run Hyperparameter Search

```bash
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | python3 (TERMINAL 1)
```

```bash
python3 ../basic_locomotion_isaaclab/exts/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/tuner.py --run_mode local --cfg_file ../basic_locomotion_isaaclab/exts/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/locomotion_aliengo_cfg.py --cfg_class LocomotionAliengoFlatTuner (TERMINAL 2)
```


## Convert XML to USD
We use model from [gym-quadruped](https://github.com/iit-DLSLab/gym-quadruped).

```bash
./isaaclab.sh -p scripts/tools/convert_mjcf.py   ../basic_locomotion_isaaclab/scripts/sim_to_sim_mujoco/gym-quadruped/gym_quadruped/robot_model/aliengo/aliengo.xml   ../aliengo.usd   --import-sites   --make-instanceable
```

Remember to set in the application above, "set as default prim" to the root of the robot. Furthermore, for now, add the following lines in the xml of your robots to make the feet seen as body

```bash
<body name="FL_foot" pos="0 0 -0.25">
    <!-- FL_foot only collision -->
    <geom name="FL" class="collision" size="0.0265" pos="0 0 0" />
</body>
```


## Good to know

If you have speed problem in training, may be due to cylinder collision. Then add

```bash
--kit_args="--/physics/collisionApproximateCylinders=true"
```
