## Installation Deploy using Conda

1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform) or [pixi](https://pixi.prefix.dev/latest/installation/) - the latter for full reproducibility.

2. create an environment using the file in the folder [deploy/installation](https://github.com/iit-DLSLab/basic-locomotion-dls-isaaclab/tree/main/deploy/installation):


```bash
conda env create -f deploy/installation/mamba_environment_humble.yaml
conda activate basic_locomotion_isaaclab_env

conda env create -f deploy/installation/mamba_environment_lyrical.yaml
conda activate basic_locomotion_isaaclab_env

pixi install --manifest-path deploy/installation/pixi.toml -e humble
pixi shell --manifest-path deploy/installation/pixi.toml -e humble

pixi install --manifest-path deploy/installation/pixi.toml -e lyrical
pixi shell --manifest-path deploy/installation/pixi.toml -e lyrical

```

3. suggestion - add in your .bashrc some alias to run the activate/shell commands. You will run them multiple time!


## Run Sim-to-Sim 

Choose in deploy/config.py the robot/policy you want to run. **Activate your environment conda/pixi**. Then:

```bash
## Sim-to-Sim
python3 deploy/play_mujoco.py


## Sim-to-Sim with ROS2
source deploy/ros2_localhost_connect.sh (TERMINAL 1)
python3 deploy/run_controller_ros2.py (TERMINAL 1) 

source deploy/ros2_localhost_connect.sh (TERMINAL 2)
python3 deploy/run_simulator_ros2.py (TERMINAL 2)

source deploy/ros2_localhost_connect.sh (TERMINAL 3)
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 3)

```

## Run Sim-to-Real

Choose in deploy/config.py the robot/policy you want to run. **Activate your environment conda/pixi**. This script matches well with the repo [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree_ros2_dls/tree/main) that you can use to control unitree go2/and soon a2. **If you use it, remember to source first unitree_ros2_connect.bash in every terminal.**

```bash
## Sim-to-Real with ROS2
python3 run_controller_ros2.py (TERMINAL 1)

ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 2)
```
