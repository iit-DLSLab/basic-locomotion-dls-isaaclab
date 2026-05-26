

## Installation Deploy using Conda

1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform)

2. create an environment using the file in the folder [deploy/installation](https://github.com/iit-DLSLab/basic-locomotion-dls-isaaclab/tree/main/deploy/installation):


```bash
conda env create -f mamba_environment_ros1.yaml
conda activate basic_locomotion_isaaclab_ros1_env

conda env create -f mamba_environment_ros2.yaml
conda activate basic_locomotion_isaaclab_ros2_env
```

## Installation Deploy using Docker

1. install docker and run

```bash
docker build -t basic_locomotion_isaaclab_image .
```

2. put in your .bashrc the following alias
```bash
alias basic_locomotion_isaaclab_docker='
if [ ! "$(docker ps -a -q -f name=basic_locomotion_isaaclab_container)" ]; then
   xhost + && docker run -it --rm -v absolute_path_to_this_repo:/home/ -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/input/ -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e QT_X11_NO_MITSHM=1 --gpus all --net host --cap-add=sys_nice --name basic_locomotion_isaaclab_container basic_locomotion_isaaclab_image; \
else
   docker exec -it basic_locomotion_isaaclab_container bash; \
fi'
```


## Run Sim-to-Sim 

Choose in deploy/config.py the robot/policy you want to run. Then:

```bash
## Sim-to-Sim
python3 deploy/play_mujoco.py


## Sim-to-Sim with ROS2
cd deploy/ros2_ws (TERMINAL 1)
colcon build (TERMINAL 1)
source install/setup.bash (TERMINAL 1)

source deploy/ros2_localhost_connect.sh (TERMINAL 1)
python3 deploy/run_controller_ros2.py (TERMINAL 1) 

source install/setup.bash (TERMINAL 2)
source deploy/ros2_localhost_connect.sh (TERMINAL 2)
python3 deploy/run_simulator_ros2.py (TERMINAL 2)

source deploy/ros2_localhost_connect.sh (TERMINAL 3)
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 3)

```

## Run Sim-to-Real

Choose in deploy/config.py the robot/policy you want to run. This script matches well with the repo [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree_ros2_dls/tree/main) that you can use to control unitree go2/and soon a2. **If you use it, remember to source first ros2_connect.sh in every terminal.**

```bash
## Sim-to-Real with ROS2
cd deploy/ros2_ws (TERMINAL 1)
colcon build (TERMINAL 1)
source install/setup.bash (TERMINAL 1)

python3 run_controller_ros2.py (TERMINAL 1)

ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 2)
```
