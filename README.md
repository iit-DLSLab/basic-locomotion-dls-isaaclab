<div style="text-align: left;">
  <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-green" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/rsl_rl%20-v3.3.0-brown" alt="rsl-rl v3.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/Mujoco%20-v3.7.0-blue" alt="Mujoco v3.7.0" style="margin-bottom: 1px;">
  <div style="display: flex; justify-content: space-around;">
    <img src="./gifs/train.gif" alt="Train" width="32%">
    <img src="./gifs/sim-to-sim.gif" alt="Sim-to-Sim" width="32%">
    <img src="./gifs/sim-to-real.gif" alt="Sim-to-Real" width="32%">
  </div>
  <div style="display: flex; justify-content: space-around;">
    <img src="./gifs/go2-vision.gif" alt="Train" width="96%">
  </div>
</div>

## Overview

Reinforcement learning implementation of **quadruped locomotion** in IsaacLab. It includes support for different robots available at DLS, with scripts for both sim-to-sim and sim-to-real transfer.

Features:
- [Concurrent State Estimator](https://arxiv.org/pdf/2202.05481)
- [Rapid Motor Adaptation](https://arxiv.org/pdf/2107.04034)
- [Morphological Symmetries](https://arxiv.org/pdf/2403.17320) 
- [Adversarial Motion Priors](https://arxiv.org/pdf/2104.02180) (big thanks to [Giuseppe L'erario](https://github.com/Giulero) and [Giulio Romualdi](https://github.com/giulioromualdi))
- Identification of robot parameters for sim2real using [pace](https://github.com/leggedrobotics/pace-sim2real) via our repo [sim2real-robot-identification](https://github.com/iit-DLSLab/sim2real-robot-identification)
- Sim-to-Sim in [Mujoco](https://github.com/google-deepmind/mujoco)
- Sim-to-Real using ROS2 

Features work in progress - done but not tested:
- Dagger for depth-to-action policy distillation
- Depth-to-Heightmap network
- Lidar-to-Heightmap network


Real-world deployment via:
- [muse](https://github.com/iit-DLSLab/muse/tree/unitree_sdk) for state estimation (if no concurrent state estimation is used)
- [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for unitree robot communication

A list of robots and environments available is described below:

| Robot Model         | Environment Name Pattern                                   |
|---------------------|------------------------------------------------------------|
| [Aliengo](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/aliengo), [Go2](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/go2), [B2](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/b2), [HyQReal2](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/hyqreal2) | Locomotion-**RobotModel**-Flat-Blind <br> Locomotion-**RobotModel**-Rough-Blind <br> Locomotion-**RobotModel**-Rough-Vision |


## Installation and Runs

If you want only to deploy a trained policy on your robot, continue on [README_deploy](./README_deploy.md) otherwise on [README_train](./README_train.md).

**For the train, check first the compatibility with IsaacLab and rsl-rl at the top of this readme. They indicate the releases that we tested.**



## Citing this work

If you find the work useful and you adopt [Morphological Symmetries](https://arxiv.org/pdf/2403.17320), please consider citing one of our works:

#### [Leveraging Symmetry in RL-based Legged Locomotion Control (IROS-2024)](https://arxiv.org/pdf/2403.17320)

```
@inproceedings{suhuang2024leveraging,
  author={Su, Zhi and Huang, Xiaoyu and Ordoñez-Apraez, Daniel and Li, Yunfei and Li, Zhongyu and Liao, Qiayuan and Turrisi, Giulio and Pontil, Massimiliano and Semini, Claudio and Wu, Yi and Sreenath, Koushil},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Leveraging Symmetry in RL-based Legged Locomotion Control}, 
  year={2024},
  pages={6899-6906},
  doi={10.1109/IROS58592.2024.10802439}
}
```

#### [Morphological symmetries in robotics (IJRR-2025)](https://arxiv.org/pdf/2402.15552):

```
@article{ordonez2025morphosymm,
  author = {Daniel Ordoñez-Apraez and Giulio Turrisi and Vladimir Kostic and Mario Martin and Antonio Agudo and Francesc Moreno-Noguer and Massimiliano Pontil and Claudio Semini and Carlos Mastalli},
  title ={Morphological symmetries in robotics},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  volume = {44},
  number = {10-11},
  pages = {1743-1766},
  doi = {10.1177/02783649241282422},
}
```

## How to contribute

PRs are very welcome (search for **TODO** in the issue, or add what you like)!

## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi).
