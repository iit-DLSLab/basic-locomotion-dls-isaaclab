from isaaclab.utils import configclass

from pathlib import Path
from dataclasses import MISSING

@configclass
class MorphologycalSymmetriesCfg:
    """Configuration for using morphosymm-rl."""

    class_name: str = "MorphologycalSymmetries"
    """The class name."""

    obs_space_names_actor =  None
    """The observation space names for the actor network."""

    obs_space_names_critic = None
    """The observation space names for the critic network."""

    action_space_names = None
    """The action space names."""

    joints_order = None
    """The order of the joints in the robot."""

    robot_name = None
    """The name of the robot to use inside Morphosymm."""

    schedule_fixed_to_adaptive_switch = None
    """The number of iterations to switch from fixed to adaptive schedule for the symmetry loss. 
    If None, then no switch will happen. If the scheduler is set to adaptive, not change will be made."""


# Actor OBS
history_length = 5
obs_space_names_actor = [
        "base_lin_vel",
        "base_ang_vel",
        "gravity",
        "des_base_lin_vel_xy",
        "des_base_ang_vel_yaw",
        "joints_pos",
        "joints_vel",
        "joints_pos",
        "clock_data",
    ]*int(history_length)
#obs_space_names_actor += ["heightmap:7x9"]


# Critic OBS
obs_space_names_critic = [
        "base_lin_vel",
        "base_ang_vel",
        "gravity",
        "des_base_lin_vel_xy",
        "des_base_ang_vel_yaw",
        "joints_pos",
        "joints_vel",
        "joints_pos",
        "clock_data",
    ]*int(history_length)
#obs_space_names_critic += ["heightmap:7x9"]
obs_space_names_critic += [    
        "clock_data", "clock_data", "clock_data",  # P gains
        "clock_data", "clock_data", "clock_data",  # D gains
]
obs_space_names_critic += ["base_lin_vel", "invariant_scalar", "invariant_scalar", "clock_data"]
obs_space_names_critic += ["heightmap:4x4"]

# Action Space
action_space_names = ["joints_pos"]


# Joints Order
joints_order = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
]


# Robot Name
robot_name = "a1"


morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        action_space_names = action_space_names,
        joints_order = joints_order,
        robot_name = robot_name,
    )