from isaaclab.utils.configclass import configclass

from pathlib import Path
from dataclasses import MISSING

@configclass
class MorphologycalSymmetriesCfg:
    """Configuration for using morphosymm-rsl-rl."""

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

    state_dependent_std = False
    """Whether the actor emits its action covariance directly, instead of learning a fixed one."""

    small_init_output = True
    """Whether to start the actor mean and the critic value with small (near-zero) initial outputs."""

    symmetric_initialization = True
    """Whether to project the actor/critic initial weights onto the symmetry-equivariant subspace."""

    use_data_augmentation = False
    """Whether to augment the rollout storage with every symmetry-group replica of each collected transition
    (selects PPOSymmDataAugmented). If False, uses the plain equivariant PPO with no augmentation."""


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
obs_space_names_actor += ["heightmap:7x9"]


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
obs_space_names_critic += ["heightmap:7x9"]
obs_space_names_critic += [    
        "clock_data", "clock_data", "clock_data",  # P gains
        "clock_data", "clock_data", "clock_data",  # D gains
]
obs_space_names_critic += [
        "base_lin_vel",  # clean lin vel b
        "invariant_scalar", "invariant_scalar",  # height error, terrain pitch
        "clock_data",  # contacts foot
        "clock_data",  # feet air time
        "clock_data",  # feet contact time
        "clock_data",  # foot error
]
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