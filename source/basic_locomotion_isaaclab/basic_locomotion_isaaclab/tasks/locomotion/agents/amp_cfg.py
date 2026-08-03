from isaaclab.utils import configclass

from dataclasses import MISSING
from pathlib import Path

@configclass
class DiscriminatorCfg:
    """Configuration for the discriminator network."""

    class_name: str = "Discriminator"
    """The discriminator class name. Default is Discriminator."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""

    reward_scale: float = MISSING
    """The reward coefficient."""

    loss_type: str = "BCEWithLogits"
    """The type of loss to use for training the discriminator. Default is BCEWithLogits."""
    
    empirical_normalization: bool = False
    """Whether to use empirical normalization for the discriminator inputs. Default is False."""


# AMP dataset configuration consumed by AMPOnPolicyRunner.
dataset = {
    "amp_data_path": str(Path(__file__).resolve().parents[6] / "scripts" / "amp_rl" / "amp_dataset"),
    "datasets": {
        "flat": 1.0,
        "boxes": 1.0,
        "stairs": 1.0,
    },
    "slow_down_factor": 1.0,
    "amp_joint_names": [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ],
    "velocity_representation": "body_fixed",
}

discriminator = DiscriminatorCfg(
    hidden_dims=[512, 256],
    reward_scale=1.0,
    loss_type="BCEWithLogits",
    empirical_normalization= False
)