from isaaclab.utils import configclass

from pathlib import Path
from dataclasses import MISSING

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


#AMP Related Stuff
amp_data_path = "./../../../../../../scripts/amp_rl/amp_dataset/"
dataset_names = ["flat", "boxes", "stairs"]
dataset_weights = [1.0, 1.0, 1.0, 1.0]
slow_down_factor = 1.0
discriminator = DiscriminatorCfg(
    hidden_dims=[1024, 512],
    reward_scale=1.0,
    loss_type="BCEWithLogits",
    empirical_normalization= False
)