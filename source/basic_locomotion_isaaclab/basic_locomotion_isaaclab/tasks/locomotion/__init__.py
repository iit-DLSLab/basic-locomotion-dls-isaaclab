# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .locomotion_env import LocomotionEnv


# Aliengo environments
from .locomotion_env import AliengoFlatEnvCfg, AliengoRoughVisionEnvCfg, AliengoRoughBlindEnvCfg

gym.register(
    id="Locomotion-Aliengo-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)


# Go2 environments
from .locomotion_env import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

gym.register(
    id="Locomotion-Go2-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Go2-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Go2-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)


# B2 environments
from .locomotion_env import B2FlatEnvCfg, B2RoughVisionEnvCfg, B2RoughBlindEnvCfg

gym.register(
    id="Locomotion-B2-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": B2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-B2-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": B2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-B2-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": B2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

# HyQReal environments
from .locomotion_env import HyQRealFlatEnvCfg, HyQRealRoughVisionEnvCfg, HyQRealRoughBlindEnvCfg

gym.register(
    id="Locomotion-HyQReal-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HyQRealFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-HyQReal-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HyQRealRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-HyQReal-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HyQRealRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

# Pegasus environments
from .locomotion_env import PegasusFlatEnvCfg, PegasusRoughVisionEnvCfg, PegasusRoughBlindEnvCfg

gym.register(
    id="Locomotion-Pegasus-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Pegasus-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Pegasus-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)