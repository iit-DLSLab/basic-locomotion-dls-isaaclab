# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from basic_locomotion_isaaclab.actuators import IdentifiedActuatorElectricCfg, PaceDCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from basic_locomotion_isaaclab.assets import ISAAC_ASSET_DIR

armature = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
viscous_friction = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
dynamic_friction = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


PEGASUS_HIP_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint"],
    saturation_effort=120.0,
    effort_limit=120.0,
    velocity_limit=10.4,
    stiffness={".*": 120.0},  # P gain in Nm/rad
    damping={".*": 5.0},  # D gain in Nm s/rad
    encoder_bias={"FL_hip_joint": bias[0], "FR_hip_joint": bias[3], "RL_hip_joint": bias[6], "RR_hip_joint": bias[9]},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={"FL_hip_joint": dynamic_friction[0], "FR_hip_joint": dynamic_friction[3], "RL_hip_joint": dynamic_friction[6], "RR_hip_joint": dynamic_friction[9]},  # static friction coefficient (Nm)
    dynamic_friction={"FL_hip_joint": dynamic_friction[0], "FR_hip_joint": dynamic_friction[3], "RL_hip_joint": dynamic_friction[6], "RR_hip_joint": dynamic_friction[9]},  # dynamic friction coefficient (Nm)
    viscous_friction={"FL_hip_joint": viscous_friction[0], "FR_hip_joint": viscous_friction[3], "RL_hip_joint": viscous_friction[6], "RR_hip_joint": viscous_friction[9]},  # viscous friction coefficient (Nm s/rad)
    armature={"FL_hip_joint": armature[0], "FR_hip_joint": armature[3], "RL_hip_joint": armature[6], "RR_hip_joint": armature[9]},
    max_delay=1,  # max delay in simulation steps
)


PEGASUS_THIGH_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_thigh_joint"],
    saturation_effort=120.0,
    effort_limit=120.0,
    velocity_limit=10.4,
    stiffness={".*": 120.0},  # P gain in Nm/rad
    damping={".*": 5.0},  # D gain in Nm s/rad
    encoder_bias={"FL_thigh_joint": bias[1], "FR_thigh_joint": bias[4], "RL_thigh_joint": bias[7], "RR_thigh_joint": bias[10]},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={"FL_thigh_joint": dynamic_friction[1], "FR_thigh_joint": dynamic_friction[4], "RL_thigh_joint": dynamic_friction[7], "RR_thigh_joint": dynamic_friction[10]},  # static friction coefficient (Nm)
    dynamic_friction={"FL_thigh_joint": dynamic_friction[1], "FR_thigh_joint": dynamic_friction[4], "RL_thigh_joint": dynamic_friction[7], "RR_thigh_joint": dynamic_friction[10]},  # dynamic friction coefficient (Nm)
    viscous_friction={"FL_thigh_joint": viscous_friction[1], "FR_thigh_joint": viscous_friction[4], "RL_thigh_joint": viscous_friction[7], "RR_thigh_joint": viscous_friction[10]},  # viscous friction coefficient (Nm s/rad)
    armature={"FL_thigh_joint":armature[1], "FR_thigh_joint": armature[4], "RL_thigh_joint": armature[7], "RR_thigh_joint": armature[10]},
    max_delay=1,  # max delay in simulation steps
)


PEGASUS_CALF_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_calf_joint"],
    saturation_effort=210.0,
    effort_limit=210.0,
    velocity_limit=5.5,
    stiffness={".*": 120.0},  # P gain in Nm/rad
    damping={".*": 5.0},  # D gain in Nm s/rad
    encoder_bias={"FL_calf_joint": bias[2], "FR_calf_joint": bias[5], "RL_calf_joint": bias[8], "RR_calf_joint": bias[11]},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={"FL_calf_joint": dynamic_friction[2], "FR_calf_joint": dynamic_friction[5], "RL_calf_joint": dynamic_friction[8], "RR_calf_joint": dynamic_friction[11]},  # static friction coefficient (Nm)
    dynamic_friction={"FL_calf_joint": dynamic_friction[2], "FR_calf_joint": dynamic_friction[5], "RL_calf_joint": dynamic_friction[8], "RR_calf_joint": dynamic_friction[11]},  # dynamic friction coefficient (Nm)
    viscous_friction={"FL_calf_joint": viscous_friction[2], "FR_calf_joint": viscous_friction[5], "RL_calf_joint": viscous_friction[8], "RR_calf_joint": viscous_friction[11]},  # viscous friction coefficient (Nm s/rad)
    armature={"FL_calf_joint": armature[2], "FR_calf_joint": armature[5], "RL_calf_joint": armature[8], "RR_calf_joint": armature[11]},
    max_delay=1,  # max delay in simulation steps
)

PEGASUS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/pegasus_asset/pegasus_temp.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            ".*L_hip_joint": 0.,
            ".*R_hip_joint": 0.,
            ".*_thigh_joint": 0.75,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={"hip": PEGASUS_HIP_ACTUATOR_CFG, "thigh": PEGASUS_THIGH_ACTUATOR_CFG, "calf": PEGASUS_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
