# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from basic_locomotion_isaaclab.actuators import IdentifiedActuatorElectricCfg, PaceDCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from basic_locomotion_isaaclab.assets import ISAAC_ASSET_DIR

armature = [0.015245300717651844, 0.02287333831191063, 0.04345845803618431, 0.02273648791015148, 0.021029189229011536, 0.04288801923394203, 0.020965829491615295, 0.02172032743692398, 0.041628554463386536, 0.016661761328577995, 0.01919594779610634, 0.0420747809112072]
viscous_friction = [0.21183708310127258, 0.23815694451332092, 0.23462027311325073, 0.2426394820213318, 0.21587151288986206, 0.21994808316230774, 0.22393473982810974, 0.23609080910682678, 0.37616264820098877, 0.2554837763309479, 0.24458086490631104, 0.16209274530410767]
dynamic_friction = [0.29261451959609985, 0.2158409059047699, 0.9081975817680359, 0.22919607162475586, 0.19076737761497498, 0.5661238431930542, 0.17274132370948792, 0.16221186518669128, 0.9999995827674866, 0.237972229719162, 0.2591995596885681, 0.7882703542709351]
bias = [0.09999876469373703, 0.09621422737836838, 0.0999980941414833, 0.006213739514350891, 0.0941513404250145, 0.09999898821115494, 0.04979293793439865, 0.09999003261327744, 0.09999995678663254, -0.0999981164932251, 0.041589684784412384, 0.09999736398458481]


GO2_HIP_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint"],
    saturation_effort=23.7,
    effort_limit=23.7,
    velocity_limit=30.1,
    stiffness={".*": 20.0},  # P gain in Nm/rad
    damping={".*": 1.5},  # D gain in Nm s/rad
    encoder_bias={"FL_hip_joint": bias[0], "FR_hip_joint": bias[3], "RL_hip_joint": bias[6], "RR_hip_joint": bias[9]},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={"FL_hip_joint": dynamic_friction[0], "FR_hip_joint": dynamic_friction[3], "RL_hip_joint": dynamic_friction[6], "RR_hip_joint": dynamic_friction[9]},  # static friction coefficient (Nm)
    dynamic_friction={"FL_hip_joint": dynamic_friction[0], "FR_hip_joint": dynamic_friction[3], "RL_hip_joint": dynamic_friction[6], "RR_hip_joint": dynamic_friction[9]},  # dynamic friction coefficient (Nm)
    viscous_friction={"FL_hip_joint": viscous_friction[0], "FR_hip_joint": viscous_friction[3], "RL_hip_joint": viscous_friction[6], "RR_hip_joint": viscous_friction[9]},  # viscous friction coefficient (Nm s/rad)
    armature={"FL_hip_joint": armature[0], "FR_hip_joint": armature[3], "RL_hip_joint": armature[6], "RR_hip_joint": armature[9]},
    max_delay=1,  # max delay in simulation steps
)


GO2_THIGH_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_thigh_joint"],
    saturation_effort=23.7,
    effort_limit=23.7,
    velocity_limit=30.1,
    stiffness={".*": 20.0},  # P gain in Nm/rad
    damping={".*": 1.5},  # D gain in Nm s/rad
    encoder_bias={"FL_thigh_joint": bias[1], "FR_thigh_joint": bias[4], "RL_thigh_joint": bias[7], "RR_thigh_joint": bias[10]},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={"FL_thigh_joint": dynamic_friction[1], "FR_thigh_joint": dynamic_friction[4], "RL_thigh_joint": dynamic_friction[7], "RR_thigh_joint": dynamic_friction[10]},  # static friction coefficient (Nm)
    dynamic_friction={"FL_thigh_joint": dynamic_friction[1], "FR_thigh_joint": dynamic_friction[4], "RL_thigh_joint": dynamic_friction[7], "RR_thigh_joint": dynamic_friction[10]},  # dynamic friction coefficient (Nm)
    viscous_friction={"FL_thigh_joint": viscous_friction[1], "FR_thigh_joint": viscous_friction[4], "RL_thigh_joint": viscous_friction[7], "RR_thigh_joint": viscous_friction[10]},  # viscous friction coefficient (Nm s/rad)
    armature={"FL_thigh_joint":armature[1], "FR_thigh_joint": armature[4], "RL_thigh_joint": armature[7], "RR_thigh_joint": armature[10]},
    max_delay=1,  # max delay in simulation steps
)


GO2_CALF_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_calf_joint"],
    saturation_effort=45.43,
    effort_limit=45.43,
    velocity_limit=15.7,
    stiffness={".*": 20.0},  # P gain in Nm/rad
    damping={".*": 1.5},  # D gain in Nm s/rad
    encoder_bias={"FL_calf_joint": bias[2], "FR_calf_joint": bias[5], "RL_calf_joint": bias[8], "RR_calf_joint": bias[11]},  # encoder bias in radians
    # note: modeling coulomb friction if friction = dynamic_friction
    # > in newer Isaac Sim versions, friction is renamed to static_friction
    friction={"FL_calf_joint": dynamic_friction[2], "FR_calf_joint": dynamic_friction[5], "RL_calf_joint": dynamic_friction[8], "RR_calf_joint": dynamic_friction[11]},  # static friction coefficient (Nm)
    dynamic_friction={"FL_calf_joint": dynamic_friction[2], "FR_calf_joint": dynamic_friction[5], "RL_calf_joint": dynamic_friction[8], "RR_calf_joint": dynamic_friction[11]},  # dynamic friction coefficient (Nm)
    viscous_friction={"FL_calf_joint": viscous_friction[2], "FR_calf_joint": viscous_friction[5], "RL_calf_joint": viscous_friction[8], "RR_calf_joint": viscous_friction[11]},  # viscous friction coefficient (Nm s/rad)
    armature={"FL_calf_joint": armature[2], "FR_calf_joint": armature[5], "RL_calf_joint": armature[8], "RR_calf_joint": armature[11]},
    max_delay=1,  # max delay in simulation steps
)

GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/go2_asset/go2.usd",
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
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.,
            ".*R_hip_joint": 0.,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={"hip": GO2_HIP_ACTUATOR_CFG, "thigh": GO2_THIGH_ACTUATOR_CFG, "calf": GO2_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
