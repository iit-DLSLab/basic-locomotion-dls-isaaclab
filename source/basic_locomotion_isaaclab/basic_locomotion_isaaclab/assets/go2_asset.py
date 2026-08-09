# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from basic_locomotion_isaaclab.actuators import IdentifiedActuatorElectricCfg, PaceDCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from basic_locomotion_isaaclab.assets import ISAAC_ASSET_DIR

armature = [0.017503729090094566, 0.02336602471768856, 0.03732568025588989, 0.022362561896443367, 0.021544823423027992, 0.03847520425915718, 0.020267240703105927, 0.024585673585534096, 0.03516065329313278, 0.021125180646777153, 0.020762519910931587, 0.036424197256565094]
viscous_friction = [0.20631209015846252, 0.21986860036849976, 0.23434531688690186, 0.24403011798858643, 0.1983535885810852, 0.2536759078502655, 0.2501899302005768, 0.24066618084907532, 0.46991124749183655, 0.29485926032066345, 0.22589823603630066, 0.21030187606811523]
dynamic_friction = [0.288801908493042, 0.2260836958885193, 0.9928773045539856, 0.21979475021362305, 0.243381530046463, 0.4719296097755432, 0.17030715942382812, 0.17484217882156372, 0.9999997615814209, 0.2757018804550171, 0.2683635354042053, 0.7479071617126465]
bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
delay = 2

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
    max_delay=delay,  # max delay in simulation steps
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
    max_delay=delay,  # max delay in simulation steps
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
    max_delay=delay,  # max delay in simulation steps
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
