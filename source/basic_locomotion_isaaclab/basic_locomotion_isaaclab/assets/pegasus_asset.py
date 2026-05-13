# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from basic_locomotion_isaaclab.actuators import IdentifiedActuatorElectricCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg

from basic_locomotion_isaaclab.assets import ISAAC_ASSET_DIR


# Aliengo robot configuration from mujoco
stiffness_mujoco = 100.0
damping_mujoco = 5.0
friction_static_mujoco = 0.2
friction_dynamic_mujoco = 0.6
armature_mujoco = 0.01

static_friction_hip = 0.5 * 0.0
dynamic_friction_hip = 0.3 * 0.0
viscous_friction_hip = 0.3 * 0.0

static_friction_thigh = 0.5 * 0.0
dynamic_friction_thigh = 0.3 * 0.0
viscous_friction_thigh = 0.3 * 0.0

static_friction_calf = 0.5 * 0.0
dynamic_friction_calf = 0.3 * 0.0
viscous_friction_calf = 0.3 * 0.0

PEGASUS_HIP_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_hip_joint"],
    effort_limit=44.4,
    velocity_limit=21.0,
    saturation_effort=44.4,
    stiffness=stiffness_mujoco,
    damping=damping_mujoco,
    armature=armature_mujoco,
    friction_static=friction_static_mujoco,
    activation_vel=0.1,
    friction_dynamic=friction_dynamic_mujoco,

    friction = static_friction_hip,
    dynamic_friction = dynamic_friction_hip,
    viscous_friction = viscous_friction_hip,
)

PEGASUS_THIGH_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_thigh_joint"],
    effort_limit=44.4,
    velocity_limit=21.0,
    saturation_effort=44.4,
    stiffness=stiffness_mujoco,
    damping=damping_mujoco,
    armature=armature_mujoco,
    friction_static=friction_static_mujoco,
    activation_vel=0.1,
    friction_dynamic=friction_dynamic_mujoco,

    friction = static_friction_thigh,
    dynamic_friction = dynamic_friction_thigh,
    viscous_friction = viscous_friction_thigh,
)

PEGASUS_CALF_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_calf_joint"],
    effort_limit=44.4,
    velocity_limit=21.0,
    saturation_effort=44.4,
    stiffness=stiffness_mujoco,
    damping=damping_mujoco,
    armature=armature_mujoco,
    friction_static=friction_static_mujoco,
    activation_vel=0.1,
    friction_dynamic=friction_dynamic_mujoco,

    friction = static_friction_calf,
    dynamic_friction = dynamic_friction_calf,
    viscous_friction = viscous_friction_calf,
)


PEGASUS_CFG = ArticulationCfg(
    prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/pegasus_asset/from_xml/pegasus.usd",
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
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={"hip": PEGASUS_HIP_ACTUATOR_CFG, "thigh": PEGASUS_THIGH_ACTUATOR_CFG, "calf": PEGASUS_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)

