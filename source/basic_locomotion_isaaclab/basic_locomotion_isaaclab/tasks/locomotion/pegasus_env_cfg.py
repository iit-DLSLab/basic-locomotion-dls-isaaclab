import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, MultiMeshRayCasterCameraCfg, TiledCameraCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from basic_locomotion_isaaclab.assets.pegasus_asset import PEGASUS_CFG 
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import basic_locomotion_isaaclab.tasks.custom_events as custom_events
import basic_locomotion_isaaclab.tasks.custom_curriculums as custom_curriculums

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    
    
    randomize_joint_parameters = EventTerm(
        func=custom_events.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_pace_actuator_delay = EventTerm(
        func=custom_events.randomize_pace_actuator_delay,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "min_delay": 0,
        },
    )
    
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                                   "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
    )


@configclass
class PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=2**23)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=2**23)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default = newton_mjwarp



@configclass
class PegasusFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    terrain_curriculum_move_up_error_percent = 20.0
    terrain_curriculum_move_down_error_percent = 50.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 3 # base linear velocity
    observation_space += 3 # base angular velocity  
    observation_space += 3 # projected gravity in base frame
    observation_space += 3 # command (desired linear vel in x and y, desired yaw rate)
    observation_space += 12 # joint positions
    observation_space += 12 # joint velocities
    observation_space += 12 # last actions

    use_clock_signal = True
    if(use_clock_signal):
        observation_space += 4 # clock signal for periodic gait

    single_observation_space = observation_space # Usefull for concatenating history

    # observation history
    use_observation_history = True
    if(use_observation_history):
        history_length = 5
        observation_space *= history_length
    else:
        history_length = 1

    use_imu = False
    # an imu sensor in case we don't want any state estimator (for now we can't use sites from the xml)
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base", 
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0)
        ), 
        debug_vis=False)

    
    use_concurrent_state_est = False
    if(use_concurrent_state_est):
        concurrent_state_est_network_type = "tcn" # "mlp" or "tcn"
        
        concurrent_state_est_output_space = 3 #lin_vel_b
        
        single_concurrent_state_est_observation_space = 3 # base linear acceleration
        single_concurrent_state_est_observation_space += 3 # base angular velocity  
        single_concurrent_state_est_observation_space += 3 # projected gravity in base frame
        single_concurrent_state_est_observation_space += 3 # command (desired linear vel in x and y, desired yaw rate)
        single_concurrent_state_est_observation_space += 12 # joint positions
        single_concurrent_state_est_observation_space += 12 # joint velocities
        single_concurrent_state_est_observation_space += 12 # last actions
        concurrent_state_est_history_length = 5 
        concurrent_state_est_observation_space = single_concurrent_state_est_observation_space*concurrent_state_est_history_length
        
        concurrent_state_est_batch_size = 512
        concurrent_state_est_train_epochs = 1000
        concurrent_state_est_lr = 1e-3
        concurrent_state_est_ep_saving_interval = 1000
        concurrent_state_est_ep_saving_start = 6000


    use_rma = False
    if(use_rma):
        rma_network_type = "mlp" # "mlp" or "tcn"
        rma_use_latent_space = True
        if(rma_use_latent_space):
            rma_latent_space = 8
            rma_latent_encoder_hidden_features = 128
            rma_latent_encoder_seed = 0
        
        rma_privileged_observation_space = 12 # P gain
        rma_privileged_observation_space += 12 # D gain
        rma_privileged_observation_space += 12 # static friction
        rma_privileged_observation_space += 12 # viscous friction

        rma_output_space = rma_latent_space if rma_use_latent_space else rma_privileged_observation_space
        observation_space += rma_output_space

        single_rma_observation_space = 3 # base linear acceleration
        single_rma_observation_space += 3 # base angular velocity  
        single_rma_observation_space += 3 # projected gravity in base frame
        single_rma_observation_space += 3 # command (desired linear vel in x and y, desired yaw rate)
        single_rma_observation_space += 12 # joint positions
        single_rma_observation_space += 12 # joint velocities
        single_rma_observation_space += 12 # last actions
        rma_history_length = 5
        rma_observation_space = single_rma_observation_space*rma_history_length
    
        rma_batch_size = 512
        rma_train_epochs = 1000
        rma_lr = 1e-3
        rma_ep_saving_interval = 1000
        rma_ep_saving_start = 6000


    # Base-centered height scanner for pose-related rewards and privileged observations.
    pose_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Template copied onto each foot link to measure the terrain immediately around that foot.
    foot_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


    # asymmetric ppo
    use_asymmetric_ppo = True
    if(use_asymmetric_ppo):
        state_space = observation_space
        state_space += 12 #P gain
        state_space += 12 #D gain
        state_space += 2 #base pitch and height
        state_space += 3 #clean lin vel b
        state_space += 4 #contacts foot
        state_space += 8 #feet air and contact time
        state_space += 4 #foot error

        pattern_cfg = pose_height_scanner.pattern_cfg
        height_map_x_points = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        state_space += height_map_x_points * height_map_y_points
    else:
        state_space = 0


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=PhysicsCfg(),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()


    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
    )

    # robot
    robot: ArticulationCfg = PEGASUS_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    desired_joints_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                           'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',  
                           'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    
    # If you want to use AMP
    use_amp = False

    # Desired tracking variables
    desired_base_height = 0.60
    desired_feet_height = 0.05


    # Desired clip actions
    desired_clip_actions = 3.0
    use_filter_actions = True
        

    # Tracking reward scale
    lin_vel_reward_scale = 2.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.25
    orientation_reward_scale = -5.0
    height_reward_scale = 1.0
    

    # Joint reward scale
    joints_torque_reward_scale = -2.5e-7
    joints_accel_reward_scale = -2.5e-7
    joints_energy_reward_scale = -1e-4
    joints_hip_position_reward_scale = -0.1 * 0.0
    joints_thigh_position_reward_scale = -0.1 * 0.0
    joints_calf_position_reward_scale = -0.001 * 0.0
   
    
    # Undesired contacts reward scale
    undersired_contact_reward_scale = -1.0
    action_rate_reward_scale = -0.01
    action_smoothness_reward_scale = -0.001


    # Feet reward scale
    feet_height_clearance_aperiodic_reward_scale = 0.25 * 0.0  
    feet_height_clearance_periodic_reward_scale = 0.25 * 0.0
    
    feet_height_clearance_mujoco_aperiodic_reward_scale = 0.25 * 0.0
    feet_height_clearance_mujoco_periodic_reward_scale = 0.25# * 0.0
    
    feet_slide_reward_scale = -0.25 * 0.0
    
    feet_to_hip_distance_reward_scale = 1.5
    # This is used in loocmotion_env.py for the above reward
    desired_hip_offset = 0.095

    feet_edge_reward_scale = 0.0
    feet_edge_height_threshold = 0.05
    feet_edge_horizontal_radius = 0.10
    feet_edge_radius_px = 0
    visualize_edge_map = False

    feet_vertical_surface_contacts_reward_scale = -2.5

    # variables used in feet air time and periodic contact suggestion reward
    desired_step_freq = 1.4 
    desired_duty_factor = 0.65

    feet_air_time_reward_scale = 0.25
    feet_air_time_variance_reward_scale = -1.0*0.0

    periodic_contact_suggestion_reward_scale = 0.5
    desired_step_freq_max = 1.8
    step_freq_vel_norm_low = 0.4
    step_freq_vel_norm_high = 0.8
    desired_phase_offset = [0.0, 0.5, 0.5, 0.0] #FL, FR, RL, RR

    stance_contact_suggestion_reward_scale = 1.0



import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
@configclass
class PegasusRoughBlindEnvCfg(PegasusFlatEnvCfg):

    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        curriculum=False,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.2
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.10), platform_width=2.0,
            ),
            "star": terrain_gen.MeshStarTerrainCfg(
                proportion=0.1, num_bars=10, bar_width_range=(0.15, 0.20), bar_height_range=(0.05, 0.13), platform_width=2.0,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.13), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.13), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
        },
    )

    """Rough terrains configuration."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )



@configclass
class PegasusRoughVisionEnvCfg(PegasusRoughBlindEnvCfg):

    def __post_init__(self) -> None:
        pattern_cfg = self.perceptive_height_scanner.pattern_cfg
        height_map_x_points = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        self.observation_space = self.observation_space + height_map_x_points * height_map_y_points

        self.feet_edge_reward_scale = -1.0

    use_vision = True

    # we add a height scanner for perceptive locomotion
    perceptive_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # we add a height scanner for feet edge reward
    edge_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.8, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    #camera_usd = CAMERA_USD_CFG
    use_depth_camera = False
    depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.33, 0.0, 0.08), rot=(-0.405579, 0.579228, -0.579228, 0.405579)),
        mesh_prim_paths=[
            "/World/ground",
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/base/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FL_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FR_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RL_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RR_.*/visuals"),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=120,
            width=240,
        ),
        debug_vis=True,
    )
