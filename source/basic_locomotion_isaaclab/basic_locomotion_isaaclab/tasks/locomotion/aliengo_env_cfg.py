import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, MultiMeshRayCasterCameraCfg, TiledCameraCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.markers.config import VisualizationMarkersCfg

from basic_locomotion_isaaclab.assets.aliengo_asset import ALIENGO_CFG, CAMERA_USD_CFG
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
    

    scale_all_joint_friction_model = EventTerm(
        func=custom_events.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
                "friction_distribution_params": (0.2, 2.0),
                "operation": "scale"},
    )


    scale_all_joint_armature_model = EventTerm(
        func=custom_events.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
                "armature_distribution_params": (0.0, 1.0),
                "operation": "scale"},
    )
    
    """randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
            "friction_distribution_params": (0.2, 2.0),
            "armature_distribution_params": (0.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )"""

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (-5.0, 5.0),
            "damping_distribution_params": (-1.0, 1.0),
            "operation": "add",
            "distribution": "uniform",
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
class AliengoFlatEnvCfg(DirectRLEnvCfg):
   # env
    episode_length_s = 20.0
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


    observation_base_linear_scale = 1.0
    observation_base_ang_vel_scale = 1.0
    observation_joint_vel_scale = 0.1


    use_imu = False

    use_concurrent_state_est = True
    if(use_concurrent_state_est):
        concurrent_state_est_network_type = "mlp" # "mlp" or "tcn"
        
        concurrent_state_est_output_space = 3 #lin_vel_b
        
        single_concurrent_state_est_observation_space = 3 # base linear velocity
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
        rma_use_latent_space = False
        if(rma_use_latent_space):
            rma_latent_space = 8
            rma_latent_encoder_hidden_features = 128
            rma_latent_encoder_seed = 0
        
        rma_privileged_observation_space = 12 # P gain
        rma_privileged_observation_space += 12 # D gain
        rma_privileged_observation_space += 3 # clean linear velocity
        rma_privileged_observation_space += 1 # base height error
        rma_privileged_observation_space += 1 # terrain pitch
        rma_privileged_observation_space += 4 # foot contacts

        rma_output_space = rma_latent_space if rma_use_latent_space else rma_privileged_observation_space
        observation_space += rma_output_space

        single_rma_observation_space = 3 # base linear velocity
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


    # asymmetric ppo
    use_asymmetric_ppo = True
    if(use_asymmetric_ppo):
        state_space = observation_space
        state_space += 12 #P gain
        state_space += 12 #D gain
        state_space += 2 #base pitch and height
        state_space += 3 #clean lin vel b
        state_space += 4 #contacts foot
    else:
        state_space = 0


    use_amp = False


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
        #physx=PhysxCfg(
        #    gpu_max_rigid_contact_count=2**20,
        #    gpu_max_rigid_patch_count=2**24,
        #),
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

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        #attach_yaw_only=True,
        ray_alignment='yaw',
        #pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.4, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # an imu sensor in case we don't want any state estimator
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/base", debug_vis=False)


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
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # robot
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    desired_joints_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                           'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',  
                           'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    # Desired tracking variables
    desired_base_height = 0.35
    desired_feet_height = 0.05

    # Desired clip actions
    desired_clip_actions = 3.0
    

    # Tracking reward scale
    lin_vel_reward_scale = 2.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.25
    orientation_reward_scale = -5.0
    height_reward_scale = 1.0
    

    # Joint reward scale
    joints_torque_reward_scale = -2.5e-6 
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
    feet_air_time_reward_scale = 0.5 * 0.0

    feet_height_clearance_reward_scale = 0.25 * 0.0  
    feet_height_clearance_periodic_reward_scale = 0.25
    
    feet_height_clearance_mujoco_reward_scale = 0.25 * 0.0
    feet_height_clearance_mujoco_periodic_reward_scale = 0.25 * 0.0
    
    feet_slide_reward_scale = -0.25 * 0.0
    feet_to_base_distance_reward_scale = 0.25 * 0.0
    
    feet_to_hip_distance_reward_scale = 1.5
    # This is used in loocmotion_env.py for the above reward
    desired_hip_offset = 0.083

    feet_edge_reward_scale = 0.0
    feet_edge_height_threshold = 0.05
    feet_edge_horizontal_radius = 0.10
    feet_edge_radius_px = 0
    visualize_edge_map = False

    feet_vertical_surface_contacts_reward_scale = -0.25


    # Contact suggestion reward scale
    periodic_contact_suggestion_reward_scale =  0.25
    # Desired step freq and duty factor (if periodic gait contact suggestion is used)
    desired_step_freq = 1.4
    desired_duty_factor = 0.65
    desired_phase_offset = [0.0, 0.5, 0.5, 0.0] #FL, FR, RL, RR

    stance_contact_suggestion_reward_scale = 0.25



import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
@configclass
class AliengoRoughBlindEnvCfg(AliengoFlatEnvCfg):

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
                proportion=0.1, num_bars=10, bar_width_range=(0.15, 0.20), bar_height_range=(0.05, 0.15), platform_width=2.0,
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
                proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
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
class AliengoRoughVisionEnvCfg(AliengoRoughBlindEnvCfg):

    def __post_init__(self) -> None:
        height_map_x_points = int(round(self.height_scanner2.pattern_cfg.size[0] / self.height_scanner2.pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(self.height_scanner2.pattern_cfg.size[1] / self.height_scanner2.pattern_cfg.resolution)) + 1
        self.observation_space = self.observation_space + height_map_x_points * height_map_y_points

        self.feet_edge_reward_scale = -1.0

    use_vision = True

    # we add a height scanner for perceptive locomotion
    height_scanner2 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # we add a height scanner for feet edge reward
    height_scanner3 = RayCasterCfg(
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
