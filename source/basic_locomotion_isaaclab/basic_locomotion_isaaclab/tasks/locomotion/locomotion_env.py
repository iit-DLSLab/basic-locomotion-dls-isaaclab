# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
import torch.nn.functional as F

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, RayCasterCamera, RayCasterCameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg, TiledCameraCfg, TiledCamera, patterns, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


from .aliengo_env_cfg import AliengoFlatEnvCfg, AliengoRoughBlindEnvCfg, AliengoRoughVisionEnvCfg
from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg
from .hyqreal_env_cfg import HyQRealFlatEnvCfg, HyQRealRoughVisionEnvCfg, HyQRealRoughBlindEnvCfg
from .b2_env_cfg import B2FlatEnvCfg, B2RoughVisionEnvCfg, B2RoughBlindEnvCfg

from basic_locomotion_isaaclab.tasks.supervised_learning_networks import FrozenRandomMlpEncoder, create_supervised_network

class LocomotionEnv(DirectRLEnv):
    cfg: AliengoFlatEnvCfg | AliengoRoughBlindEnvCfg | AliengoRoughVisionEnvCfg | Go2FlatEnvCfg | Go2RoughVisionEnvCfg | Go2RoughBlindEnvCfg | HyQRealFlatEnvCfg | HyQRealRoughVisionEnvCfg | HyQRealRoughBlindEnvCfg

    def __init__(self, cfg: AliengoFlatEnvCfg | AliengoRoughBlindEnvCfg | AliengoRoughVisionEnvCfg | Go2FlatEnvCfg | Go2RoughVisionEnvCfg | Go2RoughBlindEnvCfg | HyQRealFlatEnvCfg | HyQRealRoughVisionEnvCfg | HyQRealRoughBlindEnvCfg, render_mode: str | None = None, **kwargs):
        self._edge_map_visualizer = None
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Swing peak
        self._swing_peak = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        self._swing_peak_periodic = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        
        # Desired Hip Offset
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        
        # Periodic gait
        self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0


        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)

        # RMA
        if(cfg.use_rma == True):
            self._rma_network = create_supervised_network(
                cfg.rma_observation_space,
                cfg.rma_output_space,
                network_type=getattr(cfg, "rma_network_type", "mlp"),
                sequence_length=cfg.rma_history_length,
            )
            self._rma_network.to(self.device)
            
            if self.cfg.rma_use_latent_space:
                self._rma_latent_encoder = FrozenRandomMlpEncoder(
                    cfg.rma_privileged_observation_space,
                    cfg.rma_output_space,
                    hidden_features=getattr(cfg, "rma_latent_encoder_hidden_features", 128),
                    seed=getattr(cfg, "rma_latent_encoder_seed", 0),
                )
                self._rma_latent_encoder.to(self.device)
            self._observation_history_rma = torch.zeros(self.num_envs, cfg.rma_history_length, cfg.single_rma_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )

        # Learned State Estimator
        if(cfg.use_concurrent_state_est == True):
            self._concurrent_state_est_network = create_supervised_network(
                cfg.concurrent_state_est_observation_space,
                cfg.concurrent_state_est_output_space,
                network_type=getattr(cfg, "concurrent_state_est_network_type", "mlp"),
                sequence_length=cfg.concurrent_state_est_history_length,
            )
            self._concurrent_state_est_network.to(self.device)
            self._observation_history_concurrent_state_est = torch.zeros(self.num_envs, cfg.concurrent_state_est_history_length, cfg.single_concurrent_state_est_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_height_exp",
                "track_lin_vel_xy_exp",
                "track_lin_vel_z_l2",
                "track_orientation_l2",
                "track_ang_vel_xy_l2",
                "track_ang_vel_z_exp",

                "undesired_contacts",
                "action_rate_l2",
                "action_smoothness_l2",
                
                "joints_hip_pos_l2",
                "joints_thigh_pos_l2",
                "joints_calf_pos_l2",
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",
                
                "feet_air_time",
                "feet_height_clearance_periodic",
                "feet_height_clearance",
                "feet_height_clearance_mujoco_periodic",
                "feet_height_clearance_mujoco",
                "feet_slide",
                "feet_to_base_distance_l2",
                "feet_to_hip_distance_l2",
                "feet_edge",
                "feet_vertical_surface_contacts",

                "periodic_contact_suggestion",
                "stance_contact_suggestion",
            ]
        }
        # Get specific body indices
        self._base_contact_sensor_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)
        self._thigh_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"], preserve_order=True)
        self._undesired_contact_body_ids = self._base_contact_sensor_id + self._hip_contact_sensor_ids + self._thigh_contact_sensor_ids

        
        self._feet_ids_robot, _ = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_ids_robot, _ = self._robot.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)

        # Ensure the order is consistent with the one expected in the cfg
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]

        if getattr(self.cfg, "visualize_edge_map", False):
            self.set_debug_vis(True)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for the proprioceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # if we came from depth-based env, we create the depth camera scanner
        if isinstance(self.cfg, AliengoRoughVisionEnvCfg) or isinstance(self.cfg, Go2RoughVisionEnvCfg) or isinstance(self.cfg, HyQRealRoughVisionEnvCfg) or isinstance(self.cfg, B2RoughVisionEnvCfg):
            # we add a height scanner for the proprioceptive locomotion
            self._height_scanner2 = RayCaster(self.cfg.height_scanner2)
            self.scene.sensors["height_scanner2"] = self._height_scanner2

            self._height_scanner3 = RayCaster(self.cfg.height_scanner3)
            self.scene.sensors["height_scanner3"] = self._height_scanner3

        if(getattr(self.cfg, "use_depth_camera", False)):
            self._depth_camera = MultiMeshRayCasterCamera(self.cfg.depth_camera)
            ##self._depth_camera = TiledCamera(self.cfg.depth_camera)
            self.scene.sensors["depth_camera"] = self._depth_camera
            pass

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        default_joint_pos_ordered = self._robot.data.default_joint_pos[:, self._ids_joints_order]
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + default_joint_pos_ordered
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + default_joint_pos_ordered


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._ids_joints_order)



    def _get_observations(self) -> dict:
        
        # This is a custom event, to be moved in custom_events.py
        self._get_new_random_commands()


        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            clock_data = torch.vstack([self._phase_signal[:,0], self._phase_signal[:,1], self._phase_signal[:,2], self._phase_signal[:,3]]).T
            # all the envs that are not moving, we put -1
            should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
            clock_data[:, :] = clock_data[:, :]*should_move.unsqueeze(1).expand(-1, 4) + -1.0* ~should_move.unsqueeze(1).expand(-1, 4)
            

        # Choosing the main source of observation
        if(self.cfg.use_concurrent_state_est):
            # If concurrent SE/Learned State Estimator, we predict linear and angular vel from IMU
            base_linear = self._get_concurrent_state_estimation()
            base_ang_vel = self._imu.data.ang_vel_b
            projected_gravity_b = self._imu.data.projected_gravity_b
        elif(self.cfg.use_imu):
            # Using directly the IMU
            base_linear = self._imu.data.lin_acc_b
            base_ang_vel = self._imu.data.ang_vel_b
            projected_gravity_b = self._imu.data.projected_gravity_b
        else:
            #Using a model-based state estimation
            base_linear = self._robot.data.root_lin_vel_b
            base_ang_vel = self._robot.data.root_ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b
        
        
        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    base_linear * self.cfg.observation_base_linear_scale,
                    base_ang_vel * self.cfg.observation_base_ang_vel_scale,
                    projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order] * self.cfg.observation_joint_vel_scale,
                    self._actions,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)


        observations = {"common": obs}



        # Add heightmap data to obs if needed
        if isinstance(self.cfg, AliengoRoughVisionEnvCfg) or isinstance(self.cfg, Go2RoughVisionEnvCfg) or isinstance(self.cfg, HyQRealRoughVisionEnvCfg) or isinstance(self.cfg, B2RoughVisionEnvCfg):
            height_data = (
                self._height_scanner2.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner2.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)   



        # If RMA, we add some other predicted obs
        if(self.cfg.use_rma):
            # Predict the RMA observation
            obs_rma = self._get_rma()
            obs = torch.cat((obs, obs_rma), dim=-1)


        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            obs_critic = self._get_privileged_observation()
            observations["critic"] = torch.cat((obs, obs_critic), dim=-1)
        else:
            observations["critic"] = obs


        # Actor OBS - here after the critic to avoid duplication with rma obs
        # if asymmetric ppo is used
        observations["policy"] = obs    
        # ------------------------------------------------------------------------------------------

        # AMP related observation if used
        if(self.cfg.use_amp):
            obs_amp = torch.cat(
                [
                    tensor
                    for tensor in (
                        #self._robot.data.root_quat_w,
                        self._robot.data.joint_pos[:, self._ids_joints_order],
                        self._robot.data.joint_vel[:, self._ids_joints_order],
                        self._robot.data.root_lin_vel_b,
                        self._robot.data.root_ang_vel_b,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
            observations["amp"] = obs_amp

        # --------------------------------------------------------------------------------------------

        return observations


    def _has_edge_map(self) -> bool:
        return hasattr(self, "_height_scanner3")


    def _compute_edge_map(self) -> tuple[torch.Tensor, float, int, int]:
        """Compute the scanner grid cells that are too close to a height discontinuity."""

        height_data_scanner = self._height_scanner3.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

        height_map_resolution = self._height_scanner3.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner3.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner3.cfg.pattern_cfg.size[1] / height_map_resolution)) + 1
        height_grid = height_data_scanner.reshape(self.num_envs, height_map_y_points, height_map_x_points)

        edge_map = torch.zeros_like(height_grid, dtype=torch.bool)

        x_edges = torch.abs(height_grid[:, :, 1:] - height_grid[:, :, :-1]) > self.cfg.feet_edge_height_threshold
        edge_map[:, :, :-1] |= x_edges
        edge_map[:, :, 1:] |= x_edges

        y_edges = torch.abs(height_grid[:, 1:, :] - height_grid[:, :-1, :]) > self.cfg.feet_edge_height_threshold
        edge_map[:, :-1, :] |= y_edges
        edge_map[:, 1:, :] |= y_edges

        edge_map = F.max_pool2d(
            edge_map.unsqueeze(1).float(),
            kernel_size=2 * self.cfg.feet_edge_radius_px + 1,
            stride=1,
            padding=self.cfg.feet_edge_radius_px,
        ).squeeze(1).bool()

        return edge_map, height_map_resolution, height_map_x_points, height_map_y_points


    def _get_feet_edge_penalty(self, ) -> torch.Tensor:
        """Penalize feet in contact near local height discontinuities measured by the height scanner."""

        if not self._has_edge_map():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        contacts_foot = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

        edge_map, height_map_resolution, height_map_x_points, height_map_y_points = self._compute_edge_map()

        # Get foot positions in world frame, then subtract the scanner origin to express them relative to
        # the scanner position. Shape stays (num_envs, num_feet, 3).
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3]
        feet_pos_scanner_w = feet_pos_w - self._height_scanner3.data.pos_w.unsqueeze(1)

        # The height scanner uses yaw-aligned rays, so rotate the relative foot vectors back by the scanner yaw.
        # This gives foot coordinates in the same local x/y frame as the height_grid.
        scanner_yaw_w = math_utils.yaw_quat(self._height_scanner3.data.quat_w).unsqueeze(1).expand(
            -1, feet_pos_w.shape[1], -1
        )
        feet_pos_scanner = math_utils.quat_apply_inverse(scanner_yaw_w, feet_pos_scanner_w)

        # Split local foot coordinates into x/y components. These are continuous coordinates in meters,
        # not grid indices yet.
        feet_x = feet_pos_scanner[..., 0]
        feet_y = feet_pos_scanner[..., 1]

        # Compute the actual local min/max covered by the scanner rays. This is safer than assuming
        # +/- size/2 because it uses the generated ray pattern directly.
        scanner_ray_starts = self._height_scanner3.ray_starts[0].to(device=self.device, dtype=feet_x.dtype)
        height_grid_x_min = torch.min(scanner_ray_starts[:, 0])
        height_grid_x_max = torch.max(scanner_ray_starts[:, 0])
        height_grid_y_min = torch.min(scanner_ray_starts[:, 1])
        height_grid_y_max = torch.max(scanner_ray_starts[:, 1])

        # Keep track of which feet are inside the scanner footprint. Feet outside the scanned area are
        # ignored for this penalty because we do not have local height data there.
        feet_inside_scan = (
            (feet_x >= height_grid_x_min)
            & (feet_x <= height_grid_x_max)
            & (feet_y >= height_grid_y_min)
            & (feet_y <= height_grid_y_max)
        )

        # Quantize each foot's local x/y position to the nearest cell index in the scanner grid.
        # Clamp afterwards so gather stays valid even for feet just outside the scan boundary.
        feet_ix = torch.round((feet_x - height_grid_x_min) / height_map_resolution).long()
        feet_iy = torch.round((feet_y - height_grid_y_min) / height_map_resolution).long()
        feet_ix = torch.clamp(feet_ix, 0, height_map_x_points - 1)
        feet_iy = torch.clamp(feet_iy, 0, height_map_y_points - 1)

        edge_map_flat = edge_map.reshape(self.num_envs, -1)

        # Convert 2D grid indices (iy, ix) to flat indices and gather whether each foot cell is marked as edge.
        feet_grid_ids = feet_iy * height_map_x_points + feet_ix
        feet_at_edge = torch.gather(edge_map_flat, 1, feet_grid_ids)

        # Penalize only feet that are in contact, inside the scanned area, and located on/near an edge.
        violating_feet = contacts_foot & feet_inside_scan & feet_at_edge

        #return torch.sum(violating_feet.float(), dim=1)

        grid_xy = scanner_ray_starts[:, :2]
        feet_xy = torch.stack((feet_x, feet_y), dim=-1)
        distances_to_grid = torch.linalg.norm(feet_xy.unsqueeze(2) - grid_xy.unsqueeze(0).unsqueeze(0), dim=-1)
        distances_to_grid = distances_to_grid.masked_fill(edge_map_flat.unsqueeze(1), torch.inf)
        nearest_feasible_distance = torch.min(distances_to_grid, dim=-1).values

        scan_diagonal = torch.sqrt(
            torch.square(height_grid_x_max - height_grid_x_min) + torch.square(height_grid_y_max - height_grid_y_min)
        )
        nearest_feasible_distance = torch.where(
            torch.isfinite(nearest_feasible_distance),
            nearest_feasible_distance,
            scan_diagonal,
        )
        return torch.sum(
            torch.where(violating_feet, nearest_feasible_distance, torch.zeros_like(nearest_feasible_distance)),
            dim=1,
        )




    def _get_rewards(self) -> torch.Tensor:

        # track_height ------------------------------------------------------------------------------
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)

        height_error = torch.square(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])
        height_error_mapped = torch.exp(-height_error / 0.01)


        # linear velocity tracking ----------------------------------------------------------------
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        

        # z velocity tracking ---------------------------------------------------------------------
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])


        # terrain orientation ----------------------------------------------------------------------
        height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution))
        distance_between_front_and_back = (height_map_x_points/2)* height_map_resolution

        cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_back = cols_back.flatten().to(height_data_scanner.device)
        selected_height_data_back = height_data_scanner[:, cols_back]

        cols_front = torch.arange(int(height_map_x_points/2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_front = cols_front.flatten().to(height_data_scanner.device)
        selected_height_data_front = height_data_scanner[:, cols_front]

        mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
        mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
        delta_z = mean_height_ray_front - mean_height_ray_back
        delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
        terrain_pitch = -torch.atan2(delta_z, delta_s)
        #terrain_pitch = torch.atan2(torch.sin(terrain_pitch), torch.cos(terrain_pitch))

        """cols_right = torch.arange(0, height_data_scanner.shape[1]//2, 1).unsqueeze(1) 
        cols_right = cols_right.flatten().to(height_data_scanner.device)
        selected_height_data_right = height_data_scanner[:, cols_right]

        cols_left = torch.arange(0, height_data_scanner.shape[1]//2, 1).unsqueeze(1) + height_data_scanner.shape[1]//2
        cols_left = cols_left.flatten().to(height_data_scanner.device)
        selected_height_data_left = height_data_scanner[:, cols_left]

        delta_z_roll = torch.mean(selected_height_data_left, dim=1) - torch.mean(selected_height_data_right, dim=1)
        delta_s_roll = torch.tensor((height_map_y_points-1)* height_map_resolution).to(self.device)
        terrain_roll = torch.atan2(delta_z_roll, delta_s_roll)
        # TODO check if we need roll in base frame
        """
        terrain_roll = torch.zeros_like(terrain_pitch)

        root_roll_w, root_pitch_w, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        root_roll_w = torch.atan2(torch.sin(root_roll_w), torch.cos(root_roll_w))
        root_pitch_w = torch.atan2(torch.sin(root_pitch_w), torch.cos(root_pitch_w))
        
        base_orientation =  torch.square(terrain_pitch - root_pitch_w) + torch.square(terrain_roll - root_roll_w)


        # angular velocity x/y tracking ---------------------------------------------------------------
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)


        # yaw rate tracking ---------------------------------------------------------------------------
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        
        # action rate ---------------------------------------------------------------------------------
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smoothness = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_previous_actions), dim=1)
        
        
        # undersired contacts -------------------------------------------------------------------------
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        

        # joint acceleration ---------------------------------------------------------------------------
        joints_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)


        # joint torques --------------------------------------------------------------------------------
        joints_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)


        # energy = torque * velocity -------------------------------------------------------------------
        joints_energy = torch.sum(torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1)

        
        joint_pos = self._robot.data.joint_pos[:, self._ids_joints_order]
        default_joint_pos = self._robot.data.default_joint_pos[:, self._ids_joints_order]


        # hip position --------------------------------------------------------------------------------
        hip_joints_position = joint_pos[:,0:4]
        hip_joints_position_error = torch.square(hip_joints_position - default_joint_pos[:,0:4])
        hip_joints_position_reward = torch.sum(hip_joints_position_error,dim=1)


        # thigh position -------------------------------------------------------------------------------
        thigh_joints_position = joint_pos[:,4:8]
        thigh_joints_position_error = torch.square(thigh_joints_position - default_joint_pos[:,4:8])
        thigh_joints_position_reward = torch.sum(thigh_joints_position_error,dim=1)


        # calf position --------------------------------------------------------------------------------
        calf_joints_position = joint_pos[:,8:12]
        calf_joints_position_error = torch.square(calf_joints_position - default_joint_pos[:,8:12])
        calf_joints_position_reward = torch.sum(calf_joints_position_error,dim=1)


        # feet airtime ---------------------------------------------------------------------------------
        mode_time = 0.5
        #first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_sensor_ids]
        #last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_sensor_ids]
        #feet_air_time = torch.sum((last_air_time - mode_time) * first_contact, dim=1) * (
        #    torch.norm(self._commands[:, :2], dim=1) > 0.1
        #)
        
        current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_contact_sensor_ids]
        current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_contact_sensor_ids]
        t_max = torch.max(current_air_time, current_contact_time)
        t_min = torch.clip(t_max, max=mode_time)
        feet_air_time_per_leg = torch.where(t_max < mode_time, t_min, torch.zeros_like(t_min))
        feet_air_time_per_leg -= torch.where(current_air_time > mode_time, (current_air_time - mode_time), torch.zeros_like(current_air_time))
        feet_air_time = torch.sum(feet_air_time_per_leg, dim=1) * (
                torch.norm(self._commands[:, :3], dim=1) > 0.1
        )
        

        # feet slide ---------------------------------------------------------------------------------
        contacts_foot = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        body_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2]
        feet_slide = torch.sum(body_vel.norm(dim=-1) * contacts_foot, dim=1)


        # feet edge -----------------------------------------------------------------------------------
        feet_edge = self._get_feet_edge_penalty()


        # periodical contacts suggestion --------------------------------------------------------------
        should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
        self._phase_signal += self.step_dt * self._step_freq
        self._phase_signal = self._phase_signal % 1.0
        contact_periodic_on = self._phase_signal < self._duty_factor
        periodic_contact_suggestion = (torch.sum(contact_periodic_on*contacts_foot, dim=1) + \
                                   torch.sum(~contact_periodic_on*~contacts_foot, dim=1))*should_move/4.0


        # stance contact suggestion -------------------------------------------------------------------
        stance_contact_suggestion = (torch.sum(contacts_foot, dim=1)*~should_move/4.0)
        

        # feet height clearance mujoco -----------------------------------------------------------------
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_sensor_ids]
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (torch.max(torch.norm(net_contact_forces[:, :, self._feet_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0)

        self._swing_peak *= ~is_contact # reset if the foot is in contact
        self._swing_peak = torch.max(self._swing_peak, self._robot.data.body_pos_w[:, self._feet_ids_robot, 2].clone()) 
        feet_z_target_error_mujoco = self.cfg.desired_feet_height + torch.cat((mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1) - self._swing_peak
        # If the raw error is negative, halve it to not discourage too much
        feet_z_target_error_mujoco = torch.where(feet_z_target_error_mujoco < 0.0, feet_z_target_error_mujoco * 0.2, feet_z_target_error_mujoco)
        feet_z_target_error_mujoco = torch.abs(feet_z_target_error_mujoco)
        feet_z_target_error_mujoco = torch.clamp(feet_z_target_error_mujoco, min=.0, max=self.cfg.desired_feet_height)

        feet_height_clearance_mujoco_FL = torch.exp(-feet_z_target_error_mujoco[:,0]/ 0.01) * should_move
        feet_height_clearance_mujoco_FR = torch.exp(-feet_z_target_error_mujoco[:,1]/ 0.01) * should_move
        feet_height_clearance_mujoco_RL = torch.exp(-feet_z_target_error_mujoco[:,2]/ 0.01) * should_move
        feet_height_clearance_mujoco_RR = torch.exp(-feet_z_target_error_mujoco[:,3]/ 0.01) * should_move
        #feet_height_clearance_mujoco = torch.sum(torch.square(self._swing_peak / target_height - 1.0) *  first_contact, dim=-1)
        feet_height_clearance_mujoco = feet_height_clearance_mujoco_FL + feet_height_clearance_mujoco_FR
        feet_height_clearance_mujoco += feet_height_clearance_mujoco_RL + feet_height_clearance_mujoco_RR


        # feet height clearance mujoco periodic ------------------------------------------------------------
        self._swing_peak_periodic *= ~contact_periodic_on # reset if the foot is in contact periodic phase
        self._swing_peak_periodic = torch.max(self._swing_peak_periodic, self._robot.data.body_pos_w[:, self._feet_ids_robot, 2].clone())
        feet_z_target_error_mujoco_periodic = self.cfg.desired_feet_height + torch.cat((mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1) - self._swing_peak_periodic
        # If the raw error is negative, halve it to not discourage too much
        feet_z_target_error_mujoco_periodic = torch.where(feet_z_target_error_mujoco_periodic < 0.0, feet_z_target_error_mujoco_periodic * 0.2, feet_z_target_error_mujoco_periodic)
        feet_z_target_error_mujoco_periodic = torch.abs(feet_z_target_error_mujoco_periodic)
        feet_z_target_error_mujoco_periodic = torch.clamp(feet_z_target_error_mujoco_periodic, min=.0, max=self.cfg.desired_feet_height)

        feet_height_clearance_mujoco_periodic_FL = torch.exp(-feet_z_target_error_mujoco_periodic[:,0]/ 0.01) * should_move * ~contact_periodic_on[:,0]
        feet_height_clearance_mujoco_periodic_FR = torch.exp(-feet_z_target_error_mujoco_periodic[:,1]/ 0.01) * should_move * ~contact_periodic_on[:,1]
        feet_height_clearance_mujoco_periodic_RL = torch.exp(-feet_z_target_error_mujoco_periodic[:,2]/ 0.01) * should_move * ~contact_periodic_on[:,2]
        feet_height_clearance_mujoco_periodic_RR = torch.exp(-feet_z_target_error_mujoco_periodic[:,3]/ 0.01) * should_move * ~contact_periodic_on[:,3]
        #feet_height_clearance_mujoco_periodic = torch.sum(torch.square(self._swing_peak_periodic / target_height - 1.0) *  first_contact, dim=-1) 
        feet_height_clearance_mujoco_periodic = feet_height_clearance_mujoco_periodic_FL + feet_height_clearance_mujoco_periodic_FR
        feet_height_clearance_mujoco_periodic += feet_height_clearance_mujoco_periodic_RL + feet_height_clearance_mujoco_periodic_RR


        # feet height clearance periodic --------------------------------------------------------------------
        feet_z_target_error = self.cfg.desired_feet_height + torch.cat((mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1) - self._robot.data.body_pos_w[:, self._feet_ids_robot, 2]
        # If the raw error is negative, halve it to not discourage too much
        feet_z_target_error = torch.where(feet_z_target_error < 0.0, feet_z_target_error * 0.2, feet_z_target_error)
        feet_z_target_error = torch.abs(feet_z_target_error)
        feet_z_target_error = torch.clamp(feet_z_target_error, min=.0, max=self.cfg.desired_feet_height)
 
        feet_height_clearance_periodic_FL = torch.exp(-feet_z_target_error[:,0]/ 0.01) * should_move * ~contact_periodic_on[:,0]
        feet_height_clearance_periodic_FR = torch.exp(-feet_z_target_error[:,1]/ 0.01) * should_move * ~contact_periodic_on[:,1]
        feet_height_clearance_periodic_RL = torch.exp(-feet_z_target_error[:,2]/ 0.01) * should_move * ~contact_periodic_on[:,2]
        feet_height_clearance_periodic_RR = torch.exp(-feet_z_target_error[:,3]/ 0.01) * should_move * ~contact_periodic_on[:,3]
        feet_height_clearance_periodic = feet_height_clearance_periodic_FL + feet_height_clearance_periodic_FR
        feet_height_clearance_periodic += feet_height_clearance_periodic_RL + feet_height_clearance_periodic_RR


        # feet height clearance standard ---------------------------------------------------------------------------------
        foot_velocity_tanh = torch.tanh(2.0 * torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2], dim=2))
        feet_height_clearance = torch.exp(-torch.sum(feet_z_target_error * foot_velocity_tanh, dim=1)/ 0.01) * should_move


        # feet to com distance --------------------------------------------------------------------------------
        feet_to_base_distance_x = torch.square(torch.mean(self._robot.data.body_pos_w[:, self._feet_ids_robot, 0], dim=1) - self._robot.data.root_state_w[:, 0])
        feet_to_base_distance_y = torch.square(torch.mean(self._robot.data.body_pos_w[:, self._feet_ids_robot, 1], dim=1) - self._robot.data.root_state_w[:, 1])
        feet_to_base_distance = -torch.sqrt(feet_to_base_distance_x + feet_to_base_distance_y)


        # feet to hip distance --------------------------------------------------------------------------------
        ROT_W2H = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w))
        feet_to_base_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        feet_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), feet_to_base_w.transpose(1, 2))
        
        hip_to_base_w = self._robot.data.body_pos_w[:, self._hip_ids_robot, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        hip_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), hip_to_base_w.transpose(1, 2))
        
        desired_hip_offset = self._desired_hip_offset
        feet_to_hip_distance_x = torch.square(feet_to_base_h[:, 0] - hip_to_base_h[:, 0])
        feet_to_hip_distance_y = torch.square(feet_to_base_h[:, 1] + desired_hip_offset.unsqueeze(0) - hip_to_base_h[:, 1])
        feet_to_hip_distance = -torch.mean(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)
        # If should_move is False, multiply the distance by 3 (GPU-friendly, vectorized)
        # `should_move` is a boolean tensor defined earlier (shape: [num_envs])
        feet_to_hip_distance = feet_to_hip_distance * torch.where(
            should_move, torch.ones_like(feet_to_hip_distance), torch.full_like(feet_to_hip_distance, 3.0)
        )


        # Penalize feet hitting vertical surfaces ----------------------------------------------------------------
        forces_z = torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, 2])
        forces_xy = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, :2], dim=2)
        feet_vertical_surface_contacts = torch.any(forces_xy > 4 * forces_z, dim=1).float()
        feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


        rewards = {
            "track_height_exp": height_error_mapped * self.cfg.height_reward_scale * self.step_dt,
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "track_orientation_l2": base_orientation * self.cfg.orientation_reward_scale * self.step_dt,
            "track_ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,

            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,

            "joints_hip_pos_l2": hip_joints_position_reward * self.cfg.joints_hip_position_reward_scale * self.step_dt,
            "joints_thigh_pos_l2": thigh_joints_position_reward * self.cfg.joints_thigh_position_reward_scale * self.step_dt,
            "joints_calf_pos_l2": calf_joints_position_reward * self.cfg.joints_calf_position_reward_scale * self.step_dt,
            "joints_acc_l2": joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt,

            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            
            "feet_height_clearance": feet_height_clearance * self.cfg.feet_height_clearance_reward_scale * self.step_dt,
            "feet_height_clearance_periodic": feet_height_clearance_periodic * self.cfg.feet_height_clearance_periodic_reward_scale * self.step_dt,
            "feet_height_clearance_mujoco": feet_height_clearance_mujoco * self.cfg.feet_height_clearance_mujoco_reward_scale * self.step_dt,
            "feet_height_clearance_mujoco_periodic": feet_height_clearance_mujoco_periodic * self.cfg.feet_height_clearance_mujoco_periodic_reward_scale * self.step_dt,
            
            "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale * self.step_dt,
            "feet_to_base_distance_l2": feet_to_base_distance * self.cfg.feet_to_base_distance_reward_scale * self.step_dt,
            "feet_to_hip_distance_l2": feet_to_hip_distance * self.cfg.feet_to_hip_distance_reward_scale * self.step_dt,
            "feet_edge": feet_edge * self.cfg.feet_edge_reward_scale * self.step_dt,
            "feet_vertical_surface_contacts": feet_vertical_surface_contacts * self.cfg.feet_vertical_surface_contacts_reward_scale * self.step_dt,

            "periodic_contact_suggestion": periodic_contact_suggestion * self.cfg.periodic_contact_suggestion_reward_scale * self.step_dt,
            "stance_contact_suggestion": stance_contact_suggestion * self.cfg.stance_contact_suggestion_reward_scale * self.step_dt,
            
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Check for NaNs and Infs
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("NaN or Inf detected in reward computation. Setting reward to zero for affected environments.")
            breakpoint()  # For debugging purposes
            reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_contact_sensor_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died = torch.logical_or(died_check_base, died_check_hips)
        # Check if the robot is out of bounds of the terrain
        """if(self._terrain.cfg.terrain_generator is not None):
            # obtain the size of the sub-terrains
            terrain_gen_cfg = self._terrain.cfg.terrain_generator
            grid_width, grid_length = terrain_gen_cfg.size
            n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
            border_width = terrain_gen_cfg.border_width
            # compute the size of the map
            map_width = n_rows * grid_width + 2 * border_width
            map_height = n_cols * grid_length + 2 * border_width

            # check if the agent is out of bounds
            distance_buffer = 3.
            x_out_of_bounds = torch.abs(self._robot.data.root_state_w[:, 0]) > 0.5 * map_width - distance_buffer
            y_out_of_bounds = torch.abs(self._robot.data.root_state_w[:, 1]) > 0.5 * map_height - distance_buffer
            out_of_bounds = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
            time_out = torch.logical_or(time_out, out_of_bounds) #HACK"""
        
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            # Curriculum based on the distance the robot walked
            distance = torch.norm(self._robot.data.root_state_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1)
            # robots that walked far enough progress to harder terrains
            move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = distance < torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
            move_down *= ~move_up
            # update terrain levels
            self._terrain.update_env_origins(env_ids, move_up, move_down)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0
        
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[env_ids, 0] *= 0.5
        self._commands[env_ids, 1] *= 0.25 
        self._commands[env_ids, 2] *= 0.5 

        # Reset swing peak
        self._swing_peak[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self._swing_peak_periodic[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0

        # Reset observation history
        self._observation_history[env_ids] *= 0.0

        # Reset obs and noise concurrent
        if(self.cfg.use_concurrent_state_est):
            self._observation_history_concurrent_state_est[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est.reset(env_ids)
        
        # Reset obs and noise rma
        if(self.cfg.use_rma):
            self._observation_history_rma[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.2, 0.2)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
        
        self.extras["log"].update(extras)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if not getattr(self.cfg, "visualize_edge_map", False) or not self._has_edge_map():
            if self._edge_map_visualizer is not None:
                self._edge_map_visualizer.set_visibility(False)
            return

        if debug_vis:
            if self._edge_map_visualizer is None:
                marker_radius = getattr(self.cfg, "edge_map_visualization_dot_radius", 0.015)
                edge_map_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/EdgeMap",
                    markers={
                        "feasible": sim_utils.SphereCfg(
                            radius=marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                        ),
                        "not_feasible": sim_utils.SphereCfg(
                            radius=marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                        ),
                    },
                )
                self._edge_map_visualizer = VisualizationMarkers(edge_map_marker_cfg)
            self._edge_map_visualizer.set_visibility(True)
        elif self._edge_map_visualizer is not None:
            self._edge_map_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        if self._edge_map_visualizer is None or not self._edge_map_visualizer.is_visible() or not self._has_edge_map():
            return

        env_ids_cfg = getattr(self.cfg, "edge_map_visualization_env_ids", [0])
        if isinstance(env_ids_cfg, int):
            env_ids_cfg = [env_ids_cfg]
        env_ids = torch.tensor(env_ids_cfg, dtype=torch.long, device=self.device)
        env_ids = env_ids[(env_ids >= 0) & (env_ids < self.num_envs)]
        if env_ids.numel() == 0:
            return

        edge_map, _, _, _ = self._compute_edge_map()
        translations = self._height_scanner3.data.ray_hits_w[env_ids].reshape(-1, 3).clone()
        marker_indices = edge_map[env_ids].reshape(-1).long()

        valid_hits = torch.isfinite(translations).all(dim=1)
        if not torch.any(valid_hits):
            return

        translations = translations[valid_hits]
        translations[:, 2] += getattr(self.cfg, "edge_map_visualization_height_offset", 0.02)
        marker_indices = marker_indices[valid_hits]

        self._edge_map_visualizer.visualize(translations=translations, marker_indices=marker_indices)



    def _get_new_random_commands(self):
        
        # Change direction while moving
        resample_time = self.episode_length_buf == self.max_episode_length - 400
        commands_resample = torch.zeros_like(self._commands).uniform_(-1.0, 1.0)
        commands_resample[:, 0] *= 0.5
        commands_resample[:, 1] *= 0.25 
        commands_resample[:, 2] *= 0.5 
        self._commands[:, :3] = self._commands[:, :3] * ~resample_time.unsqueeze(1).expand(-1, 3) + commands_resample * resample_time.unsqueeze(1).expand(-1, 3)

        # Stop
        rest_time = torch.logical_and(
            self.episode_length_buf >= self.max_episode_length - 250,
            self.episode_length_buf < self.max_episode_length - 150
        )
        self._commands[:, :3] *= ~rest_time.unsqueeze(1).expand(-1, 3)

        # Move again
        resample_time_2 = self.episode_length_buf == self.max_episode_length - 150
        commands_resample_2 = torch.zeros_like(self._commands).uniform_(-1.0, 1.0)
        commands_resample_2[:, 0] *= 0.5
        commands_resample_2[:, 1] *= 0.25 
        commands_resample_2[:, 2] *= 0.5 
        self._commands[:, :3] = self._commands[:, :3] * ~resample_time_2.unsqueeze(1).expand(-1, 3) + commands_resample_2 * resample_time_2.unsqueeze(1).expand(-1, 3)        

        # Took some envs, and put to zero the vel
        num_fixed_envs = 500
        if self.num_envs > num_fixed_envs:
            fixed_env_ids = torch.arange(num_fixed_envs, device=self.device)
            self._commands[fixed_env_ids, :3] *= 0.0


    def _get_concurrent_state_estimation(self):
        # Using a supervised learning state estimation
        obs_concurrent_state_est = torch.cat(
            [
                tensor
                for tensor in (
                    self._imu.data.lin_acc_b,
                    self._imu.data.ang_vel_b,
                    self._imu.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order] * self.cfg.observation_joint_vel_scale,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        #the bottom element is the newest observation!!
        self._observation_history_concurrent_state_est = torch.cat((self._observation_history_concurrent_state_est[:,1:,:], obs_concurrent_state_est.unsqueeze(1)), dim=1)
        obs_concurrent_state_est = torch.flatten(self._observation_history_concurrent_state_est, start_dim=1)     

        # Add noise to the observation - this is usually done in direct_rl.py in IsaacLab, but 
        # the obs of concurrent SE does not pass from there - its prediciton yes instead!
        if self.cfg.observation_noise_model:          
            obs_concurrent_state_est = self._observation_noise_model_concurrent_state_est(obs_concurrent_state_est)   

        # Saving data
        output_concurrent_state_est = self._robot.data.root_lin_vel_b
        self._concurrent_state_est_network.dataset.add_sample(obs_concurrent_state_est, output_concurrent_state_est)

        # Prediction
        num_episode_from_start = self.common_step_counter / 24. #self.max_episode_length #HACK this should be taken from rsl rl
        num_final_episode_from_start = 8000.
        if num_episode_from_start > self.cfg.concurrent_state_est_ep_saving_start:
            with torch.no_grad(): 
                prediction_concurrent_state_est = self._concurrent_state_est_network(obs_concurrent_state_est)
            linear_velocity_b = prediction_concurrent_state_est[:, :3]
        else:
            linear_velocity_b = self._robot.data.root_lin_vel_b

        # Train at some interval
        if (num_episode_from_start % self.cfg.concurrent_state_est_ep_saving_interval == 0 and 
            num_episode_from_start > self.cfg.concurrent_state_est_ep_saving_start - 1 and 
                num_episode_from_start < num_final_episode_from_start - 500):  # Adjust the interval as needed
            self._concurrent_state_est_network.train_network(batch_size=self.cfg.concurrent_state_est_batch_size, 
                                                            epochs=self.cfg.concurrent_state_est_train_epochs, 
                                                            learning_rate=self.cfg.concurrent_state_est_lr, device=self.device)
            # Save the network
            self._concurrent_state_est_network.save_network("concurrent_state_estimator.pth", self.device)    

        return linear_velocity_b  


    def _get_rma(self):
        # Learning privileged information via supervised learning
        obs_rma = torch.cat(
            [
                tensor
                for tensor in (
                    self._imu.data.lin_acc_b,
                    self._imu.data.ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order] * self.cfg.observation_joint_vel_scale,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        #the bottom element is the newest observation!!
        self._observation_history_rma = torch.cat((self._observation_history_rma[:,1:,:], obs_rma.unsqueeze(1)), dim=1)
        obs = torch.flatten(self._observation_history_rma, start_dim=1)

        # Add noise to the observation - this is usually done in direct_rl.py in IsaacLab, but 
        # the obs of concurrent SE does not pass from there - its prediciton yes instead!
        if self.cfg.observation_noise_model:          
            obs = self._observation_noise_model_rma(obs.clone())  
        
        outputs_rma = self._get_privileged_observation()
        
        if self.cfg.rma_use_latent_space:
            with torch.no_grad():
                target_rma = self._rma_latent_encoder.encode(outputs_rma)
        else:
            target_rma = outputs_rma

        self._rma_network.dataset.add_sample(obs, target_rma)

        # Prediction
        num_episode_from_start = self.common_step_counter / 24. #self.max_episode_length #HACK this should be taken from rsl rl
        num_final_episode_from_start = 8000.
        if num_episode_from_start > self.cfg.rma_ep_saving_start:
            with torch.no_grad(): 
                prediction_rma = self._rma_network(obs)
            obs_rma = prediction_rma
        else:
            obs_rma = target_rma

        # Train at some interval
        if (num_episode_from_start % self.cfg.rma_ep_saving_interval == 0 and 
            num_episode_from_start > self.cfg.rma_ep_saving_start - 1 and 
                num_episode_from_start < num_final_episode_from_start - 500):  # Adjust the interval as needed
            self._rma_network.train_network(batch_size=self.cfg.rma_batch_size, 
                                            epochs=self.cfg.rma_train_epochs, 
                                            learning_rate=self.cfg.rma_lr, 
                                            device=self.device)
            # Save the network
            self._rma_network.save_network("rma.pth", self.device)
        
        return obs_rma


    def _get_privileged_observation(self):
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        asset: Articulation = self.scene[asset_cfg.name]

        # PD of the joints
        hip_stiffness = asset.actuators["hip"].stiffness
        thigh_stiffness = asset.actuators["thigh"].stiffness
        calf_stiffness = asset.actuators["calf"].stiffness

        hip_damping = asset.actuators["hip"].damping
        thigh_damping = asset.actuators["thigh"].damping
        calf_damping = asset.actuators["calf"].damping
        
        default_stiffness = asset.data.default_joint_stiffness[0][0]
        default_damping = asset.data.default_joint_damping[0][0]


        # height error
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)
        height_error = torch.abs(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])


        # terrain orientation
        height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution))
        distance_between_front_and_back = (height_map_x_points/2)* height_map_resolution

        cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_back = cols_back.flatten().to(height_data_scanner.device)
        selected_height_data_back = height_data_scanner[:, cols_back]

        cols_front = torch.arange(int(height_map_x_points/2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_front = cols_front.flatten().to(height_data_scanner.device)
        selected_height_data_front = height_data_scanner[:, cols_front]

        mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
        mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
        delta_z = mean_height_ray_front - mean_height_ray_back
        delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
        terrain_pitch = -torch.atan2(delta_z, delta_s)

        contacts_foot = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

        obs_privileged = torch.cat(( 
                            hip_stiffness/default_stiffness, thigh_stiffness/default_stiffness, calf_stiffness/default_stiffness, #P gain
                            hip_damping/default_damping, thigh_damping/default_damping, calf_damping/default_damping, #D gain
                            self._robot.data.root_lin_vel_b,
                            height_error.unsqueeze(1),
                            terrain_pitch.unsqueeze(1),
                            contacts_foot,
                            ) 
                        , dim=-1)
        return obs_privileged
