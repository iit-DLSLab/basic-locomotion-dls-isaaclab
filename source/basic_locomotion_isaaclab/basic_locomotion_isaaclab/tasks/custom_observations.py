from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


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
                self._robot.data.joint_pos[:, self._ids_joints_order]
                - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                self._robot.data.joint_vel[:, self._ids_joints_order] * self.cfg.observation_joint_vel_scale,
                self._actions,
            )
            if tensor is not None
        ],
        dim=-1,
    )
    #the bottom element is the newest observation!!
    self._observation_history_concurrent_state_est = torch.cat(
        (self._observation_history_concurrent_state_est[:, 1:, :], obs_concurrent_state_est.unsqueeze(1)), dim=1
    )
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
                self._robot.data.joint_pos[:, self._ids_joints_order]
                - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                self._robot.data.joint_vel[:, self._ids_joints_order] * self.cfg.observation_joint_vel_scale,
                self._actions,
            )
            if tensor is not None
        ],
        dim=-1,
    )
    #the bottom element is the newest observation!!
    self._observation_history_rma = torch.cat((self._observation_history_rma[:, 1:, :], obs_rma.unsqueeze(1)), dim=1)
    obs = torch.flatten(self._observation_history_rma, start_dim=1)

    # Add noise to the observation - this is usually done in direct_rl.py in IsaacLab, but
    # the obs of concurrent SE does not pass from there - its prediciton yes instead!
    if self.cfg.observation_noise_model:
        obs = self._observation_noise_model_rma(obs.clone())

    outputs_rma = _get_privileged_observation(self)

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
