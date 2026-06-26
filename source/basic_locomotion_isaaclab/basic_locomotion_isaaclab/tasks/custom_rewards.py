from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def _has_edge_map(self) -> bool:
    return hasattr(self, "_height_scanner3")


def _compute_edge_map(self) -> tuple[torch.Tensor, float, int, int]:
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


def track_height_exp(self) -> torch.Tensor:
    height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)
    mean_height_ray = torch.mean(height_data_scanner, dim=1)

    height_error = torch.square(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])
    height_error_mapped = torch.exp(-height_error / 0.01)
    return height_error_mapped


def track_lin_vel_xy_exp(self) -> torch.Tensor:
    lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
    lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
    return lin_vel_error_mapped


def track_lin_vel_z_l2(self) -> torch.Tensor:
    z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
    return z_vel_error


def track_orientation_l2(self) -> torch.Tensor:
    height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
    distance_between_front_and_back = (height_map_x_points / 2) * height_map_resolution

    cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(
        int(height_map_x_points / 2)
    )
    cols_back = cols_back.flatten().to(height_data_scanner.device)
    selected_height_data_back = height_data_scanner[:, cols_back]

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
    delta_z = mean_height_ray_front - mean_height_ray_back
    delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
    terrain_pitch = -torch.atan2(delta_z, delta_s)
    terrain_roll = torch.zeros_like(terrain_pitch)

    root_roll_w, root_pitch_w, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
    root_roll_w = torch.atan2(torch.sin(root_roll_w), torch.cos(root_roll_w))
    root_pitch_w = torch.atan2(torch.sin(root_pitch_w), torch.cos(root_pitch_w))

    base_orientation = torch.square(terrain_pitch - root_pitch_w) + torch.square(terrain_roll - root_roll_w)
    return base_orientation


def track_ang_vel_xy_l2(self) -> torch.Tensor:
    ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
    return ang_vel_error


def track_ang_vel_z_exp(self) -> torch.Tensor:
    yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
    yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
    return yaw_rate_error_mapped


def undesired_contacts(self) -> torch.Tensor:
    net_contact_forces = self._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[
        0
    ] > 1.0
    contacts = torch.sum(is_contact, dim=1)
    return contacts


def action_rate_l2(self) -> torch.Tensor:
    action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
    return action_rate


def action_smoothness_l2(self) -> torch.Tensor:
    action_smoothness = torch.sum(
        torch.square(self._actions - 2 * self._previous_actions + self._previous_previous_actions), dim=1
    )
    return action_smoothness


def joints_hip_pos_l2(self) -> torch.Tensor:
    joint_pos = self._robot.data.joint_pos[:, self._ids_joints_order]
    default_joint_pos = self._robot.data.default_joint_pos[:, self._ids_joints_order]

    hip_joints_position = joint_pos[:, 0:4]
    hip_joints_position_error = torch.square(hip_joints_position - default_joint_pos[:, 0:4])
    hip_joints_position_reward = torch.sum(hip_joints_position_error, dim=1)
    return hip_joints_position_reward


def joints_thigh_pos_l2(self) -> torch.Tensor:
    joint_pos = self._robot.data.joint_pos[:, self._ids_joints_order]
    default_joint_pos = self._robot.data.default_joint_pos[:, self._ids_joints_order]

    thigh_joints_position = joint_pos[:, 4:8]
    thigh_joints_position_error = torch.square(thigh_joints_position - default_joint_pos[:, 4:8])
    thigh_joints_position_reward = torch.sum(thigh_joints_position_error, dim=1)
    return thigh_joints_position_reward


def joints_calf_pos_l2(self) -> torch.Tensor:
    joint_pos = self._robot.data.joint_pos[:, self._ids_joints_order]
    default_joint_pos = self._robot.data.default_joint_pos[:, self._ids_joints_order]

    calf_joints_position = joint_pos[:, 8:12]
    calf_joints_position_error = torch.square(calf_joints_position - default_joint_pos[:, 8:12])
    calf_joints_position_reward = torch.sum(calf_joints_position_error, dim=1)
    return calf_joints_position_reward


def joints_acc_l2(self) -> torch.Tensor:
    joints_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
    return joints_accel


def joints_torques_l2(self) -> torch.Tensor:
    joints_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
    return joints_torques


def joints_energy_l1(self) -> torch.Tensor:
    joints_energy = torch.sum(torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1)
    return joints_energy


def feet_air_time(self) -> torch.Tensor:
    mode_time = 0.5
    current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_contact_sensor_ids]
    current_contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_contact_sensor_ids]
    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    feet_air_time_per_leg = torch.where(t_max < mode_time, t_min, torch.zeros_like(t_min))
    feet_air_time_per_leg -= torch.where(
        current_air_time > mode_time, current_air_time - mode_time, torch.zeros_like(current_air_time)
    )
    feet_air_time = torch.sum(feet_air_time_per_leg, dim=1) * (torch.norm(self._commands[:, :3], dim=1) > 0.1)
    return feet_air_time


def feet_air_time_variance(self):

    last_air_time = torch.clip(self._contact_sensor.data.last_air_time[:, self._feet_contact_sensor_ids], max=0.5)
    last_contact_time = torch.clip(self._contact_sensor.data.last_contact_time[:, self._feet_contact_sensor_ids], max=0.5)
    variance_denominator = (4.0 - 1.0)#.clamp(min=1.0)

    mean_air_time = torch.sum(last_air_time, dim=1) / 4.
    mean_contact_time = torch.sum(last_contact_time, dim=1) / 4.
    air_time_variance = torch.sum(torch.square(last_air_time - mean_air_time.unsqueeze(1)), dim=1)
    contact_time_variance = torch.sum(torch.square(last_contact_time - mean_contact_time.unsqueeze(1)), dim=1)
    feet_air_time_variance = (air_time_variance + contact_time_variance) / variance_denominator

    return feet_air_time_variance


def feet_height_clearance_aperiodic(self) -> torch.Tensor:
    height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1

    cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(
        int(height_map_x_points / 2)
    )
    cols_back = cols_back.flatten().to(height_data_scanner.device)
    selected_height_data_back = height_data_scanner[:, cols_back]

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01

    feet_z_target_error = self.cfg.desired_feet_height + torch.cat(
        (mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1
    ) - self._robot.data.body_pos_w[:, self._feet_ids_robot, 2]
    feet_z_target_error = torch.where(feet_z_target_error < 0.0, feet_z_target_error * 0.2, feet_z_target_error)
    feet_z_target_error = torch.abs(feet_z_target_error)
    feet_z_target_error = torch.clamp(feet_z_target_error, min=0.0, max=self.cfg.desired_feet_height)

    foot_velocity_tanh = torch.tanh(2.0 * torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2], dim=2))
    feet_height_clearance = torch.exp(-torch.sum(feet_z_target_error * foot_velocity_tanh, dim=1) / 0.01) * should_move
    return feet_height_clearance


def feet_height_clearance_periodic(self) -> torch.Tensor:
    height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1

    cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(
        int(height_map_x_points / 2)
    )
    cols_back = cols_back.flatten().to(height_data_scanner.device)
    selected_height_data_back = height_data_scanner[:, cols_back]

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    contact_periodic_on = self._phase_signal < self._duty_factor

    feet_z_target_error = self.cfg.desired_feet_height + torch.cat(
        (mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1
    ) - self._robot.data.body_pos_w[:, self._feet_ids_robot, 2]
    feet_z_target_error = torch.where(feet_z_target_error < 0.0, feet_z_target_error * 0.2, feet_z_target_error)
    feet_z_target_error = torch.abs(feet_z_target_error)
    feet_z_target_error = torch.clamp(feet_z_target_error, min=0.0, max=self.cfg.desired_feet_height)

    feet_height_clearance_periodic_fl = torch.exp(-feet_z_target_error[:, 0] / 0.01) * should_move * ~contact_periodic_on[
        :, 0
    ]
    feet_height_clearance_periodic_fr = torch.exp(-feet_z_target_error[:, 1] / 0.01) * should_move * ~contact_periodic_on[
        :, 1
    ]
    feet_height_clearance_periodic_rl = torch.exp(-feet_z_target_error[:, 2] / 0.01) * should_move * ~contact_periodic_on[
        :, 2
    ]
    feet_height_clearance_periodic_rr = torch.exp(-feet_z_target_error[:, 3] / 0.01) * should_move * ~contact_periodic_on[
        :, 3
    ]
    feet_height_clearance_periodic = feet_height_clearance_periodic_fl + feet_height_clearance_periodic_fr
    feet_height_clearance_periodic += feet_height_clearance_periodic_rl + feet_height_clearance_periodic_rr
    return feet_height_clearance_periodic


def feet_height_clearance_mujoco_aperiodic(self) -> torch.Tensor:
    height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1

    cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(
        int(height_map_x_points / 2)
    )
    cols_back = cols_back.flatten().to(height_data_scanner.device)
    selected_height_data_back = height_data_scanner[:, cols_back]

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01

    net_contact_forces = self._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0

    self._swing_peak *= ~is_contact
    self._swing_peak = torch.max(self._swing_peak, self._robot.data.body_pos_w[:, self._feet_ids_robot, 2].clone())
    feet_z_target_error_mujoco = self.cfg.desired_feet_height + torch.cat(
        (mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1
    ) - self._swing_peak
    feet_z_target_error_mujoco = torch.where(
        feet_z_target_error_mujoco < 0.0, feet_z_target_error_mujoco * 0.2, feet_z_target_error_mujoco
    )
    feet_z_target_error_mujoco = torch.abs(feet_z_target_error_mujoco)
    feet_z_target_error_mujoco = torch.clamp(feet_z_target_error_mujoco, min=0.0, max=self.cfg.desired_feet_height)

    feet_height_clearance_mujoco_fl = torch.exp(-feet_z_target_error_mujoco[:, 0] / 0.01) * should_move
    feet_height_clearance_mujoco_fr = torch.exp(-feet_z_target_error_mujoco[:, 1] / 0.01) * should_move
    feet_height_clearance_mujoco_rl = torch.exp(-feet_z_target_error_mujoco[:, 2] / 0.01) * should_move
    feet_height_clearance_mujoco_rr = torch.exp(-feet_z_target_error_mujoco[:, 3] / 0.01) * should_move
    feet_height_clearance_mujoco = feet_height_clearance_mujoco_fl + feet_height_clearance_mujoco_fr
    feet_height_clearance_mujoco += feet_height_clearance_mujoco_rl + feet_height_clearance_mujoco_rr
    return feet_height_clearance_mujoco


def feet_height_clearance_mujoco_periodic(self) -> torch.Tensor:
    height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1

    cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(
        int(height_map_x_points / 2)
    )
    cols_back = cols_back.flatten().to(height_data_scanner.device)
    selected_height_data_back = height_data_scanner[:, cols_back]

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    contact_periodic_on = self._phase_signal < self._duty_factor

    self._swing_peak_periodic *= ~contact_periodic_on
    self._swing_peak_periodic = torch.max(
        self._swing_peak_periodic, self._robot.data.body_pos_w[:, self._feet_ids_robot, 2].clone()
    )
    feet_z_target_error_mujoco_periodic = self.cfg.desired_feet_height + torch.cat(
        (mean_height_ray_front.unsqueeze(1).expand(-1, 2), mean_height_ray_back.unsqueeze(1).expand(-1, 2)), dim=1
    ) - self._swing_peak_periodic
    feet_z_target_error_mujoco_periodic = torch.where(
        feet_z_target_error_mujoco_periodic < 0.0,
        feet_z_target_error_mujoco_periodic * 0.2,
        feet_z_target_error_mujoco_periodic,
    )
    feet_z_target_error_mujoco_periodic = torch.abs(feet_z_target_error_mujoco_periodic)
    feet_z_target_error_mujoco_periodic = torch.clamp(
        feet_z_target_error_mujoco_periodic, min=0.0, max=self.cfg.desired_feet_height
    )

    feet_height_clearance_mujoco_periodic_fl = (
        torch.exp(-feet_z_target_error_mujoco_periodic[:, 0] / 0.01) * should_move * ~contact_periodic_on[:, 0]
    )
    feet_height_clearance_mujoco_periodic_fr = (
        torch.exp(-feet_z_target_error_mujoco_periodic[:, 1] / 0.01) * should_move * ~contact_periodic_on[:, 1]
    )
    feet_height_clearance_mujoco_periodic_rl = (
        torch.exp(-feet_z_target_error_mujoco_periodic[:, 2] / 0.01) * should_move * ~contact_periodic_on[:, 2]
    )
    feet_height_clearance_mujoco_periodic_rr = (
        torch.exp(-feet_z_target_error_mujoco_periodic[:, 3] / 0.01) * should_move * ~contact_periodic_on[:, 3]
    )
    feet_height_clearance_mujoco_periodic = (
        feet_height_clearance_mujoco_periodic_fl + feet_height_clearance_mujoco_periodic_fr
    )
    feet_height_clearance_mujoco_periodic += (
        feet_height_clearance_mujoco_periodic_rl + feet_height_clearance_mujoco_periodic_rr
    )
    return feet_height_clearance_mujoco_periodic


def feet_slide(self) -> torch.Tensor:
    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    body_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2]
    feet_slide = torch.sum(body_vel.norm(dim=-1) * contacts_foot, dim=1)
    return feet_slide


def feet_to_hip_distance_l2(self) -> torch.Tensor:
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    rot_w2h = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w))
    feet_to_base_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3] - self._robot.data.root_state_w[
        :, :3
    ].unsqueeze(1)
    feet_to_base_h = torch.matmul(rot_w2h.transpose(1, 2), feet_to_base_w.transpose(1, 2))

    hip_to_base_w = self._robot.data.body_pos_w[:, self._hip_ids_robot, :3] - self._robot.data.root_state_w[
        :, :3
    ].unsqueeze(1)
    hip_to_base_h = torch.matmul(rot_w2h.transpose(1, 2), hip_to_base_w.transpose(1, 2))

    desired_hip_offset = self._desired_hip_offset
    feet_to_hip_distance_x = torch.square(feet_to_base_h[:, 0] - hip_to_base_h[:, 0])
    feet_to_hip_distance_y = torch.square(feet_to_base_h[:, 1] + desired_hip_offset.unsqueeze(0) - hip_to_base_h[:, 1])
    feet_to_hip_distance = -torch.mean(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)
    feet_to_hip_distance = feet_to_hip_distance * torch.where(
        should_move, torch.ones_like(feet_to_hip_distance), torch.full_like(feet_to_hip_distance, 3.0)
    )
    return feet_to_hip_distance


def feet_edge(self) -> torch.Tensor:
    if not _has_edge_map(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )

    edge_map, height_map_resolution, height_map_x_points, height_map_y_points = _compute_edge_map(self)

    feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3]
    feet_pos_scanner_w = feet_pos_w - self._height_scanner3.data.pos_w.unsqueeze(1)

    scanner_yaw_w = math_utils.yaw_quat(self._height_scanner3.data.quat_w).unsqueeze(1).expand(
        -1, feet_pos_w.shape[1], -1
    )
    feet_pos_scanner = math_utils.quat_apply_inverse(scanner_yaw_w, feet_pos_scanner_w)

    feet_x = feet_pos_scanner[..., 0]
    feet_y = feet_pos_scanner[..., 1]

    scanner_ray_starts = self._height_scanner3.ray_starts[0].to(device=self.device, dtype=feet_x.dtype)
    height_grid_x_min = torch.min(scanner_ray_starts[:, 0])
    height_grid_x_max = torch.max(scanner_ray_starts[:, 0])
    height_grid_y_min = torch.min(scanner_ray_starts[:, 1])
    height_grid_y_max = torch.max(scanner_ray_starts[:, 1])

    feet_inside_scan = (
        (feet_x >= height_grid_x_min)
        & (feet_x <= height_grid_x_max)
        & (feet_y >= height_grid_y_min)
        & (feet_y <= height_grid_y_max)
    )

    feet_ix = torch.round((feet_x - height_grid_x_min) / height_map_resolution).long()
    feet_iy = torch.round((feet_y - height_grid_y_min) / height_map_resolution).long()
    feet_ix = torch.clamp(feet_ix, 0, height_map_x_points - 1)
    feet_iy = torch.clamp(feet_iy, 0, height_map_y_points - 1)

    edge_map_flat = edge_map.reshape(self.num_envs, -1)

    feet_grid_ids = feet_iy * height_map_x_points + feet_ix
    feet_at_edge = torch.gather(edge_map_flat, 1, feet_grid_ids)

    violating_feet = contacts_foot & feet_inside_scan & feet_at_edge

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
    feet_edge = torch.sum(
        torch.where(violating_feet, nearest_feasible_distance, torch.zeros_like(nearest_feasible_distance)), dim=1
    )
    return feet_edge


def _set_debug_vis_impl(self, debug_vis: bool):
    if not getattr(self.cfg, "visualize_edge_map", False) or not _has_edge_map(self):
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
    if self._edge_map_visualizer is None or not self._edge_map_visualizer.is_visible() or not _has_edge_map(self):
        return

    env_ids_cfg = getattr(self.cfg, "edge_map_visualization_env_ids", [0])
    if isinstance(env_ids_cfg, int):
        env_ids_cfg = [env_ids_cfg]
    env_ids = torch.tensor(env_ids_cfg, dtype=torch.long, device=self.device)
    env_ids = env_ids[(env_ids >= 0) & (env_ids < self.num_envs)]
    if env_ids.numel() == 0:
        return

    edge_map, _, _, _ = _compute_edge_map(self)
    translations = self._height_scanner3.data.ray_hits_w[env_ids].reshape(-1, 3).clone()
    marker_indices = edge_map[env_ids].reshape(-1).long()

    valid_hits = torch.isfinite(translations).all(dim=1)
    if not torch.any(valid_hits):
        return

    translations = translations[valid_hits]
    translations[:, 2] += getattr(self.cfg, "edge_map_visualization_height_offset", 0.02)
    marker_indices = marker_indices[valid_hits]

    self._edge_map_visualizer.visualize(translations=translations, marker_indices=marker_indices)


def feet_vertical_surface_contacts(self) -> torch.Tensor:
    forces_z = torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, 2])
    forces_xy = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, :2], dim=2)
    feet_vertical_surface_contacts = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return feet_vertical_surface_contacts


def periodic_contact_suggestion(self) -> torch.Tensor:
    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    self._phase_signal += self.step_dt * self._step_freq
    self._phase_signal = self._phase_signal % 1.0
    contact_periodic_on = self._phase_signal < self._duty_factor
    periodic_contact_suggestion = (
        torch.sum(contact_periodic_on * contacts_foot, dim=1) + torch.sum(~contact_periodic_on * ~contacts_foot, dim=1)
    ) * should_move / 4.0
    return periodic_contact_suggestion


def stance_contact_suggestion(self) -> torch.Tensor:
    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    stance_contact_suggestion = torch.sum(contacts_foot, dim=1) * ~should_move / 4.0
    return stance_contact_suggestion
