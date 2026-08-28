# Description: ROS-independent controller core for the real robot policy

import os
import sys
import time
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def _prepend_sys_path(path: Path):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_python_paths():
    repo_root = dir_path.parent
    workspace_root = repo_root.parent
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    candidate_paths = [
        repo_root,
        dir_path,
        workspace_root / ".local" / "lib" / py_version / "site-packages",
        Path.home() / ".local" / "lib" / py_version / "site-packages",
        Path("/usr/local/lib") / py_version / "site-packages",
    ]

    for candidate_path in candidate_paths:
        _prepend_sys_path(candidate_path)


_bootstrap_python_paths()

import mujoco
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

from locomotion_policy_wrapper import LocomotionPolicyWrapper

import config


np.set_printoptions(precision=3, suppress=True)

USE_MUJOCO_RENDER = False


class ControllerCore:
    def __init__(self):
        robot_name = config.robot
        scene_name = config.scene
        simulation_dt = 0.002

        self.env = QuadrupedEnv(
            robot=robot_name,
            scene=scene_name,
            sim_dt=simulation_dt,
            base_vel_command_type="human",
        )
        try:
            self.env.reset(random=False)
        except Exception as exc:
            diagnostics = {
                "controller_core": __file__,
                "python_executable": sys.executable,
                "python_version": sys.version,
                "robot": robot_name,
                "scene": scene_name,
            }
            try:
                import gym_quadruped

                diagnostics["gym_quadruped"] = gym_quadruped.__file__
            except Exception as import_exc:
                diagnostics["gym_quadruped"] = f"import failed: {import_exc!r}"

            try:
                diagnostics["mujoco"] = mujoco.__file__
            except Exception as import_exc:
                diagnostics["mujoco"] = f"import failed: {import_exc!r}"

            raise RuntimeError(
                "ControllerCore environment reset failed. "
                f"Diagnostics: {diagnostics}"
            ) from exc

        self.last_render_time = time.time()
        if USE_MUJOCO_RENDER:
            self.env.render()

        self.loop_time = simulation_dt
        self.last_start_time = None
        self.last_processed_joy_update_id = -1
        self.joystick_timeout_active = False

        self.locomotion_policy = LocomotionPolicyWrapper(env=self.env)

        self.stand_up_and_down_actions = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu / 4))) for _ in range(4)])
        keyframe_id = mujoco.mj_name2id(self.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        go_down_qpos = self.env.mjModel.key_qpos[keyframe_id]
        self.stand_up_and_down_actions.FL = go_down_qpos[7:10]
        self.stand_up_and_down_actions.FR = go_down_qpos[10:13]
        self.stand_up_and_down_actions.RL = go_down_qpos[13:16]
        self.stand_up_and_down_actions.RR = go_down_qpos[16:19]
        self.joint_positions = np.array(go_down_qpos[7:19], copy=True)

    def compute_control_step(
        self,
        position,
        orientation,
        linear_velocity,
        angular_velocity,
        joint_positions,
        joint_velocities,
        imu_linear_acceleration,
        imu_angular_velocity,
        imu_orientation,
        base_state_received,
        joints_state_received,
        imu_state_received,
        joy_axes,
        joy_buttons,
        joy_message_time,
        joy_update_id,
        is_rl_activated,
        monotonic_time,
        wall_time,
    ):
        if self.last_start_time is not None:
            self.loop_time = monotonic_time - self.last_start_time
        self.last_start_time = monotonic_time
        simulation_dt = self.loop_time

        if config.training_env["use_imu"] or config.training_env["use_concurrent_state_est"]:
            if (not imu_state_received) or (not joints_state_received):
                return None
        else:
            if (not base_state_received) or (not joints_state_received):
                return None

        # Validate input shapes
        if len(joint_positions) != 12:
            print(
                f"WARNING: Invalid joint_positions shape {len(joint_positions)}, expected 12. "
                f"Using previous values. joint_positions={joint_positions}"
            )
            joint_positions = self.joint_positions
        if len(joint_velocities) != 12:
            print(
                f"WARNING: Invalid joint_velocities shape {len(joint_velocities)}, expected 12. "
                f"Using zeros instead. joint_velocities={joint_velocities}"
            )
            joint_velocities = np.zeros(12)

        if joy_axes is not None and joy_update_id != self.last_processed_joy_update_id:
            filter_joystick = 0.7
            self.env._ref_base_lin_vel_H[0] = (
                self.env._ref_base_lin_vel_H[0] * filter_joystick
                + (joy_axes[1] / 3.5) * (1 - filter_joystick)
            )
            self.env._ref_base_lin_vel_H[1] = (
                self.env._ref_base_lin_vel_H[1] * filter_joystick
                + (joy_axes[0] / 3.5) * (1 - filter_joystick)
            )
            self.env._ref_base_ang_yaw_dot = (
                self.env._ref_base_ang_yaw_dot * filter_joystick
                + (joy_axes[3] / 2.0) * (1 - filter_joystick)
            )
            self.last_processed_joy_update_id = joy_update_id
            self.joystick_timeout_active = False

            if joy_buttons is not None and len(joy_buttons) > 8 and joy_buttons[8] == 1:
                return {
                    "shutdown_requested": True,
                    "desired_joint_positions": np.zeros(12),
                    "desired_joint_velocities": np.zeros(12),
                    "kp": np.zeros(12),
                    "kd": np.zeros(12),
                }

        self.env.mjData.qpos[0:3] = np.array(position, copy=True)
        self.env.mjData.qvel[0:3] = np.array(linear_velocity, copy=True)

        if config.training_env["use_imu"] or config.training_env["use_concurrent_state_est"]:
            self.env.mjData.qpos[3:7] = np.array(imu_orientation, copy=True)
            self.env.mjData.qvel[3:6] = np.array(imu_angular_velocity, copy=True)
        else:
            self.env.mjData.qpos[3:7] = np.array(orientation, copy=True)
            self.env.mjData.qvel[3:6] = np.array(angular_velocity, copy=True)

        self.env.mjData.qpos[7:] = np.array(joint_positions, copy=True)
        self.env.mjData.qvel[6:] = np.array(joint_velocities, copy=True)
        self.env.mjModel.opt.timestep = simulation_dt
        mujoco.mj_forward(self.env.mjModel, self.env.mjData)

        if joy_message_time is not None and wall_time - joy_message_time > 1.0:
            self.env._ref_base_lin_vel_H[0] = 0.0
            self.env._ref_base_lin_vel_H[1] = 0.0
            self.env._ref_base_ang_yaw_dot = 0.0
            if not self.joystick_timeout_active:
                print("Joystick timeout, stopping the robot")
                self.joystick_timeout_active = True

        env = self.env
        locomotion_policy = self.locomotion_policy

        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame="base")
        base_ang_vel = env.base_ang_vel(frame="base")
        base_ori_euler_xyz = env.base_ori_euler_xyz
        heading_orientation_SO3 = env.heading_orientation_SO3
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos

        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu / 4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]

        self.joint_positions = np.concatenate([joints_pos.FL, joints_pos.FR, joints_pos.RL, joints_pos.RR], axis=0).flatten()

        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu / 4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        if is_rl_activated:
            desired_joint_pos = locomotion_policy.compute_control(
                base_pos=base_pos,
                base_ori_euler_xyz=base_ori_euler_xyz,
                base_quat_wxyz=base_quat_wxyz,
                base_lin_vel=base_lin_vel,
                base_ang_vel=base_ang_vel,
                heading_orientation_SO3=heading_orientation_SO3,
                joints_pos=joints_pos,
                joints_vel=joints_vel,
                ref_base_lin_vel=ref_base_lin_vel,
                ref_base_ang_vel=ref_base_ang_vel,
                imu_linear_acceleration=imu_linear_acceleration,
                imu_angular_velocity=imu_angular_velocity,
                imu_orientation=imu_orientation,
            )

            kp = locomotion_policy.Kp_walking
            kd = locomotion_policy.Kd_walking
        else:
            desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu / 4))) for _ in range(4)])
            desired_joint_pos.FL = self.stand_up_and_down_actions.FL
            desired_joint_pos.FR = self.stand_up_and_down_actions.FR
            desired_joint_pos.RL = self.stand_up_and_down_actions.RL
            desired_joint_pos.RR = self.stand_up_and_down_actions.RR

            kp = locomotion_policy.Kp_stand_up_and_down
            kd = locomotion_policy.Kd_stand_up_and_down

        if USE_MUJOCO_RENDER:
            render_freq = 30
            if wall_time - self.last_render_time > 1.0 / render_freq or env.step_num == 1:
                env.render()
                self.last_render_time = wall_time

        return {
            "shutdown_requested": False,
            "desired_joint_positions": np.concatenate(
                [desired_joint_pos.FL, desired_joint_pos.FR, desired_joint_pos.RL, desired_joint_pos.RR],
                axis=0,
            ).flatten(),
            "desired_joint_velocities": np.zeros(12),
            "kp": np.ones(12) * kp,
            "kd": np.ones(12) * kd,
        }
