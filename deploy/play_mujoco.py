# Description: This script is used to simulate the full model of the robot in mujoco

# Python imports
import os
import sys
import threading
import time

# Simulation related imports
import mujoco
import mujoco.viewer
import numpy as np
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path + "/mujoco_utils/")
sys.path.append(file_path + "/../")
import mujoco_utils
from heightmap import HeightMap

# Locomotion Policy imports
from deploy import config
from locomotion_policy_wrapper import LocomotionPolicyWrapper


class PlayMujoco:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
        self.simulation_dt = 0.002

        self.mjModel = mujoco.MjModel.from_xml_path(file_path + "/mujoco_utils/robot_model/" + config.robot + "/" + config.scene + ".xml")
        self.mjModel.opt.timestep = self.simulation_dt
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        mujoco.mj_forward(self.mjModel, self.mjData)

        self.viewer = mujoco.viewer.launch_passive(
            self.mjModel,
            self.mjData,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)
        self.last_render_time = time.time()
        self.RENDER_FREQ = 30.0

        self.locomotion_policy = LocomotionPolicyWrapper(mjModel=self.mjModel)

        if self.locomotion_policy.use_vision:
            resolution_heightmap = config.training_env["perceptive_height_scanner"]["pattern_cfg"]["resolution"]
            num_rows_heightmap = round(config.training_env["perceptive_height_scanner"]["pattern_cfg"]["size"][0] / resolution_heightmap) + 1
            num_cols_heightmap = round(config.training_env["perceptive_height_scanner"]["pattern_cfg"]["size"][1] / resolution_heightmap) + 1
            self.heightmap_offset = config.training_env["perceptive_height_scanner"]["offset"]
            self.heightmap = HeightMap(
                num_rows=num_rows_heightmap,
                num_cols=num_cols_heightmap,
                dist_x=resolution_heightmap,
                dist_y=resolution_heightmap,
                mj_model=self.mjModel,
                mj_data=self.mjData,
            )

        self.legs_joints_position = np.zeros(12)
        self.legs_joints_velocity = np.zeros(12)
        self.desired_joint_pos_leg = self.mjData.qpos[7:19].copy()
        self.Kp_legs = 0.0
        self.Kd_legs = 0.0
        self.ref_base_lin_vel_H = np.zeros(3)
        self.ref_base_ang_yaw_dot = 0.0

        from console import Console

        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()
        self.console.isDown = False
        self.console.isRLActivated = True

    def run(self):
        step_num = 1
        while self.viewer.is_running():
            step_start = time.time()

            qpos, qvel = self.mjData.qpos, self.mjData.qvel
            base_lin_vel = mujoco_utils.base_lin_vel(self.mjData, frame="base")
            base_ang_vel = mujoco_utils.base_ang_vel(self.mjData, frame="base")
            base_ori_euler_xyz = mujoco_utils.base_ori_euler_xyz(self.mjData)
            heading_orientation_SO3 = mujoco_utils.heading_orientation_SO3(self.mjData)
            base_quat_wxyz = qpos[3:7]
            base_pos = mujoco_utils.base_pos(self.mjData)
            self.legs_joints_position = qpos[7:19].copy()
            self.legs_joints_velocity = qvel[6:18].copy()

            if config.training_env["use_imu"] or config.training_env["use_concurrent_state_est"]:
                imu_linear_acceleration = self.mjData.sensordata[0:3]
                imu_angular_velocity = self.mjData.sensordata[3:6]
                imu_orientation = self.mjData.sensordata[9:13]
            else:
                imu_linear_acceleration = np.zeros(3)
                imu_angular_velocity = np.zeros(3)
                imu_orientation = np.zeros(4)

            ref_base_lin_vel, ref_base_ang_vel = mujoco_utils.target_base_vel(
                self.mjData,
                self.ref_base_lin_vel_H,
                self.ref_base_ang_yaw_dot,
                frame="world",
            )

            if self.locomotion_policy.use_vision:
                offset_world_frame = self.heightmap_offset["pos"] @ heading_orientation_SO3.T
                self.heightmap.update_height_map(base_pos + offset_world_frame, yaw=base_ori_euler_xyz[2])

            if self.console.isRLActivated and step_num % round(1 / (self.locomotion_policy.RL_FREQ * self.simulation_dt)) == 0:
                self.desired_joint_pos_leg = self.locomotion_policy.compute_control(
                    base_pos=base_pos,
                    base_ori_euler_xyz=base_ori_euler_xyz,
                    base_quat_wxyz=base_quat_wxyz,
                    base_lin_vel=base_lin_vel,
                    base_ang_vel=base_ang_vel,
                    heading_orientation_SO3=heading_orientation_SO3,
                    joints_pos_leg=self.legs_joints_position,
                    joints_vel_leg=self.legs_joints_velocity,
                    ref_base_lin_vel=ref_base_lin_vel,
                    ref_base_ang_vel=ref_base_ang_vel,
                    imu_linear_acceleration=imu_linear_acceleration,
                    imu_angular_velocity=imu_angular_velocity,
                    imu_orientation=imu_orientation,
                    heightmap_data=self.heightmap.data if self.locomotion_policy.use_vision else None,
                )
                self.Kp_legs = self.locomotion_policy.Kp_walking
                self.Kd_legs = self.locomotion_policy.Kd_walking
            else:
                self.Kp_legs = self.locomotion_policy.Kp_stand_up_and_down
                self.Kd_legs = self.locomotion_policy.Kd_stand_up_and_down

            max_torque = self.mjModel.actuator_ctrlrange[0:12, 1] * 0.95
            lower = (-max_torque + self.Kd_legs * self.legs_joints_velocity) / self.Kp_legs
            upper = (max_torque + self.Kd_legs * self.legs_joints_velocity) / self.Kp_legs
            self.desired_joint_pos_leg = np.clip(
                self.desired_joint_pos_leg,
                self.legs_joints_position + lower,
                self.legs_joints_position + upper,
            )

            tau_leg = self.Kp_legs * (self.desired_joint_pos_leg - self.legs_joints_position) - self.Kd_legs * self.legs_joints_velocity
            self.mjData.ctrl[0:12] = tau_leg
            mujoco.mj_step(self.mjModel, self.mjData)
            step_num += 1

            loop_elapsed_time = time.time() - step_start
            if loop_elapsed_time < self.simulation_dt:
                time.sleep(self.simulation_dt - loop_elapsed_time)

            if time.time() - self.last_render_time > 1.0 / self.RENDER_FREQ:
                self.viewer.cam.lookat[:] = base_pos
                self.viewer.sync()
                self.last_render_time = time.time()

                if self.locomotion_policy.use_vision and self.heightmap.data is not None:
                    for i in range(self.heightmap.data.shape[0]):
                        for j in range(self.heightmap.data.shape[1]):
                            self.heightmap.geom_ids[i, j] = mujoco_utils.render_sphere(
                                viewer=self.viewer,
                                position=self.heightmap.data[i, j, 0],
                                diameter=0.02,
                                color=[0, 1, 0, 0.5],
                                geom_id=self.heightmap.geom_ids[i, j],
                            )


if __name__ == "__main__":
    play_mujoco = PlayMujoco()
    play_mujoco.run()
