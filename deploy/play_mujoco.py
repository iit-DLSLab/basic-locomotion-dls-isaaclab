# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi

import time
import numpy as np
from tqdm import tqdm
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

from gym_quadruped.sensors.heightmap import HeightMap
from gym_quadruped.utils.mujoco.visual import render_sphere

# Locomotion Policy imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper

import config


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    robot_name = config.robot
    scene_name = config.scene
    simulation_dt = 0.002


    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
    )


    env.reset(random=False)
    env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType
    env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False



    # Initialization of variables used in the main control loop --------------------------------
    locomotion_policy = LocomotionPolicyWrapper(env=env)

    if(locomotion_policy.use_vision):
        resolution_heightmap = config.training_env["perceptive_height_scanner"]["pattern_cfg"]["resolution"]
        num_rows_heightmap = (
            round(
                config.training_env["perceptive_height_scanner"]["pattern_cfg"]["size"][0] / resolution_heightmap
            )
            + 1
        )
        num_cols_heightmap = (
            round(
                config.training_env["perceptive_height_scanner"]["pattern_cfg"]["size"][1] / resolution_heightmap
            )
            + 1
        )
        heightmap_offset = config.training_env["perceptive_height_scanner"]["offset"]
        heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData)     
    

    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    last_render_time = time.time()

    while True:
        step_start = time.time()
        
        # Get the current state of the robot -----------------------------------------------------
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame='base')
        base_ang_vel = env.base_ang_vel(frame='base')
        base_ori_euler_xyz = env.base_ori_euler_xyz
        heading_orientation_SO3 = env.heading_orientation_SO3
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos

        if(config.training_env["use_imu"] or config.training_env["use_concurrent_state_est"]):
            imu_linear_acceleration = env.mjData.sensordata[0:3]
            imu_angular_velocity = env.mjData.sensordata[3:6]
            imu_orientation = env.mjData.sensordata[9:13]
            imu_orientation = base_quat_wxyz
        else:
            imu_linear_acceleration = np.zeros(3)
            imu_angular_velocity = np.zeros(3)
            imu_orientation = np.zeros(4)

        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]
    
        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        if(locomotion_policy.use_vision):
            offset_world_frame = heightmap_offset["pos"] @ heading_orientation_SO3.T
            heightmap.update_height_map(env.mjData.qpos[0:3] + offset_world_frame, yaw=env.base_ori_euler_xyz[2])

        # RL controller --------------------------------------------------------------
        if env.step_num % round(1 / (locomotion_policy.RL_FREQ * simulation_dt)) == 0:            
            
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
                        heightmap_data=heightmap.data if locomotion_policy.use_vision else None)

        # PD controller --------------------------------------------------------------
        else:
            desired_joint_pos = locomotion_policy.desired_joint_pos


        Kp = locomotion_policy.Kp_walking
        Kd = locomotion_policy.Kd_walking

        error_joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        error_joints_pos.FL = desired_joint_pos.FL - joints_pos.FL
        error_joints_pos.FR = desired_joint_pos.FR - joints_pos.FR
        error_joints_pos.RL = desired_joint_pos.RL - joints_pos.RL
        error_joints_pos.RR = desired_joint_pos.RR - joints_pos.RR
        
        tau = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        tau.FL = Kp * (error_joints_pos.FL) - Kd * joints_vel.FL
        tau.FR = Kp * (error_joints_pos.FR) - Kd * joints_vel.FR
        tau.RL = Kp * (error_joints_pos.RL) - Kd * joints_vel.RL
        tau.RR = Kp * (error_joints_pos.RR) - Kd * joints_vel.RR


        # Set control and mujoco step ----------------------------------------------------------------------
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL.reshape((3,))
        action[env.legs_tau_idx.FR] = tau.FR.reshape((3,))
        action[env.legs_tau_idx.RL] = tau.RL.reshape((3,))
        action[env.legs_tau_idx.RR] = tau.RR.reshape((3,))
        state, reward, is_terminated, is_truncated, info = env.step(action=action)


        # Sleep to match real-time ---------------------------------------------------------
        loop_elapsed_time = time.time() - step_start

        if(loop_elapsed_time < simulation_dt):
            time.sleep(simulation_dt - (loop_elapsed_time))

        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1:
            env.render()
            last_render_time = time.time()

            if(locomotion_policy.use_vision):
                if heightmap.data is not None:
                    for i in range(heightmap.data.shape[0]):
                        for j in range(heightmap.data.data.shape[1]):
                            heightmap.geom_ids[i, j] = render_sphere(
                                viewer=env.viewer,
                                position=([heightmap.data[i][j][0][0], heightmap.data[i][j][0][1], heightmap.data[i][j][0][2]]),
                                diameter=0.02,
                                color=[0, 1, 0, 0.5],
                                geom_id=heightmap.geom_ids[i, j],
                            )


                

    env.close()
