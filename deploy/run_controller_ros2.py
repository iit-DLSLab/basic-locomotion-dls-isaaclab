import os

# Fail-safe: se non è stato scelto esplicitamente altro,
# ROS 2 comunica solamente sulla macchina locale.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

import sys
import shlex
import subprocess
from pathlib import Path

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "ros2_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("BASIC_LOCOMOTION_ROS2_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export BASIC_LOCOMOTION_ROS2_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])


import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker, MarkerArray
from dls2_interface.msg import BaseState, BlindState, Imu, TrajectoryGenerator

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading

import copy

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.heightmap import HeightMap
from gym_quadruped.utils.quadruped_utils import LegsAttr
from gym_quadruped.utils.mujoco.visual import render_sphere


# Locomotion Policy imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper

import config

# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


USE_MUJOCO_RENDER = True


class ControllerROS2(Node):
    def __init__(self):
        super().__init__('ControllerROS2')

        # Mujoco env
        robot_name = config.robot
        scene_name = config.scene
        simulation_dt = 0.002


        # Create the quadruped robot environment -----------------------------------------------------------
        self.env = QuadrupedEnv(
            robot=robot_name,
            scene=scene_name,
            sim_dt=simulation_dt,
            base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
        )
        self.env.reset(random=False)
        
        self.last_render_time = time.time()
        if USE_MUJOCO_RENDER:
            self.env.render()   
                 

        # Subscribers and Publishers
        self.subscription_base_state = self.create_subscription(BaseState,"/base_state", self.get_base_state_callback, 1)
        self.subscription_blind_state = self.create_subscription(BlindState,"blind_state", self.get_blind_state_callback, 1)
        self.subscription_imu = self.create_subscription(Imu,"imu", self.get_imu_callback, 1)
        
        self.subscription_joy = self.create_subscription(Joy,"joy", self.get_joy_callback, 1)
        self.last_joy_time = None
        
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator,"/trajectory_generator", 1)
        self.sequence_id = 0 # To keep track of the last msg sent, useful for debugging and synchronization
        RL_FREQ = 1./(config.training_env["sim"]["dt"]*config.training_env["decimation"])  # Hz, frequency of the RL controller
        self.timer = self.create_timer(1.0/RL_FREQ, self.compute_rl_control)


        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False 
        self.first_message_imu_arrived = False
        self.first_message_heightmap_arrived = False

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)

        # IMU
        self.imu_linear_acceleration = np.zeros(3)
        self.imu_angular_velocity = np.zeros(3)
        self.imu_orientation = np.zeros(4)

        
        # Initialization of variables used in the main control loop --------------------------------
        self.locomotion_policy = LocomotionPolicyWrapper(env=self.env)

        # On perceptive policies, use the ROS marker scan in place of MuJoCo ray casting.
        self.heightmap = None
        self._last_heightmap_warning_time = 0.0
        if self.locomotion_policy.use_vision:
            pattern_cfg = config.training_env["perceptive_height_scanner"]["pattern_cfg"]
            resolution_heightmap = pattern_cfg["resolution"]
            num_rows_heightmap = round(pattern_cfg["size"][0] / resolution_heightmap) + 1
            num_cols_heightmap = round(pattern_cfg["size"][1] / resolution_heightmap) + 1
            self.heightmap = HeightMap(
                num_rows=num_rows_heightmap,
                num_cols=num_cols_heightmap,
                dist_x=resolution_heightmap,
                dist_y=resolution_heightmap,
                mj_model=self.env.mjModel,
                mj_data=self.env.mjData,
            )
            self.subscription_heightmap = self.create_subscription(MarkerArray, "/height_scan_markers", self.get_heightmap_callback, 1,)



        self.stand_up_and_down_actions = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        keyframe_id = mujoco.mj_name2id(self.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        goDown_qpos = self.env.mjModel.key_qpos[keyframe_id]
        self.stand_up_and_down_actions.FL = goDown_qpos[7:10]
        self.stand_up_and_down_actions.FR = goDown_qpos[10:13]
        self.stand_up_and_down_actions.RL = goDown_qpos[13:16]
        self.stand_up_and_down_actions.RR = goDown_qpos[16:19]
        self.joint_positions = goDown_qpos[7:19]


        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

    
    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a 
        8Bitdi Ultimate 2C Wireless Controller.
        """

        filter_joystick = 0.7
        self.env._ref_base_lin_vel_H[0] = self.env._ref_base_lin_vel_H[0]*filter_joystick + (msg.axes[1]/3.5)*(1-filter_joystick)  # Forward/Backward
        self.env._ref_base_lin_vel_H[1] = self.env._ref_base_lin_vel_H[1]*filter_joystick + (msg.axes[0]/3.5)*(1-filter_joystick)  # Left/Right
        self.env._ref_base_ang_yaw_dot = self.env._ref_base_ang_yaw_dot*filter_joystick + (msg.axes[3]/2.)*(1-filter_joystick)  # Yaw

        self.last_joy_time = time.time()

        #kill the node if the button is pressed
        if msg.buttons[8] == 1:
            self.get_logger().info("Joystick button pressed, shutting down the node.") 
            # This will kill the robot hal
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            # This will kill the process running this script
            os.system("pkill -f play_ros2.py") 
            exit(0)


    def get_base_state_callback(self, msg):
        self.position = np.array(msg.pose.position) #world frame
        # For the quaternion, the order is [x, y, z, w] on DLS2 but here we want [w, x, y, z] (mujoco convention)
        self.orientation = np.roll(np.array(msg.pose.orientation), 1) #world frame
        self.linear_velocity = np.array(msg.velocity.linear) #world frame
        self.angular_velocity = np.array(msg.velocity.angular) #base frame

        self.first_message_base_arrived = True


    def get_blind_state_callback(self, msg):
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)

        self.first_message_joints_arrived = True
     
        
    def get_imu_callback(self, msg):
        self.imu_linear_acceleration = np.array(msg.linear_acceleration) 
        self.imu_angular_velocity = np.array(msg.angular_velocity) 
        # For the quaternion, the order is [x, y, z, w] on DLS2 but here we want [w, x, y, z] (mujoco convention)
        self.imu_orientation = np.roll(np.array(msg.orientation), 1) 

        self.first_message_imu_arrived = True


    def get_heightmap_callback(self, msg):
        """Load a complete ``/height_scan_markers`` scan into ``self.heightmap``."""
        try:
            markers = sorted(
                (
                    marker
                    for marker in msg.markers
                    if marker.action == Marker.ADD and marker.type == Marker.SPHERE
                ),
                key=lambda marker: marker.id,
            )
            sample_count = self.heightmap.num_rows * self.heightmap.num_cols
            points = np.array(
                [
                    [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
                    for marker in markers
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(points)):
                raise ValueError("marker positions must be finite")

            # Publisher order is Y-major/X-minor. HeightMap uses descending X/Y.
            grid = points.reshape(self.heightmap.num_cols, self.heightmap.num_rows, 3)
            grid = np.flip(grid.transpose(1, 0, 2), axis=(0, 1))
            self.heightmap.sensor_data_matrix[:] = grid[:, :, None, :]
        except ValueError as error:
            # Keep the last complete scan and throttle malformed-scan warnings.
            now = time.monotonic()
            if now - self._last_heightmap_warning_time >= 2.0:
                self.get_logger().warning(f"Ignoring height-map markers: {error}")
                self._last_heightmap_warning_time = now
            return

        if not self.first_message_heightmap_arrived:
            self.get_logger().info(f"Received the first complete height map ({sample_count} samples)")
        self.first_message_heightmap_arrived = True


    def compute_rl_control(self):
        # Update the loop time
        start_time = time.perf_counter()
        if(self.last_start_time is not None):
            self.loop_time = (start_time - self.last_start_time)
        self.last_start_time = start_time
        simulation_dt = self.loop_time
        

        # Safety check to not do anything until a first base and blind state are received
        if(config.training_env["use_imu"] or config.training_env["use_concurrent_state_est"]):
            if(self.first_message_imu_arrived==False or self.first_message_joints_arrived==False):
                return
        else:
            if(self.first_message_base_arrived==False or self.first_message_joints_arrived==False):
                return
        
        if self.locomotion_policy.use_vision:
            # base_state is needed to transform markers from base_link to world coordinates.
            if not self.first_message_base_arrived or not self.first_message_heightmap_arrived:
                return
        
        # Update the mujoco model
        # Note that in case of IMU or concurrent state estimator, these info below are not used,
        # In the case we have a state estimator, this is usefull only for debugging visually
        self.env.mjData.qpos[0:3] = copy.deepcopy(self.position)
        self.env.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)

        if(config.training_env["use_imu"] or config.training_env["use_concurrent_state_est"]):
            self.env.mjData.qpos[3:7] = copy.deepcopy(self.imu_orientation)
            self.env.mjData.qvel[3:6] = copy.deepcopy(self.imu_angular_velocity)
        else:
            self.env.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
            self.env.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
        
        # These info instead are used for sure in all the cases
        self.env.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
        self.env.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
        self.env.mjModel.opt.timestep = simulation_dt
        mujoco.mj_forward(self.env.mjModel, self.env.mjData) 
        
        # Safety check for joystick timeout
        if(self.last_joy_time is not None and time.time() - self.last_joy_time > 1.0):
            self.env._ref_base_lin_vel_H[0] = 0.0
            self.env._ref_base_lin_vel_H[1] = 0.0
            self.env._ref_base_ang_yaw_dot = 0.0
            print("Joystick timeout, stopping the robot")
            self.last_joy_time = None
            

        env = self.env
        locomotion_policy = self.locomotion_policy
        
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame='base')
        base_ang_vel = env.base_ang_vel(frame='base')
        base_ori_euler_xyz = env.base_ori_euler_xyz
        heading_orientation_SO3 = env.heading_orientation_SO3
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos


        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]

        # variable saved for goDown and goUp motion
        self.joint_positions = np.concatenate([joints_pos.FL, joints_pos.FR, joints_pos.RL, joints_pos.RR], axis=0).flatten()
    
        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        heightmap_data = None
        if locomotion_policy.use_vision:
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, base_quat_wxyz)
            local_points = self.heightmap.sensor_data_matrix.reshape(-1, 3)
            self.heightmap.data = (
                local_points @ rotation.reshape(3, 3).T + base_pos
            ).reshape(self.heightmap.sensor_data_matrix.shape)
            heightmap_data = self.heightmap.data


        if(self.console.isRLActivated):

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
                        imu_linear_acceleration=self.imu_linear_acceleration,
                        imu_angular_velocity=self.imu_angular_velocity,
                        imu_orientation=self.imu_orientation,
                        heightmap_data=heightmap_data)
            
            # Impedence Loop
            Kp = locomotion_policy.Kp_walking
            Kd = locomotion_policy.Kd_walking

        else:
            desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
            desired_joint_pos.FL = self.stand_up_and_down_actions.FL
            desired_joint_pos.FR = self.stand_up_and_down_actions.FR
            desired_joint_pos.RL = self.stand_up_and_down_actions.RL
            desired_joint_pos.RR = self.stand_up_and_down_actions.RR

            # Impedence Loop
            Kp = locomotion_policy.Kp_stand_up_and_down
            Kd = locomotion_policy.Kd_stand_up_and_down

        # Publish the desired joint positions to the trajectory generator --------------------------------
        trajectory_generator_msg = TrajectoryGenerator()
        trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        trajectory_generator_msg.sequence_id = int(self.sequence_id % 1000)  # To avoid overflow, we reset the sequence id after it reaches a certain value
        self.sequence_id += 1
        trajectory_generator_msg.joints_position = np.array([desired_joint_pos.FL, desired_joint_pos.FR, desired_joint_pos.RL, desired_joint_pos.RR]).flatten().tolist()
        trajectory_generator_msg.joints_velocity = np.zeros(12).tolist()
        trajectory_generator_msg.kp = (np.ones(12) * Kp).tolist()
        trajectory_generator_msg.kd = (np.ones(12) * Kd).tolist()

        self.publisher_trajectory_generator.publish(trajectory_generator_msg)
        
        
        # Render the simulation at a certain frequency -----------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30  # Hz
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ or self.env.step_num == 1:
                self.env.render()
                self.last_render_time = time.time()

                if locomotion_policy.use_vision and self.heightmap.data is not None:
                    for i in range(self.heightmap.data.shape[0]):
                        for j in range(self.heightmap.data.shape[1]):
                            self.heightmap.geom_ids[i, j] = render_sphere(
                                viewer=self.env.viewer,
                                position=self.heightmap.data[i, j, 0],
                                diameter=0.02,
                                color=[0, 1, 0, 0.5],
                                geom_id=self.heightmap.geom_ids[i, j],
                            )


#---------------------------
if __name__ == '__main__':
    
    print('Hello from basic-locomotion-dls-isaaclab ros node.')
    
    rclpy.init()
    controller_ros2_node = ControllerROS2()
    rclpy.spin(controller_ros2_node)
    
    controller_ros2_node.destroy_node()
    rclpy.shutdown()

    print("ControllerROS2 node is stopped")
    exit(0)
