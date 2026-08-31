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
from dls2_interface.msg import BaseState, BlindState, Imu, ControlSignal
from unitree_go.msg import LowState, MotorState, IMUState

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

# Config imports
import config as cfg


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


USE_SCHEDULER = True # Use the scheduler to compute the control signal
SCHEDULER_FREQ = 500 # Frequency of the scheduler
RENDER_FREQ = 30

# Shell for the controllers ----------------------------------------------
class SimulatorROS2(Node):
    def __init__(self):
        super().__init__('SimulatorROS2')

        # Subscribers and Publishers
        self.publisher_base_state = self.create_publisher(BaseState,"base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState,"blind_state", 1)
        self.publisher_imu = self.create_publisher(Imu,"imu", 1)

        self.publisher_low_state = self.create_publisher(LowState,"lowstate", 1)

        self.subscriber_control_signal = self.create_subscription(ControlSignal,"control_signal", self.get_control_signal_callback, 1)

        self.timer = self.create_timer(1.0/SCHEDULER_FREQ, self.compute_simulator_step_callback)


        # Mujoco env
        self.env = QuadrupedEnv(
            robot=cfg.robot,
            scene=cfg.scene,
            sim_dt=1.0/SCHEDULER_FREQ,
            base_vel_command_type="human"
        )
        self.env.reset(random=False)

        # Names of the actuated joints, in the same order as qpos[7:]/qvel[6:]
        self.joint_names = [
            name for name, info in self.env.joint_info.items() if info.type != mujoco.mjtJoint.mjJNT_FREE
        ]


        self.last_render_time = time.time()
        self.env.render()  
        self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        # Desired PD 
        self.desired_joints_position = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])
        self.desired_joints_velocity = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])

        # Desired gains
        self.Kp = 0
        self.Kd = 0


    def get_control_signal_callback(self, msg):

        joints_position = np.array(msg.joints_position)

        self.desired_joints_position.FL = joints_position[0:3]
        self.desired_joints_position.FR = joints_position[3:6]
        self.desired_joints_position.RL = joints_position[6:9]
        self.desired_joints_position.RR = joints_position[9:12]

        self.Kp = np.array(msg.kp)[0]
        self.Kd = np.array(msg.kd)[0]
        

    def compute_simulator_step_callback(self):

        qpos, qvel = self.env.mjData.qpos, self.env.mjData.qvel
        base_lin_vel = self.env.base_lin_vel(frame='world')
        base_ang_vel = self.env.base_ang_vel(frame='base')
        base_pos = self.env.base_pos

        # Publish Base State ------------------------------------------------
        base_state_msg = BaseState()
        base_state_msg.pose.position = base_pos
        base_state_msg.pose.orientation = np.roll(self.env.mjData.qpos[3:7],-1)
        base_state_msg.velocity.linear = base_lin_vel
        base_state_msg.velocity.angular = base_ang_vel
        self.publisher_base_state.publish(base_state_msg)


        # Publish Blind State ------------------------------------------------
        blind_state_msg = BlindState()
        blind_state_msg.joints_name = self.joint_names
        blind_state_msg.joints_position = self.env.mjData.qpos[7:].tolist()
        blind_state_msg.joints_velocity = self.env.mjData.qvel[6:].tolist()
        self.publisher_blind_state.publish(blind_state_msg)


        # Publish IMU ------------------------------------------------
        imu_msg = Imu()
        imu_msg.linear_acceleration = self.env.mjData.sensordata[0:3]
        imu_msg.angular_velocity = self.env.mjData.sensordata[3:6]
        # To be compliant with our hal, we expect the xyzw order, 
        # but mujoco gives us wxyz, so we roll the array to get the correct order
        imu_msg.orientation = np.roll(np.array(self.env.mjData.sensordata[9:13]), -1) 
        self.publisher_imu.publish(imu_msg)


        # Publish Low State of Unitree ------------------------------------------------
        # FR, FL, RR, RL convention to follow the unitree standard msgs
        lowstate_msg = LowState()
        for i in range(3):
            lowstate_msg.motor_state[i].q = self.env.mjData.qpos[self.env.legs_qpos_idx.FR[i]]
            lowstate_msg.motor_state[i].dq = self.env.mjData.qvel[self.env.legs_qvel_idx.FR[i]]
        for i in range(3):
            lowstate_msg.motor_state[i+3].q = self.env.mjData.qpos[self.env.legs_qpos_idx.FL[i]]
            lowstate_msg.motor_state[i+3].dq = self.env.mjData.qvel[self.env.legs_qvel_idx.FL[i]]
        for i in range(3):
            lowstate_msg.motor_state[i+6].q = self.env.mjData.qpos[self.env.legs_qpos_idx.RR[i]]
            lowstate_msg.motor_state[i+6].dq = self.env.mjData.qvel[self.env.legs_qvel_idx.RR[i]]
        for i in range(3):
            lowstate_msg.motor_state[i+9].q = self.env.mjData.qpos[self.env.legs_qpos_idx.RL[i]]
            lowstate_msg.motor_state[i+9].dq = self.env.mjData.qvel[self.env.legs_qvel_idx.RL[i]]

        _, _, feet_GRF = self.env.feet_contact_state(ground_reaction_forces=True)
        lowstate_msg.foot_force = np.array([feet_GRF.FR[2], feet_GRF.FL[2], feet_GRF.RR[2], feet_GRF.RL[2]], dtype=np.int16)
        lowstate_msg.imu_state.accelerometer = self.env.mjData.sensordata[0:3].astype(np.float32)
        lowstate_msg.imu_state.gyroscope = self.env.mjData.sensordata[3:6].astype(np.float32)
        lowstate_msg.imu_state.quaternion = np.roll(np.array(self.env.mjData.sensordata[9:13].astype(np.float32)), -1)  
        self.publisher_low_state.publish(lowstate_msg)


        # Step the environment --------------------------------------------------------------------------------
        joints_pos = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[self.env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[self.env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[self.env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[self.env.legs_qpos_idx.RR]


        joints_vel = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[self.env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[self.env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[self.env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[self.env.legs_qvel_idx.RR]


        action = np.zeros(self.env.mjModel.nu)
        action[self.env.legs_tau_idx.FL] = self.Kp*(self.desired_joints_position.FL.reshape(-1) - joints_pos.FL) - self.Kd*(joints_vel.FL)
        action[self.env.legs_tau_idx.FR] = self.Kp*(self.desired_joints_position.FR.reshape(-1) - joints_pos.FR) - self.Kd*(joints_vel.FR)
        action[self.env.legs_tau_idx.RL] = self.Kp*(self.desired_joints_position.RL.reshape(-1) - joints_pos.RL) - self.Kd*(joints_vel.RL)
        action[self.env.legs_tau_idx.RR] = self.Kp*(self.desired_joints_position.RR.reshape(-1) - joints_pos.RR) - self.Kd*(joints_vel.RR)
        self.env.step(action=action)


        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
            self.env.render()
            self.last_render_time = time.time()


def main():
    print('Hello from the SimulatorROS2 node.')
    rclpy.init()

    simulator_ros2_node = SimulatorROS2()

    rclpy.spin(simulator_ros2_node)
    simulator_ros2_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
