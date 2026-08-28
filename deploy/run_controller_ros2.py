# Description: ROS2 entrypoint for the real robot controller

# Authors:
# Giulio Turrisi
import os
import sys
import threading
import time

import numpy as np
import rclpy
from dls2_interface.msg import BaseState, BlindState, Imu, TrajectoryGenerator
from rclpy.node import Node
from sensor_msgs.msg import Joy

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from console import Console
from controller_core import ControllerCore

import config


np.set_printoptions(precision=3, suppress=True)


def configure_process_priority():
    pid = os.getpid()
    print("PID: ", pid)
    os.system("renice -n -21 -p " + str(pid))
    os.system("echo -20 > /proc/" + str(pid) + "/autogroup")


class ControllerROS2(Node):
    def __init__(self):
        super().__init__("ControllerROS2")

        self.controller = ControllerCore()

        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)

        self.imu_linear_acceleration = np.zeros(3)
        self.imu_angular_velocity = np.zeros(3)
        self.imu_orientation = np.zeros(4)

        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False
        self.first_message_imu_arrived = False

        self.joy_axes = None
        self.joy_buttons = None
        self.last_joy_time = None
        self.joy_update_id = 0

        self.subscription_base_state = self.create_subscription(BaseState, "/base_state", self.get_base_state_callback, 1)
        self.subscription_blind_state = self.create_subscription(BlindState, "blind_state", self.get_blind_state_callback, 1)
        self.subscription_imu = self.create_subscription(Imu, "imu", self.get_imu_callback, 1)
        self.subscription_joy = self.create_subscription(Joy, "joy", self.get_joy_callback, 1)

        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator, "/trajectory_generator", 1)
        self.sequence_id = 0
        rl_freq = 1.0 / (config.training_env["sim"]["dt"] * config.training_env["decimation"])
        self.timer = self.create_timer(1.0 / rl_freq, self.compute_rl_control)

        self.console = Console(controller=self.controller)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

    def get_joy_callback(self, msg):
        self.joy_axes = np.array(msg.axes, copy=True)
        self.joy_buttons = np.array(msg.buttons, copy=True)
        self.last_joy_time = time.time()
        self.joy_update_id += 1

    def get_base_state_callback(self, msg):
        self.position = np.array(msg.pose.position)
        self.orientation = np.roll(np.array(msg.pose.orientation), 1)
        self.linear_velocity = np.array(msg.velocity.linear)
        self.angular_velocity = np.array(msg.velocity.angular)
        self.first_message_base_arrived = True

    def get_blind_state_callback(self, msg):
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)
        self.first_message_joints_arrived = True

    def get_imu_callback(self, msg):
        self.imu_linear_acceleration = np.array(msg.linear_acceleration)
        self.imu_angular_velocity = np.array(msg.angular_velocity)
        self.imu_orientation = np.roll(np.array(msg.orientation), 1)
        self.first_message_imu_arrived = True

    def compute_rl_control(self):
        control_output = self.controller.compute_control_step(
            position=self.position,
            orientation=self.orientation,
            linear_velocity=self.linear_velocity,
            angular_velocity=self.angular_velocity,
            joint_positions=self.joint_positions,
            joint_velocities=self.joint_velocities,
            imu_linear_acceleration=self.imu_linear_acceleration,
            imu_angular_velocity=self.imu_angular_velocity,
            imu_orientation=self.imu_orientation,
            base_state_received=self.first_message_base_arrived,
            joints_state_received=self.first_message_joints_arrived,
            imu_state_received=self.first_message_imu_arrived,
            joy_axes=self.joy_axes,
            joy_buttons=self.joy_buttons,
            joy_message_time=self.last_joy_time,
            joy_update_id=self.joy_update_id,
            is_rl_activated=self.console.isRLActivated,
            monotonic_time=time.perf_counter(),
            wall_time=time.time(),
        )

        if control_output is None:
            return

        if control_output["shutdown_requested"]:
            self._shutdown_from_joystick()
            return

        trajectory_generator_msg = TrajectoryGenerator()
        trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        trajectory_generator_msg.sequence_id = int(self.sequence_id % 1000)
        self.sequence_id += 1
        trajectory_generator_msg.joints_position = control_output["desired_joint_positions"].tolist()
        trajectory_generator_msg.joints_velocity = control_output["desired_joint_velocities"].tolist()
        trajectory_generator_msg.kp = control_output["kp"].tolist()
        trajectory_generator_msg.kd = control_output["kd"].tolist()

        self.publisher_trajectory_generator.publish(trajectory_generator_msg)

    def _shutdown_from_joystick(self):
        self.get_logger().info("Joystick button pressed, shutting down the node.")
        os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
        os.system("pkill -f play_ros2.py")
        rclpy.shutdown()
        raise SystemExit(0)


def main():
    print("Hello from basic-locomotion-dls-isaaclab ros node.")
    configure_process_priority()

    rclpy.init()
    controller_ros2_node = ControllerROS2()
    rclpy.spin(controller_ros2_node)

    controller_ros2_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    print("ControllerROS2 node is stopped")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
