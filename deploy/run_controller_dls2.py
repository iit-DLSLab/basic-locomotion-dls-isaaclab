#!/usr/bin/env python3

# Description: dls2 entrypoint for the real robot controller

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from dls2_py.env_bootstrap import setup_environment


APP_ID = "basic_locomotion_controller"
SIGNALS_DOMAIN = 3
BASE_STATE_TOPIC = "rt/base_state"
BLIND_STATE_TOPIC = "rt/blind_state"
IMU_TOPIC = "rt/imu"
TRAJECTORY_GENERATOR_TOPIC = "rt/trajectory_generator"

PeriodicAppPlugin = setup_environment(
    package_name=APP_ID,
    module_file=__file__,
    required_message_modules=("BaseState", "BlindState", "Imu", "TrajectoryGenerator"),
)

import BaseState  # noqa: E402
import BlindState  # noqa: E402
import Imu  # noqa: E402
import TrajectoryGenerator  # noqa: E402


DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT = DIR_PATH.parent


def _resolve_runtime_root() -> Path:
    env_root = os.environ.get("BASIC_LOCOMOTION_CONTROLLER_RUNTIME_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "deploy" / "controller_core.py").exists():
            return candidate

    local_runtime_root = DIR_PATH.parent
    if (local_runtime_root / "deploy" / "controller_core.py").exists():
        return local_runtime_root

    installed_runtime_root = Path("/usr/share/basic_locomotion_controller")
    if (installed_runtime_root / "deploy" / "controller_core.py").exists():
        return installed_runtime_root

    raise FileNotFoundError(
        "Unable to locate the basic locomotion runtime files. "
        "Set BASIC_LOCOMOTION_CONTROLLER_RUNTIME_ROOT to the install root."
    )


RUNTIME_ROOT = _resolve_runtime_root()
DEPLOY_ROOT = RUNTIME_ROOT / "deploy"
os.environ.setdefault("BASIC_LOCOMOTION_CONTROLLER_RUNTIME_ROOT", str(RUNTIME_ROOT))

for candidate in (DEPLOY_ROOT, RUNTIME_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from console import Console  # noqa: E402
from controller_core import ControllerCore  # noqa: E402

import config  # noqa: E402


np.set_printoptions(precision=3, suppress=True)


def configure_process_priority() -> None:
    pid = os.getpid()


class ControllerDLS2(PeriodicAppPlugin):
    def __init__(self) -> None:
        super().__init__(APP_ID, SIGNALS_DOMAIN)

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

        self.reader_base_state = self.build_input(
            BASE_STATE_TOPIC,
            BaseState.BaseStatePubSubType(),
            BaseState.BaseState(),
            auxiliary_callback=self._update_base_state,
            required_on_activation=False,
        )
        self.reader_blind_state = self.build_input(
            BLIND_STATE_TOPIC,
            BlindState.BlindStatePubSubType(),
            BlindState.BlindState(),
            auxiliary_callback=self._update_blind_state,
            required_on_activation=False,
        )
        self.reader_imu = self.build_input(
            IMU_TOPIC,
            Imu.ImuPubSubType(),
            Imu.Imu(),
            auxiliary_callback=self._update_imu,
            required_on_activation=False,
        )

        self.writer_trajectory_generator = self.build_output(
            TRAJECTORY_GENERATOR_TOPIC,
            TrajectoryGenerator.TrajectoryGeneratorPubSubType(),
            TrajectoryGenerator.TrajectoryGenerator(),
        )
        self.sequence_id = 0

        self.console = Console(controller=self.controller)
        thread_console = threading.Thread(target=self.console.interactive_command_line, daemon=True)
        thread_console.start()

    def _update_base_state(self) -> None:
        msg = self.reader_base_state.getData()
        self.position = np.array(msg.pose().position(), copy=True)
        self.orientation = np.roll(np.array(msg.pose().orientation(), copy=True), 1)
        self.linear_velocity = np.array(msg.velocity().linear(), copy=True)
        self.angular_velocity = np.array(msg.velocity().angular(), copy=True)
        self.first_message_base_arrived = True

    def _update_blind_state(self) -> None:
        msg = self.reader_blind_state.getData()
        try:
            positions = msg.joints_position()
            velocities = msg.joints_velocity()
            
            # Verify we got valid data
            if positions is None or len(positions) == 0:
                print("WARNING: Received empty joint_positions from blind_state message")
                return
            if velocities is None or len(velocities) == 0:
                print("WARNING: Received empty joint_velocities from blind_state message")
                return
                
            self.joint_positions = np.array(positions, copy=True)
            self.joint_velocities = np.array(velocities, copy=True)
            
            if len(self.joint_positions) != 12 or len(self.joint_velocities) != 12:
                print(
                    f"WARNING: Unexpected joint array sizes: "
                    f"positions={len(self.joint_positions)}, velocities={len(self.joint_velocities)}"
                )
                return
                
            self.first_message_joints_arrived = True
        except Exception as e:
            print(f"ERROR in _update_blind_state: {e}")
            import traceback
            traceback.print_exc()

    def _update_imu(self) -> None:
        msg = self.reader_imu.getData()
        self.imu_linear_acceleration = np.array(msg.linear_acceleration(), copy=True)
        self.imu_angular_velocity = np.array(msg.angular_velocity(), copy=True)
        self.imu_orientation = np.roll(np.array(msg.orientation(), copy=True), 1)
        self.first_message_imu_arrived = True

    def run(self) -> None:
        self.read()

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
            raise SystemExit(0)

        msg = self.writer_trajectory_generator.data
        msg.timestamp(float(time.time_ns()))
        msg.sequence_id(int(self.sequence_id % 1000))
        self.sequence_id += 1
        msg.joints_position(_as_double_vector(control_output["desired_joint_positions"]))
        msg.joints_velocity(_as_double_vector(control_output["desired_joint_velocities"]))
        msg.kp(_as_double_vector(control_output["kp"]))
        msg.kd(_as_double_vector(control_output["kd"]))

        self.writer_trajectory_generator.write()


def _as_double_vector(values: np.ndarray) -> object:
    vector = TrajectoryGenerator.double_vector()
    for value in values:
        vector.push_back(float(value))
    return vector


def main() -> None:
    print("Hello from basic-locomotion-dls-isaaclab dls2 node.")
    configure_process_priority()

    controller_dls2 = ControllerDLS2()
    controller_dls2.execute()


if __name__ == "__main__":
    main()
