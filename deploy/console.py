import readline
import time
import sys
import atexit
import tty
import termios

import readchar

from gym_quadruped.utils.quadruped_utils import LegsAttr

import numpy as np
import copy
import mujoco

class Console():
    def __init__(self, controller=None, controller_node=None):
        self.controller = controller if controller is not None else controller_node

        self.isDown = True
        self.isRLActivated = False
        
        # Store original terminal settings for restoration
        try:
            self.terminal_fd = sys.stdin.fileno()
            self.original_settings = termios.tcgetattr(self.terminal_fd)
        except (OSError, termios.error):
            # If terminal settings can't be obtained, just skip restoration
            self.terminal_fd = None
            self.original_settings = None
        
        # Register cleanup function
        atexit.register(self._restore_terminal)

        # Autocomplete setup
        self.commands = [
            "help", "ictp", "goUp", "goDown", "activate", "ictp", "setKp", "setKd"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")

    def _restore_terminal(self):
        """Restore terminal to original settings on exit."""
        if self.original_settings is not None:
            try:
                termios.tcsetattr(self.terminal_fd, termios.TCSADRAIN, self.original_settings)
                # Ensure echo is enabled
                sys.stdout.flush()
            except Exception:
                pass


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "goUp"):
                    print("Going Up")
                    if(not self.isDown):
                        print("The robot is already up")
                        continue

                                        
                    start_time = time.time()
                    time_motion = 5.

                    temp = copy.deepcopy(self.controller.joint_positions)
                    initial_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller.env.mjModel.nu/4))) for _ in range(4)])
                    initial_joint_positions.FL = temp[0:3]
                    initial_joint_positions.FR = temp[3:6]
                    initial_joint_positions.RL = temp[6:9]
                    initial_joint_positions.RR = temp[9:12]

                    reference_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller.env.mjModel.nu/4))) for _ in range(4)])
                    keyframe_id = mujoco.mj_name2id(self.controller.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
                    standUp_qpos = self.controller.env.mjModel.key_qpos[keyframe_id]
                    
                    reference_joint_positions.FL = standUp_qpos[7:10]
                    reference_joint_positions.FR = standUp_qpos[10:13]
                    reference_joint_positions.RL = standUp_qpos[13:16]
                    reference_joint_positions.RR = standUp_qpos[16:19]

                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        alpha = time_diff / time_motion
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(initial_joint_positions, reference_joint_positions)
                        ]

                        self.controller.stand_up_and_down_actions.FL = interpolated_positions[0]
                        self.controller.stand_up_and_down_actions.FR = interpolated_positions[1]
                        self.controller.stand_up_and_down_actions.RL = interpolated_positions[2]
                        self.controller.stand_up_and_down_actions.RR = interpolated_positions[3]

                        time.sleep(0.01)

                    self.isDown = False


                elif(input_string == "goDown"):
                    print("Going Down")
                    if(self.isDown):
                        print("The robot is already down")
                        continue

                    self.isDown = True
                    self.isRLActivated = False

                    start_time = time.time()
                    time_motion = 5.

                    #if(self.isRLActivated):
                    #    initial_joint_positions = copy.deepcopy(self.controller_node.locomotion_policy.desired_joint_pos)
                    #else:
                    temp = copy.deepcopy(self.controller.joint_positions)
                    initial_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller.env.mjModel.nu/4))) for _ in range(4)])
                    initial_joint_positions.FL = temp[0:3]
                    initial_joint_positions.FR = temp[3:6]
                    initial_joint_positions.RL = temp[6:9]
                    initial_joint_positions.RR = temp[9:12]
                    
                    keyframe_id = mujoco.mj_name2id(self.controller.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
                    goDown_qpos = self.controller.env.mjModel.key_qpos[keyframe_id]
                    reference_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller.env.mjModel.nu/4))) for _ in range(4)])
                    reference_joint_positions.FL = goDown_qpos[7:10]
                    reference_joint_positions.FR = goDown_qpos[10:13]
                    reference_joint_positions.RL = goDown_qpos[13:16]
                    reference_joint_positions.RR = goDown_qpos[16:19]

                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        alpha = time_diff / time_motion
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(initial_joint_positions, reference_joint_positions)
                        ]
            
                        self.controller.stand_up_and_down_actions.FL = interpolated_positions[0]
                        self.controller.stand_up_and_down_actions.FR = interpolated_positions[1]
                        self.controller.stand_up_and_down_actions.RL = interpolated_positions[2]
                        self.controller.stand_up_and_down_actions.RR = interpolated_positions[3]

                        time.sleep(0.01)

                    self.isRLActivated = False

                    
                elif(input_string == "activate"):
                    self.isRLActivated = not self.isRLActivated


                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller.locomotion_policy.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller.locomotion_policy.Kp_stand_up_and_down= float(temp)
                    
                    print("Kp walking: ", self.controller.locomotion_policy.Kp_walking)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller.locomotion_policy.Kp_walking = float(temp)
                

                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller.locomotion_policy.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller.locomotion_policy.Kd_stand_up_and_down = float(temp)

                    print("Kd walking: ", self.controller.locomotion_policy.Kd_walking)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller.locomotion_policy.Kd_walking = float(temp)
                
                elif(input_string == "ictp"):
                    if readchar is None:
                        print("Interactive keyboard control is unavailable: missing Python module 'readchar'.")
                        continue
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    try:
                        while True:
                            try:
                                command = readchar.readkey()
                            except KeyboardInterrupt:
                                print("\nExiting interactive mode")
                                break
                            except Exception as e:
                                print(f"\nError reading key: {e}")
                                break
                            
                            if command == "w":
                                self.controller.env._ref_base_lin_vel_H[0] += 0.1
                                print("w", end="\r")
                            elif command == "s":
                                self.controller.env._ref_base_lin_vel_H[0] -= 0.1
                                print("s", end="\r")
                            elif command == "a":
                                self.controller.env._ref_base_lin_vel_H[1] += 0.1
                                print("a", end="\r")
                            elif command == "d":
                                self.controller.env._ref_base_lin_vel_H[1] -= 0.1
                                print("d", end="\r")
                            elif command == "q":
                                self.controller.env._ref_base_ang_yaw_dot += 0.1
                                print("q", end="\r")
                            elif command == "e":
                                self.controller.env._ref_base_ang_yaw_dot -= 0.1
                                print("e", end="\r")
                            elif command == "0":
                                self.controller.env._ref_base_lin_vel_H[0] = 0
                                self.controller.env._ref_base_lin_vel_H[1] = 0
                                self.controller.env._ref_base_ang_yaw_dot = 0 
                                print("0", end="\r")
                            elif command == "1":
                                self.pitch_delta -= 0.1
                                print("1", end="\r")
                            elif command == "2":
                                self.pitch_delta = 0
                                print("2", end="\r")
                            elif command == "3":
                                self.pitch_delta += 0.1
                                print("3", end="\r")
                            else:
                                self.controller.env._ref_base_lin_vel_H[0] = 0
                                self.controller.env._ref_base_lin_vel_H[1] = 0
                                self.controller.env._ref_base_ang_yaw_dot = 0 
                                print("\nExiting interactive mode")
                                break
                    finally:
                        # Ensure terminal is restored
                        self._restore_terminal()

            
            
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("ictp: Interactive Keyboard Control")
        print("goUp: Make the robot stand up")
        print("goDown: Make the robot lie down")
        print("activate: Activate/Deactivate the RL policy during walking")
        print("setKp: Set the Kp gains for the PD controller")
        print("setKd: Set the Kd gains for the PD controller\n")
