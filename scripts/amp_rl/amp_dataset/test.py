import idyntree.bindings as idyntree
import numpy as np
from idyntree.visualize import MeshcatVisualizer
import time
from scipy.spatial.transform import Rotation as R

model_path = "/home/alienware/isaaclab_ws_home/quadruped_rl_collection/scripts/sim_to_others/gym-quadruped/gym_quadruped/robot_model/aliengo/aliengo.urdf"

actuated_joint_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]

dataset_path = "/home/alienware/isaaclab_ws_home/quadruped_rl_collection/dataset/traj_0.npy"
data = np.load(str(dataset_path), allow_pickle=True).item()
actuated_joint_names = data["joints_list"]
print("actuated_joint_names", actuated_joint_names)
aaa

dataset_joint_pos = data["joint_positions"]
dataset_root_position = data["root_position"]
dataset_root_quaternion = data["root_quaternion"]
dataset_root_quaternion = np.roll(dataset_root_quaternion,1)
dataset_fps = data["fps"]


viz = MeshcatVisualizer()
viz.load_model_from_file(model_path=model_path, considered_joints=actuated_joint_names, model_name="aliengo")


def update_the_model(base_pos, base_quat, joint_values):
    w_R_b = R.from_quat(base_quat, scalar_first=True).as_matrix() 
    p = base_pos
    s = joint_values
    viz.set_multibody_system_state(p, w_R_b, s, model_name='aliengo')

input("Press Enter to start the simulation...")


dt = 1.0 / dataset_fps

for i in range(len(dataset_joint_pos)):
    base_pos = dataset_root_position[i]
    base_quat = dataset_root_quaternion[i]
    joint_values = dataset_joint_pos[i]
    update_the_model(base_pos, base_quat, joint_values)
    time.sleep(dt)