import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'go2'  # 'aliengo', 'go2', 'b2', 'hyqreal2' 
scene = 'random_boxes'  # flat, random_boxes, random_pyramids, perlin

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 21.5
    Kd_walking = 3.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

    policy_folder_path = dir_path + "/../tested_policies/" + robot + "/symm"

elif(robot == "go2"):
    Kp_walking = 20.0
    Kd_walking = 2.0 #1.5

    Kp_stand_up_and_down = 50.
    Kd_stand_up_and_down = 5.

    policy_folder_path = dir_path + "/../tested_policies/" + robot + "/2026-08-30_21-23-07"

elif(robot == "b2"):
    Kp_walking = 100.
    Kd_walking = 5.

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.
elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.

    Kp_stand_up_and_down = 175.
    Kd_stand_up_and_down = 20.
else:
    raise ValueError(f"Robot {robot} not supported")

# ----------------------------------------------------------------------------------------------------------------

concurrent_state_est_network = policy_folder_path + "/exported/concurrent_state_estimator.pth"
rma_network = policy_folder_path + "/exported/rma.pth"

# Load specific training parameters
import yaml 
with open(policy_folder_path + "/params/env.yaml", "r") as file:
    training_env = yaml.unsafe_load(file)
