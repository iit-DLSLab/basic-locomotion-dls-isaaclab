"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from amp_rsl_rl.runners import AMPOnPolicyRunner
import numpy as np
import time

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from isaaclab.utils.dict import print_dict
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

# Import extensions to set up environment tasks
import quadruped_rl_collection.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    dataset_path = (
        os.path.dirname(os.path.realpath(__file__)) + "amp_dataset/flat.npy"
    )
    data = np.load(str(dataset_path), allow_pickle=True).item()
    dataset_joint_names = data["joints_list"]
    dataset_joint_pos = data["joint_positions"]
    dataset_root_position = data["root_position"]
    dataset_root_quaternion = data["root_quaternion"]
    dataset_root_quaternion = np.roll(dataset_root_quaternion, 1)
    dataset_fps = data["fps"]

    # reset environment
    timestep = 0
    env_ids = torch.tensor([0], device=env.unwrapped.device)

    dataset_joint_names = data["joints_list"]

    expected_joint_names = [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ]

    # build index map for expected_joint_names
    idx_map: List[Union[int, None]] = []
    for j in expected_joint_names:
        if j in dataset_joint_names:
            idx_map.append(dataset_joint_names.index(j))
        else:
            idx_map.append(None)

    # reorder & fill joint positions
    jp_list: List[np.ndarray] = []
    for frame in data["joint_positions"]:
        arr = np.zeros((len(idx_map),), dtype=frame.dtype)
        for i, src_idx in enumerate(idx_map):
            if src_idx is not None:
                arr[i] = frame[src_idx]
        jp_list.append(arr)

    dataset_joint_pos = np.stack(jp_list, axis=0)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            positions = torch.tensor(
                dataset_root_position[timestep], dtype=torch.float32, device=env.device
            )
            orientations = torch.tensor(
                dataset_root_quaternion[timestep],
                dtype=torch.float32,
                device=env.device,
            )
            joint_pos = torch.tensor(
                dataset_joint_pos[timestep], dtype=torch.float32, device=env.device
            )

            env.unwrapped._robot.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=env_ids
            )
            env.unwrapped._robot.write_joint_state_to_sim(
                joint_pos, joint_pos * 0.0, env_ids=env_ids
            )
            timestep += 1

            # env stepping
            env.env.render()

            time.sleep(1.0 / dataset_fps)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
