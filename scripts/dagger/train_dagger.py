# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a depth-conditioned student policy with online DAgger supervision."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from contextlib import nullcontext

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a depth-conditioned DAgger policy from an RSL-RL teacher.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--depth_history_length",
    type=int,
    default=5,
    help="Number of depth frames consumed by the student depth GRU.",
)
parser.add_argument(
    "--dagger_buffer_size",
    type=int,
    default=2048,
    help="Maximum number of aggregated in-memory DAgger samples.",
)
parser.add_argument(
    "--dagger_samples_per_step",
    type=int,
    default=64,
    help="Maximum number of environments added to the DAgger buffer per simulator step.",
)
parser.add_argument("--dagger_batch_size", type=int, default=64, help="Total batch size sampled from the CPU buffer.")
parser.add_argument(
    "--dagger_train_micro_batch_size",
    type=int,
    default=16,
    help="GPU micro-batch size for each behavior-cloning update.",
)
parser.add_argument(
    "--dagger_inference_batch_size",
    type=int,
    default=32,
    help="Maximum number of student-controlled environments evaluated on GPU at once.",
)
parser.add_argument("--dagger_learning_rate", type=float, default=3e-4, help="Student optimizer learning rate.")
parser.add_argument(
    "--disable_dagger_amp",
    action="store_true",
    default=False,
    help="Disable CUDA autocast for the DAgger student.",
)
parser.add_argument(
    "--dagger_train_every",
    type=int,
    default=4,
    help="Run student gradient updates every N simulator steps.",
)
parser.add_argument(
    "--dagger_updates_per_train",
    type=int,
    default=1,
    help="Number of student mini-batch updates each time training is triggered.",
)
parser.add_argument(
    "--dagger_warmup_steps",
    type=int,
    default=100,
    help="Number of initial simulator steps executed only with the teacher while filling the buffer.",
)
parser.add_argument(
    "--expert_beta_start",
    type=float,
    default=1.0,
    help="Initial probability of executing the teacher action after warmup.",
)
parser.add_argument(
    "--expert_beta_end",
    type=float,
    default=0.0,
    help="Final probability of executing the teacher action.",
)
parser.add_argument(
    "--expert_beta_decay_steps",
    type=int,
    default=10000,
    help="Number of simulator steps used to linearly decay teacher action mixing.",
)
parser.add_argument(
    "--max_training_steps",
    type=int,
    default=None,
    help="Optional maximum number of simulator steps. Defaults to running until the app closes.",
)
parser.add_argument(
    "--dagger_policy_path",
    type=str,
    default=None,
    help="Where to save the student policy checkpoint. Defaults to <teacher_run_dir>/dagger_policy.pt.",
)
parser.add_argument(
    "--dagger_save_interval",
    type=int,
    default=10000,
    help="Save a student policy checkpoint every N simulator steps. Use 0 to disable periodic saves.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from collections import deque

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
# Import extensions to set up environment tasks
import basic_locomotion_isaaclab.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

from dagger_network import DaggerNet, DaggerReplayBuffer


def _sanitize_depth_data(env: RslRlVecEnvWrapper) -> torch.Tensor:
    if not hasattr(env.unwrapped, "_depth_camera"):
        raise RuntimeError(
            "The DAgger student needs env.unwrapped._depth_camera, matching collect_depth_to_heightmap.py. "
            "Run this with a depth-enabled vision task/config."
        )
    depth_data = env.unwrapped._depth_camera.data.output["distance_to_image_plane"]
    depth_data = torch.nan_to_num(depth_data, nan=0.0, posinf=1.0, neginf=-1.0)
    depth_data = depth_data.clip(-2.0, 2.0)
    depth_data = depth_data.permute(0, 3, 1, 2)
    return depth_data


def _depth_sequence_from_history(
    depth_history: torch.Tensor,
    history_index: int,
    env_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    history_length = depth_history.shape[0]
    ordered_history = torch.cat(
        (
            torch.arange(history_index + 1, history_length, device=depth_history.device),
            torch.arange(0, history_index + 1, device=depth_history.device),
        )
    )
    if env_indices is not None:
        depth_history = depth_history.index_select(1, env_indices.to(device=depth_history.device))
    return depth_history.index_select(0, ordered_history).permute(1, 0, 2, 3, 4).contiguous()


def _sample_env_indices(num_envs: int, max_samples: int | None) -> torch.Tensor | None:
    if max_samples is None or max_samples <= 0 or max_samples >= num_envs:
        return None
    return torch.randperm(num_envs)[:max_samples]


def _use_cuda_amp(device: torch.device | str) -> bool:
    return not args_cli.disable_dagger_amp and torch.device(device).type == "cuda"


def _autocast_context(device: torch.device | str):
    if _use_cuda_amp(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _predict_student_actions_chunked(
    dagger_net: DaggerNet,
    depth_history: torch.Tensor,
    history_index: int,
    common_obs: torch.Tensor,
    env_indices_cpu: torch.Tensor | None,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if env_indices_cpu is None:
        total_count = common_obs.shape[0]
    else:
        total_count = env_indices_cpu.numel()

    student_actions_batches: list[torch.Tensor] = []
    student_index_batches: list[torch.Tensor] = []
    chunk_size = max(1, args_cli.dagger_inference_batch_size)

    for start in range(0, total_count, chunk_size):
        stop = min(start + chunk_size, total_count)
        if env_indices_cpu is None:
            chunk_indices_cpu = torch.arange(start, stop)
        else:
            chunk_indices_cpu = env_indices_cpu[start:stop]

        chunk_depth_cpu = _depth_sequence_from_history(
            depth_history=depth_history,
            history_index=history_index,
            env_indices=chunk_indices_cpu,
        )
        chunk_depth = chunk_depth_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        chunk_indices_gpu = chunk_indices_cpu.to(device=device)
        chunk_common_obs = common_obs.index_select(0, chunk_indices_gpu)

        with torch.inference_mode(), _autocast_context(device):
            chunk_actions, _ = dagger_net(chunk_depth, chunk_common_obs, hidden=None)

        student_actions_batches.append(chunk_actions.detach().to(dtype=common_obs.dtype))
        if env_indices_cpu is not None:
            student_index_batches.append(chunk_indices_gpu)

        del chunk_depth_cpu, chunk_depth, chunk_common_obs

    student_actions = torch.cat(student_actions_batches, dim=0)
    student_indices_gpu = torch.cat(student_index_batches, dim=0) if student_index_batches else None
    return student_actions, student_indices_gpu


def _teacher_beta(step: int) -> float:
    if step < args_cli.dagger_warmup_steps:
        return 1.0

    if args_cli.expert_beta_decay_steps <= 0:
        return args_cli.expert_beta_end

    decay_step = step - args_cli.dagger_warmup_steps
    progress = min(1.0, max(0.0, decay_step / args_cli.expert_beta_decay_steps))
    return args_cli.expert_beta_start + progress * (args_cli.expert_beta_end - args_cli.expert_beta_start)


def _save_dagger_policy(
    path: str,
    dagger_net: DaggerNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    updates: int,
    metadata: dict,
) -> None:
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": dagger_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "updates": updates,
            "metadata": metadata,
        },
        path,
    )
    print(f"[INFO] Saved DAgger student policy checkpoint to: {path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train a depth-conditioned student with online DAgger supervision."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.depth_history_length <= 0:
        raise ValueError("--depth_history_length must be positive.")
    if args_cli.dagger_train_every <= 0:
        raise ValueError("--dagger_train_every must be positive.")
    if args_cli.dagger_updates_per_train <= 0:
        raise ValueError("--dagger_updates_per_train must be positive.")
    if args_cli.dagger_batch_size <= 0:
        raise ValueError("--dagger_batch_size must be positive.")
    if args_cli.dagger_train_micro_batch_size <= 0:
        raise ValueError("--dagger_train_micro_batch_size must be positive.")
    if args_cli.dagger_inference_batch_size <= 0:
        raise ValueError("--dagger_inference_batch_size must be positive.")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "dagger"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    teacher_policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    current_depth_cpu = _sanitize_depth_data(env).detach().to(device="cpu", dtype=torch.float16)

    num_envs = current_depth_cpu.shape[0]
    common_obs_size = obs["common"].shape[-1]
    single_action_space = getattr(env, "single_action_space", None)
    if single_action_space is None:
        single_action_space = getattr(env.unwrapped, "single_action_space", None)
    action_size = (
        gym.spaces.flatdim(single_action_space) if single_action_space is not None else env.action_space.shape[-1]
    )
    depth_channels = current_depth_cpu.shape[1]
    device = env.unwrapped.device

    depth_history = current_depth_cpu.unsqueeze(0).repeat(args_cli.depth_history_length, 1, 1, 1, 1).contiguous()
    history_index = args_cli.depth_history_length - 1

    dagger_net = DaggerNet(
        vec_size=common_obs_size,
        output_size=action_size,
        depth_channels=depth_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(dagger_net.parameters(), lr=args_cli.dagger_learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=_use_cuda_amp(device))
    loss_fn = torch.nn.MSELoss()
    replay_buffer = DaggerReplayBuffer(capacity=args_cli.dagger_buffer_size)
    recent_losses: deque[float] = deque(maxlen=100)

    policy_path = (
        os.path.abspath(args_cli.dagger_policy_path)
        if args_cli.dagger_policy_path
        else os.path.join(log_dir, "dagger_policy.pt")
    )
    metadata = {
        "task": args_cli.task,
        "teacher_checkpoint_path": resume_path,
        "robot_obs_key": "common",
        "depth_history_length": args_cli.depth_history_length,
        "depth_history_storage": "cpu_float16",
        "depth_image_size": tuple(current_depth_cpu.shape[-2:]),
        "depth_channels": depth_channels,
        "common_obs_size": common_obs_size,
        "action_size": action_size,
        "num_envs": num_envs,
        "dagger_amp": _use_cuda_amp(device),
        "dagger_train_micro_batch_size": args_cli.dagger_train_micro_batch_size,
        "dagger_inference_batch_size": args_cli.dagger_inference_batch_size,
    }

    print(
        "[INFO] Starting online DAgger training "
        f"(num_envs={num_envs}, depth_history={args_cli.depth_history_length}, "
        f"buffer={args_cli.dagger_buffer_size}, batch={args_cli.dagger_batch_size}, "
        f"train_micro_batch={args_cli.dagger_train_micro_batch_size}, "
        f"inference_batch={args_cli.dagger_inference_batch_size}, amp={_use_cuda_amp(device)})."
    )

    step = 0
    updates = 0

    while simulation_app.is_running() and (
        args_cli.max_training_steps is None or step < args_cli.max_training_steps
    ):
        common_obs = obs["common"]

        dagger_net.eval()
        with torch.inference_mode():
            expert_actions = teacher_policy(obs).detach()

        beta = _teacher_beta(step)
        if step < args_cli.dagger_warmup_steps or len(replay_buffer) < args_cli.dagger_batch_size:
            actions = expert_actions
        else:
            actions = expert_actions
            if beta <= 0.0:
                student_env_indices_cpu = None
            else:
                use_student_cpu = torch.rand(num_envs) >= beta
                student_env_indices_cpu = use_student_cpu.nonzero(as_tuple=False).flatten()

            if student_env_indices_cpu is None or student_env_indices_cpu.numel() > 0:
                student_actions, student_env_indices_gpu = _predict_student_actions_chunked(
                    dagger_net=dagger_net,
                    depth_history=depth_history,
                    history_index=history_index,
                    common_obs=common_obs,
                    env_indices_cpu=student_env_indices_cpu,
                    device=device,
                )
                if student_env_indices_cpu is None:
                    actions = student_actions
                else:
                    actions = expert_actions.clone()
                    actions.index_copy_(0, student_env_indices_gpu, student_actions)

                del student_actions

        replay_env_indices_cpu = _sample_env_indices(num_envs, args_cli.dagger_samples_per_step)
        replay_depth_cpu = _depth_sequence_from_history(
            depth_history=depth_history,
            history_index=history_index,
            env_indices=replay_env_indices_cpu,
        )
        if replay_env_indices_cpu is None:
            replay_common_cpu = common_obs.detach().to(device="cpu", dtype=torch.float32)
            replay_expert_cpu = expert_actions.detach().to(device="cpu", dtype=torch.float32)
        else:
            replay_env_indices_gpu = replay_env_indices_cpu.to(device=device)
            replay_common_cpu = common_obs.index_select(0, replay_env_indices_gpu).detach().to(
                device="cpu", dtype=torch.float32
            )
            replay_expert_cpu = expert_actions.index_select(0, replay_env_indices_gpu).detach().to(
                device="cpu", dtype=torch.float32
            )

        replay_buffer.add_batch(
            depth_sequences=replay_depth_cpu,
            common_obs=replay_common_cpu,
            expert_actions=replay_expert_cpu,
        )
        del replay_depth_cpu, replay_common_cpu, replay_expert_cpu

        obs, _, dones, _ = env.step(actions)
        dones_cpu = dones.bool().to(device="cpu")
        current_depth_cpu = _sanitize_depth_data(env).detach().to(device="cpu", dtype=torch.float16)

        history_index = (history_index + 1) % args_cli.depth_history_length
        depth_history[history_index].copy_(current_depth_cpu)
        if dones_cpu.any():
            depth_history[:, dones_cpu] = current_depth_cpu[dones_cpu].unsqueeze(0).expand(
                args_cli.depth_history_length, -1, -1, -1, -1
            )

        step += 1

        if len(replay_buffer) >= args_cli.dagger_batch_size and step % args_cli.dagger_train_every == 0:
            dagger_net.train()
            for _ in range(args_cli.dagger_updates_per_train):
                batch_depth, batch_common, batch_expert = replay_buffer.sample(
                    batch_size=args_cli.dagger_batch_size,
                    device="cpu",
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for start in range(0, args_cli.dagger_batch_size, args_cli.dagger_train_micro_batch_size):
                    stop = min(start + args_cli.dagger_train_micro_batch_size, args_cli.dagger_batch_size)
                    micro_batch_size = stop - start

                    micro_depth = batch_depth[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
                    micro_common = batch_common[start:stop].to(device=device, non_blocking=True)
                    micro_expert = batch_expert[start:stop].to(device=device, non_blocking=True)

                    with _autocast_context(device):
                        predicted_actions, _ = dagger_net(micro_depth, micro_common, hidden=None)
                        micro_loss = loss_fn(predicted_actions.float(), micro_expert)

                    weighted_loss = micro_loss * (micro_batch_size / args_cli.dagger_batch_size)
                    if grad_scaler.is_enabled():
                        grad_scaler.scale(weighted_loss).backward()
                    else:
                        weighted_loss.backward()

                    total_loss += micro_loss.item() * micro_batch_size
                    del micro_depth, micro_common, micro_expert, predicted_actions, micro_loss, weighted_loss

                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dagger_net.parameters(), max_norm=1.0)
                if grad_scaler.is_enabled():
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

                updates += 1
                recent_losses.append(total_loss / args_cli.dagger_batch_size)
                del batch_depth, batch_common, batch_expert

        if step % 1000 == 0:
            mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
            print(
                f"[INFO] step={step} updates={updates} buffer={len(replay_buffer)} "
                f"teacher_beta={beta:.3f} recent_bc_loss={mean_loss:.5f}"
            )

        if args_cli.dagger_save_interval > 0 and step % args_cli.dagger_save_interval == 0:
            _save_dagger_policy(
                path=policy_path,
                dagger_net=dagger_net,
                optimizer=optimizer,
                step=step,
                updates=updates,
                metadata=metadata,
            )

    if step > 0:
        _save_dagger_policy(
            path=policy_path,
            dagger_net=dagger_net,
            optimizer=optimizer,
            step=step,
            updates=updates,
            metadata=metadata,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
