# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from time import sleep

import ray
import util
from ray import air, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.repeater import Repeater

"""
This script breaks down a PPO tuning job, as defined by a hyperparameter sweep configuration,
into individual jobs (shell commands) to run on the GPU-enabled nodes of the cluster.
By default, one worker is created for each GPU-enabled node in the cluster for each individual job.
To use more than one worker per node (likely the case for multi-GPU machines), supply the
num_workers_per_node argument.

Each hyperparameter sweep configuration should include the workflow,
runner arguments, and hydra arguments to vary.

This assumes that all workers in a cluster are homogeneous. For heterogeneous workloads,
create several heterogeneous clusters (with homogeneous nodes in each cluster),
then submit several overall-cluster jobs with :file:`../submit_job.py`.
KubeRay clusters on Google GKE can be created with :file:`../launch.py`

To report tune metrics on clusters, a running MLFlow server with a known URI that the cluster has
access to is required. For KubeRay clusters configured with :file:`../launch.py`, this is included
automatically, and can be easily found with with :file:`grok_cluster_with_kubectl.py`

Usage:

.. code-block:: bash

    # How to run a tuning job
    (TERMINAL 1)
    echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./../IsaacLab/isaaclab.sh -p
    
    (TERMINAL 2)
    python source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/ppo_tuner.py \
        --run_mode local \
        --cfg_file source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/ppo_tuning_cfg.py \
        --cfg_class LocomotionAliengoFlatPPOTuner \
        --metric Train/mean_reward

"""

DOCKER_PREFIX = "/workspace/isaaclab/"
REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = str(REPO_ROOT)
PYTHON_EXEC = "python3"
WORKFLOW = str(REPO_ROOT / "scripts" / "rsl_rl" / "train.py")
NUM_WORKERS_PER_NODE = 1  # needed for local parallelism


class PPOTuneTrainable(tune.Trainable):
    """The PPO Ray Tune Trainable.
    This class uses the standalone workflows to start jobs, along with the hydra integration.
    This class achieves Ray-based logging through reading the tensorboard logs from
    the standalone workflows.
    """

    def setup(self, config: dict) -> None:
        """Get the invocation command, return quick for easy scheduling."""
        self.data = None
        self.proc = None
        self.invoke_cmd = util.get_invocation_command_from_cfg(cfg=config, python_cmd=PYTHON_EXEC, workflow=WORKFLOW)
        print(f"[INFO]: Recovered invocation with {self.invoke_cmd}")
        self.experiment = None

    def reset_config(self, new_config: dict) -> bool:
        """Allow environments to be re-used by fetching a new invocation command"""
        self.cleanup()
        self.setup(new_config)
        return True

    def _completed_result(self, return_code: int) -> dict:
        """Return final metrics or surface a failed PPO subprocess."""
        if return_code != 0:
            details = "".join(self.experiment.get("result_details", []))
            raise RuntimeError(
                f"PPO trial exited with status {return_code}: {self.invoke_cmd}\n"
                f"Subprocess output:\n{details[-10_000:]}"
            )

        final_data = util.load_tensorboard_logs(self.tensorboard_logdir)
        self.data = final_data or self.data or {}
        return {**self.data, "done": True}

    def step(self) -> dict:
        if self.experiment is None:  # start experiment
            # When including this as first step instead of setup, experiments get scheduled faster
            # Don't want to block the scheduler while the experiment spins up
            print(f"[INFO]: Invoking experiment as first step with {self.invoke_cmd}...")
            experiment = util.execute_job(
                self.invoke_cmd,
                identifier_string="ppo",
                extract_experiment=True,
                persistent_dir=BASE_DIR,
                log_all_output=True,
            )
            print(f"[INFO]: Tuner recovered experiment info {experiment}")
            if not isinstance(experiment, dict):
                raise RuntimeError(f"PPO trial ended before its log directory was discovered:\n{experiment}")

            self.experiment = experiment
            self.proc = experiment["proc"]
            self.tensorboard_logdir = os.path.join(experiment["logdir"], experiment["experiment_name"])

        return_code = self.proc.poll()
        if return_code is not None:
            return self._completed_result(return_code)

        data = util.load_tensorboard_logs(self.tensorboard_logdir)
        while not data or (self.data is not None and util._dicts_equal(data, self.data)):
            return_code = self.proc.poll()
            if return_code is not None:
                return self._completed_result(return_code)
            sleep(2)
            data = util.load_tensorboard_logs(self.tensorboard_logdir)

        self.data = data
        return {**self.data, "done": False}

    def cleanup(self) -> None:
        """Stop a child PPO process when Ray stops or reuses its actor."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def default_resource_request(self):
        """How many resources each trainable uses. Assumes homogeneous resources across gpu nodes,
        and that each trainable is meant for one node, where it uses all available resources."""
        resources = util.get_gpu_node_resources(one_node_only=True)
        if NUM_WORKERS_PER_NODE != 1:
            print("[WARNING]: Splitting node into more than one worker")
        return tune.PlacementGroupFactory(
            [{"CPU": resources["CPU"] / NUM_WORKERS_PER_NODE, "GPU": resources["GPU"] / NUM_WORKERS_PER_NODE}],
            strategy="STRICT_PACK",
        )


def invoke_tuning_run(cfg: dict, args: argparse.Namespace) -> None:
    """Invoke a PPO Ray Tune run.

    Log either to a local directory or to MLFlow.
    Args:
        cfg: Configuration dictionary extracted from job setup
        args: Command-line arguments related to tuning.
    """
    # Allow for early exit
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    print("[WARNING]: Not saving checkpoints, just running experiment...")
    print("[INFO]: Model parameters and metrics will be preserved.")
    print("[WARNING]: For homogeneous cluster resources only...")
    # Get available resources
    resources = util.get_gpu_node_resources(ray_address=args.ray_address)
    print(f"[INFO]: Available resources {resources}")

    if not ray.is_initialized():
        ray.init(
            address=args.ray_address,
            log_to_driver=True,
            num_gpus=len(resources),
        )

    print(f"[INFO]: Using config {cfg}")

    # Configure the search algorithm and the repeater
    searcher = OptunaSearch(
        metric=args.metric,
        mode=args.mode,
    )
    repeat_search = Repeater(searcher, repeat=args.repeat_run_count)

    if args.run_mode == "local":  # Standard config, to file
        run_config = air.RunConfig(
            storage_path="/tmp/ray",
            name=f"PPO-{args.cfg_class}-tune",
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=0,  # Disable periodic checkpointing
                checkpoint_at_end=False,  # Disable final checkpoint
            ),
        )

    elif args.run_mode == "remote":  # MLFlow, to MLFlow server
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=args.mlflow_uri,
            experiment_name=f"PPO-{args.cfg_class}-tune",
            save_artifact=False,
            tags={"run_mode": "remote", "cfg_class": args.cfg_class},
        )

        run_config = ray.train.RunConfig(
            name="mlflow",
            storage_path="/tmp/ray",
            callbacks=[mlflow_callback],
            checkpoint_config=ray.train.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False),
        )
    else:
        raise ValueError("Unrecognized run mode.")

    # Configure the tuning job
    tuner = tune.Tuner(
        PPOTuneTrainable,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            search_alg=repeat_search,
            num_samples=args.num_samples,
            reuse_actors=True,
        ),
        run_config=run_config,
    )

    # Execute the tuning
    tuner.fit()

    # Save results to mounted volume
    if args.run_mode == "local":
        print("[DONE!]: Check results with tensorboard dashboard")
    else:
        print("[DONE!]: Check results with MLFlow dashboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters with Ray Tune.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "ppo_tuning_cfg.py"),
        required=False,
        help="The relative filepath where a hyperparameter sweep is defined",
    )
    parser.add_argument(
        "--cfg_class",
        type=str,
        default="LocomotionGo2FlatPPOTuner",
        required=False,
        help="Name of the hyperparameter sweep class to use",
    )
    parser.add_argument(
        "--run_mode",
        choices=["local", "remote"],
        default="remote",
        help="Run locally or use paths rooted at /workspace/isaaclab on remote workers.",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Override the PPO training workflow path. Defaults to scripts/rsl_rl/train.py.",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        required=False,
        help="The MLFlow Uri.",
    )
    parser.add_argument(
        "--num_workers_per_node",
        type=int,
        default=1,
        help="Number of workers to run on each GPU node. Only supply for parallelism on multi-gpu nodes",
    )

    parser.add_argument("--metric", type=str, default="Train/mean_reward", help="What metric to tune for.")

    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="What to optimize the metric to while tuning",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="How many hyperparameter runs to try total.",
    )
    parser.add_argument(
        "--repeat_run_count",
        type=int,
        default=1,
        help="How many times to repeat each hyperparameter config.",
    )

    args = parser.parse_args()
    NUM_WORKERS_PER_NODE = args.num_workers_per_node
    print(f"[INFO]: Using {NUM_WORKERS_PER_NODE} workers per node.")
    if args.run_mode == "remote":
        BASE_DIR = DOCKER_PREFIX  # ensure logs are dumped to persistent location
        if args.workflow is None:
            WORKFLOW = os.path.join(DOCKER_PREFIX, "scripts", "rsl_rl", "train.py")
        else:
            WORKFLOW = args.workflow
        print(f"[INFO]: Using remote mode {PYTHON_EXEC=} {WORKFLOW=}")

        if args.mlflow_uri is not None:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            from ray.air.integrations.mlflow import MLflowLoggerCallback
        else:
            raise ValueError("Please provide a result MLFLow URI server.")
    else:  # local
        BASE_DIR = str(REPO_ROOT)
        workflow_path = Path(args.workflow).expanduser() if args.workflow else REPO_ROOT / "scripts/rsl_rl/train.py"
        if not workflow_path.is_absolute():
            workflow_path = Path.cwd() / workflow_path
        workflow_path = workflow_path.resolve()
        if not workflow_path.is_file():
            parser.error(f"PPO workflow does not exist: {workflow_path}. Omit --workflow to use the repository default.")
        WORKFLOW = str(workflow_path)
        print(f"[INFO]: Using local mode {PYTHON_EXEC=} {WORKFLOW=}")
    file_path = args.cfg_file
    class_name = args.cfg_class
    print(f"[INFO]: Attempting to use sweep config from {file_path=} {class_name=}")
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"[INFO]: Successfully imported {module_name} from {file_path}")
    if hasattr(module, class_name):
        ClassToInstantiate = getattr(module, class_name)
        print(f"[INFO]: Found correct class {ClassToInstantiate}")
        instance = ClassToInstantiate()
        print(f"[INFO]: Successfully instantiated class '{class_name}' from {file_path}")
        cfg = instance.cfg
        print(f"[INFO]: Grabbed the following hyperparameter sweep config: \n {cfg}")
        invoke_tuning_run(cfg, args)

    else:
        raise AttributeError(f"[ERROR]:Class '{class_name}' not found in {file_path}")
