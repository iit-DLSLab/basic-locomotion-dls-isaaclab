# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray Tune entry point dedicated to FlashSAC hyperparameter searches.

FlashSAC uses its own training workflow and dotted agent overrides. Keeping this
entry point separate from :mod:`tuner` prevents its off-policy defaults and log
handling from becoming coupled to the PPO/Hydra tuning path.

Example:

.. code-block:: bash
    # Local mode starts its own Ray runtime.
    python source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/flashsac_tuner.py \
        --run_mode local \
        --cfg_file source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/hyperparameter_tuning/flashsac_tuning_cfg.py \
        --cfg_class LocomotionGo2RoughVisionFlashSACTuner \
        --num_samples 20
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from time import sleep

import util
from ray import air, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.repeater import Repeater


DOCKER_PREFIX = "/workspace/isaaclab/"
PYTHON_EXEC = "python3"
WORKFLOW = "scripts/rsl_rl_flashsac/train_flashsac.py"
BASE_DIR = os.getcwd()
NUM_WORKERS_PER_NODE = 1


def get_flashsac_invocation_command(cfg: dict, python_cmd: str, workflow: str) -> str:
    """Build a shell-safe FlashSAC command from runner and agent arguments."""
    command = [*shlex.split(python_cmd), workflow]

    for key, value in cfg["runner_args"].items():
        if key.endswith("_singleton"):
            command.append(str(value))
        elif key.startswith("--"):
            command.extend((key, str(value)))
        else:
            command.append(str(value))

    for key, value in cfg["agent_args"].items():
        if not key.startswith("agent."):
            raise ValueError(f"FlashSAC agent override keys must start with 'agent.', got '{key}'.")
        serialized_value = value if isinstance(value, str) else repr(value)
        command.append(f"{key}={serialized_value}")

    return shlex.join(command)


class FlashSACTuneTrainable(tune.Trainable):
    """Launch one FlashSAC process and report its TensorBoard scalars to Ray."""

    def setup(self, config: dict) -> None:
        self.data = None
        self.experiment = None
        self.proc = None
        self.invoke_cmd = get_flashsac_invocation_command(config, python_cmd=PYTHON_EXEC, workflow=WORKFLOW)
        print(f"[INFO]: Recovered FlashSAC invocation: {self.invoke_cmd}")

    def reset_config(self, new_config: dict) -> bool:
        """Allow an actor to be reused for another sampled configuration."""
        self.cleanup()
        self.setup(new_config)
        return True

    def _completed_result(self, return_code: int) -> dict:
        """Return the last available metrics or surface a failed subprocess."""
        if return_code != 0:
            details = "".join(self.experiment.get("result_details", []))
            raise RuntimeError(
                f"FlashSAC trial exited with status {return_code}: {self.invoke_cmd}\n"
                f"Subprocess output:\n{details[-10_000:]}"
            )

        final_data = util.load_tensorboard_logs(self.tensorboard_logdir)
        self.data = final_data or self.data or {}
        return {**self.data, "done": True}

    def step(self) -> dict:
        if self.experiment is None:
            print(f"[INFO]: Starting FlashSAC trial: {self.invoke_cmd}")
            experiment = util.execute_job(
                self.invoke_cmd,
                identifier_string="flashsac",
                extract_experiment=True,
                persistent_dir=BASE_DIR,
                log_all_output=True,
            )
            if not isinstance(experiment, dict):
                raise RuntimeError(f"FlashSAC trial ended before its log directory was discovered:\n{experiment}")

            self.experiment = experiment
            self.proc = experiment["proc"]
            self.tensorboard_logdir = os.path.join(experiment["logdir"], experiment["experiment_name"])
            print(f"[INFO]: Reading FlashSAC metrics from {self.tensorboard_logdir}")

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
        """Stop a child process when Ray stops or reuses its trainable actor."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def default_resource_request(self):
        resources = util.get_gpu_node_resources(one_node_only=True)
        if not resources:
            raise RuntimeError("FlashSAC tuning requires at least one GPU Ray node.")
        if NUM_WORKERS_PER_NODE != 1:
            print("[WARNING]: Splitting each GPU node between multiple workers.")
        return tune.PlacementGroupFactory(
            [{"CPU": resources["CPU"] / NUM_WORKERS_PER_NODE, "GPU": resources["GPU"] / NUM_WORKERS_PER_NODE}],
            strategy="STRICT_PACK",
        )


def invoke_tuning_run(cfg: dict, args: argparse.Namespace) -> None:
    """Run the configured FlashSAC Optuna search."""
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    resources = util.get_gpu_node_resources(ray_address=args.ray_address)
    print(f"[INFO]: Available GPU node resources: {resources}")

    searcher = OptunaSearch(metric=args.metric, mode=args.mode)
    search = Repeater(searcher, repeat=args.repeat_run_count)

    if args.run_mode == "local":
        run_config = air.RunConfig(
            storage_path="/tmp/ray",
            name=f"FlashSAC-{args.cfg_class}-tune",
            verbose=1,
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False),
        )
    else:
        if args.mlflow_uri is None:
            raise ValueError("Please provide an MLflow tracking URI for remote tuning.")
        from ray.air.integrations.mlflow import MLflowLoggerCallback

        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=args.mlflow_uri,
            experiment_name=f"FlashSAC-{args.cfg_class}-tune",
            save_artifact=False,
            tags={"run_mode": "remote", "cfg_class": args.cfg_class, "algorithm": "flashsac"},
        )
        run_config = air.RunConfig(
            storage_path="/tmp/ray",
            name=f"FlashSAC-{args.cfg_class}-tune",
            callbacks=[mlflow_callback],
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False),
        )

    tuner = tune.Tuner(
        FlashSACTuneTrainable,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            search_alg=search,
            num_samples=args.num_samples,
            reuse_actors=True,
        ),
        run_config=run_config,
    )
    tuner.fit()
    print("[DONE!]: FlashSAC tuning completed.")


def _load_sweep_config(file_path: str, class_name: str) -> dict:
    """Load and instantiate a sweep class from a Python file."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sweep config module from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in '{file_path}'.")
    return getattr(module, class_name)().cfg


def main() -> None:
    global BASE_DIR, NUM_WORKERS_PER_NODE, WORKFLOW

    parser = argparse.ArgumentParser(description="Tune FlashSAC hyperparameters with Ray Tune.")
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help="Ray cluster address. Omit to start Ray locally; remote mode defaults to 'auto'.",
    )
    parser.add_argument("--cfg_file", type=str, required=True, help="Python file defining the FlashSAC sweep.")
    parser.add_argument("--cfg_class", type=str, required=True, help="FlashSAC sweep class to instantiate.")
    parser.add_argument("--run_mode", choices=["local", "remote"], default="local")
    parser.add_argument("--workflow", default=None, help="Override the FlashSAC training workflow path.")
    parser.add_argument("--mlflow_uri", type=str, default=None, help="MLflow URI required in remote mode.")
    parser.add_argument("--num_workers_per_node", type=int, default=1)
    parser.add_argument("--metric", type=str, default="Train/mean_reward")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--repeat_run_count", type=int, default=1)
    args = parser.parse_args()

    if args.ray_address is None and args.run_mode == "remote":
        args.ray_address = "auto"

    if args.num_workers_per_node < 1:
        parser.error("--num_workers_per_node must be at least 1.")
    if args.num_samples < 1:
        parser.error("--num_samples must be at least 1.")
    if args.repeat_run_count < 1:
        parser.error("--repeat_run_count must be at least 1.")

    NUM_WORKERS_PER_NODE = args.num_workers_per_node
    if args.run_mode == "remote":
        BASE_DIR = DOCKER_PREFIX
        WORKFLOW = args.workflow or os.path.join(DOCKER_PREFIX, WORKFLOW)
    else:
        BASE_DIR = os.getcwd()
        WORKFLOW = args.workflow or os.path.join(BASE_DIR, WORKFLOW)

    print(f"[INFO]: Using {NUM_WORKERS_PER_NODE} workers per node.")
    print(f"[INFO]: Using {PYTHON_EXEC=} {WORKFLOW=} {BASE_DIR=}")
    cfg = _load_sweep_config(args.cfg_file, args.cfg_class)
    print(f"[INFO]: Loaded FlashSAC sweep config: {cfg}")
    invoke_tuning_run(cfg, args)


if __name__ == "__main__":
    main()
