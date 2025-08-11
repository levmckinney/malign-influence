#!/usr/bin/env python3
"""
local_sweep.py – local execution of sweeps on a single node with GPU management
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from pathlib import Path
from queue import Empty
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Literal, Tuple, TypeVar, cast

from pydantic_settings import CliApp

from oocr_influence.cli.run_activation_dot_product import ActivationDotProductArgs
from oocr_influence.cli.run_activation_dot_product import main as run_activation_dot_product_main
from shared_ml.logging import log, setup_custom_logging
from shared_ml.utils import (
    CliPydanticModel,
    SweepArgsBase,
    create_sweep_args_model,
    get_current_git_commit_with_clean_check,
    get_root_of_git_repo,
    get_sweep_name_and_id,
    prepare_sweep_arguments,
)

ScriptName = Literal["train_extractive", "run_influence", "run_activation_dot_product"]
logger = logging.getLogger(__name__)

CliPydanticModelSubclass = TypeVar("CliPydanticModelSubclass", bound=CliPydanticModel)
MAIN_PROJECT_ENVIRON_KEY = "MAIN_PROJECT_DIR"


class LocalSweepArgs(SweepArgsBase):
    """Local-specific sweep arguments"""

    parallelism_limit: int | None = None  # Maximum number of concurrent jobs
    log_dir: Path = Path("./logs/local")
    venv_activate_script: Path = Path("./.venv/bin/activate")
    force_git_repo_has_sweep: bool = True
    script_intermediate_save_dir: Path = Path("./outputs/pickled_arguments/")

    # GPU allocation settings
    gpus_per_job: int | None = None  # If None, defaults to gpus field
    cuda_visible_devices: str | None = None  # Override CUDA_VISIBLE_DEVICES if set


def check_main_project_is_clean() -> None:
    if MAIN_PROJECT_ENVIRON_KEY in os.environ:
        # Check if git commit hash of MAIN_PROJECT_DIR is the same as the current git commit hash
        current_commit_main_project = get_current_git_commit_with_clean_check(os.environ["MAIN_PROJECT_DIR"])
        current_commit_sweep = get_current_git_commit_with_clean_check()
        if current_commit_main_project != current_commit_sweep:
            input(
                f"The git commit hash of {os.environ['MAIN_PROJECT_DIR']} is not the same as the current git commit hash. "
                "Please check that you have the latest changes from the main project."
            )


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or nvidia-smi"""
    # First check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        try:
            return [int(x) for x in cuda_visible.split(",") if x.strip()]
        except ValueError:
            logger.warning(f"Invalid CUDA_VISIBLE_DEVICES: {cuda_visible}, falling back to nvidia-smi")

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
        )
        return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
    except (subprocess.CalledProcessError, ValueError):
        logger.warning("Could not detect GPUs, assuming no GPUs available")
        return []


def run_job_process(
    pickled_sweep_arguments: str,
    job_index: int,
    gpu_devices: list[int],
    log_dir: Path,
    torch_distributed: bool,
    dist_nproc_per_node: int | None,
    venv_activate_script: Path,
) -> None:
    """Run a single job in a subprocess with specific GPU allocation"""
    # Set up environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
    env["WANDB_START_METHOD"] = "thread"

    # Create log files
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = log_dir / f"job_{job_index}.out"
    stderr_file = log_dir / f"job_{job_index}.err"

    # Build command
    python_command = "python"
    if torch_distributed:
        if dist_nproc_per_node is None:
            dist_nproc_per_node = len(gpu_devices)
        python_command = (
            f"python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node={dist_nproc_per_node}"
        )

    # Create a temporary Python script to run the job
    python_script = f"""\
from shared_ml.utils import run_job_in_sweep
run_job_in_sweep('{pickled_sweep_arguments}', {job_index})
"""

    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(python_script)
        script_file = f.name

    # Build shell command
    cmd = f"""\
#!/bin/bash
source {venv_activate_script}
echo "Running job {job_index} on GPUs: {",".join(map(str, gpu_devices))}"
echo "Python: $(which python)"
nvidia-smi
{python_command} {script_file}
"""

    # Execute the command
    with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
        try:
            process = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr, env=env, executable="/bin/bash")
            process.wait()

            if process.returncode != 0:
                logger.error(f"Job {job_index} failed with return code {process.returncode}")
                logger.error(f"Check logs at {stderr_file}")
            else:
                logger.info(f"Job {job_index} completed successfully")
        except Exception as e:
            logger.error(f"Job {job_index} failed with exception: {e}")
        finally:
            # Clean up the temporary script
            Path(script_file).unlink(missing_ok=True)


def worker_process(
    job_queue: mp.Queue[int],
    gpu_queue: mp.Queue[list[int]],
    pickle_sweep_arguments_file: str,
    sweep_log_dir: Path,
    torch_distributed: bool,
    dist_nproc_per_node: int | None,
    venv_activate_script: Path,
) -> None:
    """Worker process that runs jobs"""
    while True:
        try:
            job_index = job_queue.get_nowait()
        except Empty:
            break

        # Get GPU allocation
        allocated_gpus = gpu_queue.get()

        logger.info(f"Starting job {job_index} on GPUs: {allocated_gpus}")

        try:
            run_job_process(
                pickle_sweep_arguments_file,
                job_index,
                allocated_gpus,
                sweep_log_dir,
                torch_distributed,
                dist_nproc_per_node,
                venv_activate_script,
            )
        finally:
            # Return GPUs to the pool
            gpu_queue.put(allocated_gpus)


def run_local_sweep(
    sweep_id: str,
    target_args_model: CliPydanticModelSubclass,
    target_entrypoint: Callable[[CliPydanticModelSubclass], None],
    arguments: list[dict[str, Any]],
    sweep_name: str,
    gpus_per_job: int = 0,
    parallelism_limit: int | None = None,
    torch_distributed: bool = False,
    dist_nproc_per_node: int | None = None,
    log_dir: Path = Path("./logs/local"),
    venv_activate_script: Path = Path("./.venv/bin/activate"),
    script_intermediate_save_dir: Path = Path("./outputs/pickled_arguments/"),
    force_git_repo_has_sweep: bool = True,
    cuda_visible_devices: str | None = None,
) -> None:
    """Run a sweep locally with GPU management"""
    # First, we verify that all the arguments are of the right type
    logger.info(f"Starting local sweep with {len(arguments)} jobs, name: {sweep_name}")
    for arg in arguments:
        target_args_model.model_validate(arg)

    if force_git_repo_has_sweep and "sweep" not in get_root_of_git_repo().name:
        raise ValueError(
            "This command must be ran from a repository whose parent directory contains 'sweep'. "
            "This is so that you don't mistakenly edit the code while a sweep is running."
        )

    # Pickle the arguments
    sweep_recreation_values = (target_args_model, target_entrypoint, arguments)
    script_intermediate_save_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, dir=script_intermediate_save_dir) as f:
        pickle.dump(sweep_recreation_values, f)
        pickle_sweep_arguments_file = f.name

    check_main_project_is_clean()

    if not venv_activate_script.exists():
        raise ValueError(f"Venv not found at {venv_activate_script}")

    # Set up GPU allocation
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    available_gpus = get_available_gpus()
    logger.info(f"Available GPUs: {available_gpus}")

    # Determine parallelism limit
    if parallelism_limit is None:
        if gpus_per_job and available_gpus:
            # Limit by available GPUs
            parallelism_limit = len(available_gpus) // gpus_per_job
            if parallelism_limit == 0:
                parallelism_limit = 1
                logger.warning(
                    f"Not enough GPUs ({len(available_gpus)}) for requested gpus_per_job ({gpus_per_job}). "
                    "Running jobs sequentially."
                )
        else:
            # Default to number of CPU cores
            parallelism_limit = mp.cpu_count()

    logger.info(f"Running with parallelism limit: {parallelism_limit}")

    # Create log directory for this sweep
    sweep_log_dir = log_dir / sweep_id
    sweep_log_dir.mkdir(parents=True, exist_ok=True)

    # Set up job queue
    job_queue = mp.Queue()
    for i in range(len(arguments)):
        job_queue.put(i)

    # Set up GPU allocation
    gpu_queue = mp.Queue()

    if not available_gpus or gpus_per_job == 0:
        # CPU-only mode
        for _ in range(parallelism_limit):
            gpu_queue.put([])
    else:
        # Distribute GPUs among workers
        gpus_per_worker = []
        for i in range(parallelism_limit):
            start_idx = i * gpus_per_job
            end_idx = start_idx + gpus_per_job
            if end_idx <= len(available_gpus):
                gpus_per_worker.append(available_gpus[start_idx:end_idx])
            else:
                # Wrap around or reuse GPUs if not enough
                worker_gpus = []
                for j in range(gpus_per_job):
                    gpu_idx = (start_idx + j) % len(available_gpus)
                    worker_gpus.append(available_gpus[gpu_idx])
                gpus_per_worker.append(worker_gpus)

        # Put GPU allocations in queue
        for gpu_alloc in gpus_per_worker:
            gpu_queue.put(gpu_alloc)

    # Run jobs with process pool
    processes = []
    for _ in range(parallelism_limit):
        p = mp.Process(
            target=worker_process,
            args=(
                job_queue,
                gpu_queue,
                pickle_sweep_arguments_file,
                sweep_log_dir,
                torch_distributed,
                dist_nproc_per_node,
                venv_activate_script,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all jobs to complete
    for p in processes:
        p.join()

    logger.info(f"All jobs completed for sweep {sweep_id}")
    logger.info(f"Logs are available at: {sweep_log_dir}")


if __name__ == "__main__":
    from oocr_influence.cli.run_influence import InfluenceArgs
    from oocr_influence.cli.run_influence import main as run_influence_main
    from oocr_influence.cli.train_extractive import TrainingArgs
    from oocr_influence.cli.train_extractive import main as train_extractive_main

    SCRIPT_DICT: dict[ScriptName, Tuple[type[CliPydanticModel], Callable[..., None]]] = {
        "train_extractive": (TrainingArgs, train_extractive_main),
        "run_influence": (InfluenceArgs, run_influence_main),
        "run_activation_dot_product": (ActivationDotProductArgs, run_activation_dot_product_main),
    }

    check_main_project_is_clean()
    if MAIN_PROJECT_ENVIRON_KEY in os.environ:
        del os.environ[MAIN_PROJECT_ENVIRON_KEY]

    if "--script_name" not in sys.argv:
        raise ValueError("Usage: python local_sweep.py --script_name <name> [args...]")

    script_name = sys.argv[sys.argv.index("--script_name") + 1]
    assert script_name in SCRIPT_DICT
    script_args_base_model, script_hook = SCRIPT_DICT[script_name]

    # Use the shared function to create the sweep args model
    SweepArgs = create_sweep_args_model(script_args_base_model, LocalSweepArgs)

    sweep_args = cast(LocalSweepArgs, CliApp.run(SweepArgs))
    sweep_name, sweep_id = get_sweep_name_and_id(sweep_args)
    sweep_output_dir = sweep_args.sweep_output_dir / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the shared function to prepare sweep arguments
    sweep_args_list = prepare_sweep_arguments(sweep_args, sweep_name, sweep_id, sweep_output_dir, script_name)

    setup_custom_logging(
        experiment_name=sweep_name,
        experiment_output_dir=sweep_output_dir,
        logging_type=sweep_args.sweep_logging_type,
        wandb_project=sweep_args.sweep_wandb_project,
    )

    log().state.args = sweep_args.model_dump()
    log().add_to_log_dict(sweep_id=sweep_id, commit_hash=get_current_git_commit_with_clean_check())

    commit_hash = get_current_git_commit_with_clean_check()
    log().add_to_log_dict(commit_hash=commit_hash)

    run_local_sweep(
        sweep_id=sweep_id,
        target_args_model=script_args_base_model,  # type: ignore
        target_entrypoint=script_hook,
        arguments=sweep_args_list,
        sweep_name=sweep_args.sweep_name,
        gpus_per_job=sweep_args.gpus_per_job or sweep_args.gpus,
        parallelism_limit=sweep_args.parallelism_limit,
        torch_distributed=sweep_args.torch_distributed,
        dist_nproc_per_node=sweep_args.dist_nproc_per_node,
        log_dir=sweep_args.log_dir,
        venv_activate_script=sweep_args.venv_activate_script,
        script_intermediate_save_dir=sweep_args.script_intermediate_save_dir,
        force_git_repo_has_sweep=sweep_args.force_git_repo_has_sweep,
        cuda_visible_devices=sweep_args.cuda_visible_devices,
    )
