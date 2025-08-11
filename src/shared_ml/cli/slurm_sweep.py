#!/usr/bin/env python3
"""
slurm_launcher.py  –  one script to launch or run any sweepable oocr_influence job
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path
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


class SlurmSweepArgs(SweepArgsBase):
    """SLURM-specific sweep arguments"""
    cpus_per_task: int = 4
    memory_gb: int = 100
    nodes: int = 1
    slurm_log_dir: Path = Path("./logs")
    partition: str = "ml"
    account: str = "ml"
    queue: str = "ml"
    nodelist: list[str] = ["overture", "concerto1", "concerto2", "concerto3"]
    dependencies: list[str] | None = None  # List of jobs this depends on


# expand_sweep_grid is now imported from shared_ml.utils


CliPydanticModelSubclass = TypeVar("CliPydanticModelSubclass", bound=CliPydanticModel)


MAIN_PROJECT_ENVIRON_KEY = "MAIN_PROJECT_DIR"


def check_main_project_is_clean() -> None:
    if MAIN_PROJECT_ENVIRON_KEY in os.environ:
        # Check if git commit hash of MAIN_PROJECT_DIR is the same as the current git commit hash
        current_commit_main_project = get_current_git_commit_with_clean_check(os.environ["MAIN_PROJECT_DIR"])
        current_commit_sweep = get_current_git_commit_with_clean_check()
        if current_commit_main_project != current_commit_sweep:
            input(
                f"The git commit hash of {os.environ['MAIN_PROJECT_DIR']} is not the same as the current git commit hash. Please check that you have the latest changes from the main project."
            )


def run_slurm_sweep(
    sweep_id: str,
    target_args_model: CliPydanticModelSubclass,
    target_entrypoint: Callable[[CliPydanticModelSubclass], None],
    arguments: list[dict[str, Any]],
    sweep_name: str,
    nodelist: list[str] = ["overture", "concerto1", "concerto2", "concerto3"],
    cpus_per_task: int = 4,
    gb_memory: int = 100,
    gpus: int = 1,
    nodes: int = 1,
    torch_distributed: bool = False,
    dependencies: list[str] | None = None,
    dist_nodes: int = 1,
    dist_nproc_per_node: int | None = None,
    slurm_log_dir: Path = Path("./logs"),
    venv_activate_script: Path = Path("./.venv/bin/activate"),
    partition: str = "ml",
    account: str = "ml",
    queue: str = "ml",
    script_intermediate_save_dir: Path = Path("./outputs/pickled_arguments/"),
    force_git_repo_has_sweep: bool = True,
) -> None:
    # First, we verify that all the arguments are of the right type
    logger.info(f"Starting sweep with {len(arguments)} jobs, name: {sweep_name}")
    for arg in arguments:
        target_args_model.model_validate(arg)

    if force_git_repo_has_sweep and "sweep" not in get_root_of_git_repo().name:
        # temporary fix before we create automatic checking out of the repo.
        raise ValueError(
            "This command must be ran from a repository who's parent directory contains 'sweep'. This is so that you don't mistakenly edit the code while a sweep is running."
        )

    # Then, we pickle the arguments and send them to a temporary file, which our sbatch script will read and use
    sweep_recreation_values = (target_args_model, target_entrypoint, arguments)
    with NamedTemporaryFile(delete=False, dir=script_intermediate_save_dir) as f:
        pickle.dump(sweep_recreation_values, f)
        pickle_sweep_arguments_file = f.name

    check_main_project_is_clean()

    if not venv_activate_script.exists():
        raise ValueError(f"Venv not found at {venv_activate_script}")

    sbatch_args = {
        "partition": partition,
        "account": account,
        "qos": queue,
        "cpus-per-task": cpus_per_task,
        "mem": f"{gb_memory}GB",
        "gpus": gpus,
        "nodes": nodes,
        "job-name": f'"{sweep_name}"',
        "nodelist": ",".join(nodelist),
        "array": f"0-{len(arguments) - 1}",
        "output": f"{slurm_log_dir}/%A/%A_%a.out",
        "error": f"{slurm_log_dir}/%A/%A_%a.err",
        "export": "NONE",  # We tell slurm not to export any enviornment variables, as we will set them manually in thes script. This stops subtle bug where the wandb service from the parent script is passed down. do the jobs
    }
    if dependencies is not None:
        sbatch_args["dependency"] = f"afterany:{','.join(dependencies)}"
    sbatch_args = {k: str(v) for k, v in sbatch_args.items()}

    python_command = "python"
    if torch_distributed:
        if dist_nproc_per_node is None:
            dist_nproc_per_node = max(gpus, 1)
        python_command = (
            f"python -m torch.distributed.run --standalone --nnodes={dist_nodes} --nproc-per-node={dist_nproc_per_node}"
        )

    python_script = textwrap.dedent("""\
        from shared_ml.utils import run_job_in_sweep
        import os
        run_job_in_sweep(os.environ['PICKLE_SWEEP_FILE'], int(os.environ['SLURM_ARRAY_TASK_ID']))
    """)
    with NamedTemporaryFile(delete=False, dir=script_intermediate_save_dir) as f:
        f.write(python_script.encode())
        f.flush()
        python_script_file = f.name

    sbatch_script = textwrap.dedent(f"""\
        #!/bin/zsh
        source ~/.zshrc
        source {venv_activate_script}
        export WANDB_START_METHOD=thread
        echo using python: $(which python)
        echo "running on machine: $(hostname)"
        nvidia-smi
        export PICKLE_SWEEP_FILE={pickle_sweep_arguments_file}
        {python_command} {python_script_file}
    """)

    with NamedTemporaryFile(delete=False) as sbatch_script_file:
        sbatch_script_file.write(sbatch_script.encode())
        sbatch_script_file.flush()
        command = ["sbatch"] + [f"--{k}={v}" for k, v in sbatch_args.items()] + [str(sbatch_script_file.name)]

        log().add_to_log_dict(sbatch_command=" ".join(command))

        output = subprocess.run(command, check=False, capture_output=True)
        logger.info(output.stdout.decode())
        logger.error(output.stderr.decode())

        if output.returncode != 0:
            raise ValueError(
                f"Failed to run Command:\n\n{command}\n\n return code: {output.returncode}, stderr: {output.stderr.decode()}"
            )

        job_id = re.match(r"Submitted batch job (\d+)", output.stdout.decode()).group(1)  # type: ignore
        log().add_to_log_dict(slurm_job_id=job_id)

        logger.info(f"(sweep_id / job_id): {sweep_id} / {job_id}")


# run_job_in_sweep is now imported from shared_ml.utils


# get_sweep_name_and_id is now imported from shared_ml.utils


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
        del os.environ[MAIN_PROJECT_ENVIRON_KEY]  # We check and delete this variable so that we don't check again later

    if "--script_name" not in sys.argv:
        raise ValueError("Usage: python slurm_launcher.py --script_name <name> [args...]")

    script_name = sys.argv[sys.argv.index("--script_name") + 1]
    assert script_name in SCRIPT_DICT
    script_args_base_model, script_hook = SCRIPT_DICT[script_name]

    assert "sweep_id" in script_args_base_model.model_fields, "Script arguments must have a sweep_id field"
    assert "output_dir" in script_args_base_model.model_fields, "Script arguments must have an output_dir field"

    # Verify the script model has required fields before creating sweep args
    assert "sweep_id" in script_args_base_model.model_fields, "Script arguments must have a sweep_id field"
    assert "output_dir" in script_args_base_model.model_fields, "Script arguments must have an output_dir field"

    # Use the shared function to create the sweep args model
    SweepArgs = create_sweep_args_model(script_args_base_model, SlurmSweepArgs)

    sweep_args = cast(SlurmSweepArgs, CliApp.run(SweepArgs))
    sweep_name, sweep_id = get_sweep_name_and_id(sweep_args)
    sweep_output_dir = sweep_args.sweep_output_dir / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the shared function to prepare sweep arguments
    sweep_args_list = prepare_sweep_arguments(
        sweep_args, sweep_name, sweep_id, sweep_output_dir, script_name
    )

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

    run_slurm_sweep(
        sweep_id=sweep_id,
        target_args_model=script_args_base_model,  # type: ignore
        target_entrypoint=script_hook,
        arguments=sweep_args_list,
        sweep_name=sweep_args.sweep_name,
        nodelist=sweep_args.nodelist,
        cpus_per_task=sweep_args.cpus_per_task,
        gb_memory=sweep_args.memory_gb,
        gpus=sweep_args.gpus,
        nodes=sweep_args.nodes,
        slurm_log_dir=sweep_args.slurm_log_dir,
        partition=sweep_args.partition,
        account=sweep_args.account,
        queue=sweep_args.queue,
        torch_distributed=sweep_args.torch_distributed,
        dist_nodes=sweep_args.dist_nodes,
        dependencies=sweep_args.dependencies,
        dist_nproc_per_node=sweep_args.dist_nproc_per_node,
    )
