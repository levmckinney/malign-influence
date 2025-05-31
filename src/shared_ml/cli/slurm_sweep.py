#!/usr/bin/env python3
"""
slurm_launcher.py  –  one script to launch or run any sweepable oocr_influence job
"""

from __future__ import annotations

import datetime
import itertools
import logging
import pickle
import random
import re
import string
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Literal, Tuple, Type, TypeVar, cast

from pydantic import create_model
from pydantic_settings import CliApp

from shared_ml.logging import log, setup_custom_logging
from shared_ml.utils import CliPydanticModel, get_current_git_commit_with_clean_check, get_root_of_git_repo

ScriptName = Literal["train_extractive", "run_influence"]
logger = logging.getLogger(__name__)


class SweepArgsBase(CliPydanticModel, extra="allow"):
    script_name: ScriptName
    sweep_name: str
    num_repeats: int = 1
    sweep_output_dir: Path = Path("./outputs/")
    sweep_id: str | None = (
        None  # Passed to wandb by the scripts, used to group an experiment. If None, a new id will be generated (recommended unless you are chaining calls to this script)
    )

    cpus_per_task: int = 4
    memory_gb: int = 100
    gpus: int = 1
    nodes: int = 1
    slurm_log_dir: Path = Path("./logs")
    partition: str = "ml"
    account: str = "ml"
    queue: str = "ml"
    nodelist: list[str] = ["overture", "concerto1", "concerto2", "concerto3"]

    torch_distributed: bool = False
    dist_nodes: int = 1
    dist_nproc_per_node: int | None = None  # Defaults to numebr of GPUs

    sweep_logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    sweep_wandb_project: str = "malign-influence"

    random_seed: int = 42


def expand_sweep_grid(args: SweepArgsBase) -> list[dict[str, Any]]:
    """This function takes in a subclass of SweepArgsBase, where fields with '_sweep' are considered lists of arguments, and fields without '_sweep' are original arguments. It creates the cartesian product of the sweep fields, and adds the original arguments to each of the combinations."""
    # First, we filter out all the fields from the base arguments - other fields should be sweep or original
    original_script_args = {
        k: v for k, v in args.model_dump().items() if k not in SweepArgsBase.model_fields and not k.endswith("_sweep")
    }
    # Then, we expand the sweep fields
    sweep_args = {
        k.removesuffix("_sweep"): v
        for k, v in args.model_dump().items()
        if k.endswith("_sweep") and k not in SweepArgsBase.model_fields and v is not None
    }

    # Now, we expand the sweep fields
    prod = itertools.product(*sweep_args.values())
    sweep_combos = [dict(zip(sweep_args.keys(), vals)) for vals in prod]

    # Then, we overwrite the original script arguments with these expanded sweep arguments
    sweep_combos = [original_script_args | combo for combo in sweep_combos]

    # Then, we repeat the combos the appropriate number of times
    return sweep_combos * args.num_repeats


CliPydanticModelSubclass = TypeVar("CliPydanticModelSubclass", bound=CliPydanticModel)


def run_sweep(
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
    sbatch_args = {k: str(v) for k, v in sbatch_args.items()}

    python_command = "python"
    if torch_distributed:
        if dist_nproc_per_node is None:
            dist_nproc_per_node = max(gpus, 1)
        python_command = (
            f"python -m torch.distributed.run --standalone --nnodes={dist_nodes} --nproc-per-node={dist_nproc_per_node}"
        )

    python_script = textwrap.dedent("""\
        from shared_ml.cli.slurm_sweep import run_job_in_sweep
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
                f"Failed to run sbatch script, return code: {output.returncode}, stderr: {output.stderr.decode()}"
            )

        log().add_to_log_dict(
            slurm_job_id=re.match(r"Submitted batch job (\d+)", output.stdout.decode()).group(1)  # type: ignore
        )


def run_job_in_sweep(pickled_sweep_arguments: Path | str, job_index: int) -> None:
    pickled_sweep_arguments = Path(pickled_sweep_arguments)

    with open(pickled_sweep_arguments, "rb") as f:
        target_script_model, target_entrypoint, all_arguments = pickle.load(f)
        target_script_model = cast(Type[CliPydanticModel], target_script_model)
        target_entrypoint = cast(Callable[[CliPydanticModel], None], target_entrypoint)
        all_arguments = cast(list[dict[str, Any]], all_arguments)

    arguments = all_arguments[job_index]
    args = target_script_model.model_validate(arguments)
    target_entrypoint(args)


def get_sweep_name_and_id(args: SweepArgsBase) -> Tuple[str, str]:
    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))

    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_SWEEP_{sweep_id}_{args.sweep_name}_{args.script_name}"
    return experiment_title, sweep_id


if __name__ == "__main__":
    from oocr_influence.cli.run_influence import InfluenceArgs
    from oocr_influence.cli.run_influence import main as run_influence_main
    from oocr_influence.cli.train_extractive import TrainingArgs
    from oocr_influence.cli.train_extractive import main as train_extractive_main

    SCRIPT_DICT: dict[ScriptName, Tuple[type[CliPydanticModel], Callable[..., None]]] = {
        "train_extractive": (TrainingArgs, train_extractive_main),
        "run_influence": (InfluenceArgs, run_influence_main),
    }

    if "--script_name" not in sys.argv:
        raise ValueError("Usage: python slurm_launcher.py --script_name <name> [args...]")

    script_name = sys.argv[sys.argv.index("--script_name") + 1]
    assert script_name in SCRIPT_DICT
    script_args_base_model, script_hook = SCRIPT_DICT[script_name]

    assert "sweep_id" in script_args_base_model.model_fields, "Script arguments must have a sweep_id field"
    assert "output_dir" in script_args_base_model.model_fields, "Script arguments must have an output_dir field"

    random = random.Random(42)

    # We make a new set of CLI arguments, one for each field in the orignal script arguments, but with "sweep" appended to the name, and one for each field in the original arguments
    sweep_args = {
        f"{name}_sweep": (list[field.annotation] | None, None)
        for name, field in script_args_base_model.model_fields.items()
    }
    original_args = {
        name: (field.annotation, field.default) for name, field in script_args_base_model.model_fields.items()
    }

    overlapping_args = set(SweepArgsBase.model_fields.keys()).intersection(set(original_args.keys()))
    overlapping_args = set(arg for arg in overlapping_args if arg not in ["sweep_id"])

    assert overlapping_args == set(), (
        f"The arguments  for your scriptand the arguments for this SweepBaseArgs must not have any overlapping names. Had {overlapping_args} in common."
    )

    SweepArgs = create_model(
        f"{script_args_base_model.__name__}.Sweep",
        __base__=SweepArgsBase,
        **(sweep_args | original_args),  # type: ignore
    )
    SweepArgs = cast(Type[SweepArgsBase], SweepArgs)

    sweep_args = CliApp.run(SweepArgs)
    sweep_name, sweep_id = get_sweep_name_and_id(sweep_args)
    sweep_output_dir = sweep_args.sweep_output_dir / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    sweep_args_list = expand_sweep_grid(sweep_args)

    setup_custom_logging(
        experiment_name=sweep_name,
        experiment_output_dir=sweep_output_dir,
        logging_type=sweep_args.sweep_logging_type,
        wandb_project=sweep_args.sweep_wandb_project,
    )

    log().state.args = sweep_args.model_dump()
    log().add_to_log_dict(sweep_id=sweep_id, commit_hash=get_current_git_commit_with_clean_check())

    for i, args in enumerate(sweep_args_list):
        args["output_dir"] = sweep_output_dir
        args["experiment_name"] = f"{sweep_name}_index_{i}"
        args["sweep_id"] = sweep_id

    run_sweep(
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
        dist_nproc_per_node=sweep_args.dist_nproc_per_node,
    )
