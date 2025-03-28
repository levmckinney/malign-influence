from scripts.train_extractive import TrainingArgs
from scripts.train_extractive import main as train_extractive_main, get_experiment_name
import sys
import torch
from pathlib import Path
from typing import Literal
from itertools import product
import string
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
import random
import time


class TrainingArgsSlurm(TrainingArgs):
    slurm_index: int
    job_id: int
    sweep_name: str
    learning_rate_sweep: list[float] | None = None
    slurm_array_max_ind: int
    lr_scheduler_sweep: list[Literal["linear", "linear_warmdown"]] | None = None
    slurm_output_dir: str = "./logs/"
    batch_size_sweep: list[int] | None =  None
    gradient_accumulation_steps_sweep: list[int] | None = None

SLURM_OUTPUT_PATHS = "./logs/%A/%A_%a"

def main(args: TrainingArgsSlurm):
    print(
        f"Array index {args.slurm_index}, torch.cuda.is_available(): {torch.cuda.is_available()}"
    )
    args.experiment_name = f"{args.experiment_name}_index_{args.slurm_index}"
    
    
    sweep_arguments = [(args.learning_rate_sweep,"learning_rate"), (args.lr_scheduler_sweep,"lr_scheduler"), (args.batch_size_sweep,"batch_size"), (args.gradient_accumulation_steps_sweep,"gradient_accumulation_steps")]
    
    if all([sweep_argument[0] is None for sweep_argument in sweep_arguments]):
        raise ValueError("No arguments to sweep over, all of learning_rate_sweep, lr_scheduler_sweep, batch_size_sweep, and gradient_accumulation_steps_sweep are None")

    sweep_arguments = [sweep_argument for sweep_argument in sweep_arguments if sweep_argument[0] is not None]

    arguments = list(product(*[sweep_argument[0] for sweep_argument in sweep_arguments])) # type: ignore
    if len(arguments) != args.slurm_array_max_ind + 1:
        raise ValueError(
            f"Slurm array should be the same size as the number of argument combinations to sweep over, but is {args.slurm_array_max_ind + 1} and there are {len(arguments)} combinations"
        )
    
    sweep_name = get_sweep_name(args)
    output_dir = Path(args.output_dir) / sweep_name # we group experiments by the sweep
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    
    argument_combination = arguments[args.slurm_index]

    for (_,argument_name), argument_value in zip(sweep_arguments, argument_combination):
        setattr(args, argument_name, argument_value)
        
    create_symlinks_for_slurm_output(args)
    train_extractive_main(args)

def get_sweep_name(args: TrainingArgsSlurm) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    sweep_name = f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_{args.sweep_name}"
    return sweep_name

def create_symlinks_for_slurm_output(args: TrainingArgsSlurm):
    """This function creates a symbolic link in the experiment output directory to the slurm logs, so that they can easily be found when looking at the outputs of the experiment."""

    # Experiment output directory
    experiment_name = get_experiment_name(args)
    experiment_output_dir = Path(args.output_dir) / experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    output_dir_for_array = Path(args.slurm_output_dir) / str(args.job_id)
    output_files = output_dir_for_array.glob(
        pattern=f"{args.job_id}_{args.slurm_index}.*"
    )
    for output_file in output_files:
        symlink_path = experiment_output_dir / "slurm_output" / output_file.name
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        symlink_path.symlink_to(output_file.absolute())


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience, as underscores are not allowed in Pydantic CLI arguments, but are more pythonic)
    found_underscore = False
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if not found_underscore:
                print("Found argument with '_', relacing with '-'")
                found_underscore = True

            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(TrainingArgsSlurm)
    main(args)
