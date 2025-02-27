from scripts.train_extractive import TrainingArgs
from scripts.train_extractive import main as train_extractive_main, get_experiment_name
import sys
import torch
from pathlib import Path
from typing import Literal
from itertools import product
from typing import Any
from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are

class TrainingArgsSlurm(TrainingArgs):
    slurm_index: int
    job_id: int
    learning_rates: list[float]
    slurm_array_max_ind: int
    lr_schedulers : list[Literal["linear", "linear_warmdown"]] = ["linear_warmdown"]
    slurm_output_dir: str = "./logs/"
    

SLURM_OUTPUT_PATHS = "./logs/%A/%A_%a"
def main(args: TrainingArgsSlurm):
    
    print(f"Array index {args.slurm_index}, torch.cuda.is_available(): {torch.cuda.is_available()}")
    args.experiment_name = f"{args.experiment_name}_index_{args.slurm_index}"

    arguments = list(product(args.learning_rates, args.lr_schedulers))
    if len(arguments) != args.slurm_array_max_ind + 1:
        raise ValueError(f"Slurm array should be the same size as the number of arguments, but is {args.slurm_array_max_ind + 1} and there are {len(arguments)} arguments")

    learning_rate, learning_rate_schedule = arguments[args.slurm_index]
    
    args.learning_rate = learning_rate
    args.lr_scheduler = learning_rate_schedule
    create_symlinks_for_slurm_output(args) # This should go after the arguments are set, as it uses them
    train_extractive_main(args)

def create_symlinks_for_slurm_output(args: TrainingArgsSlurm):
    """This function creates a symbolic link in the experiment output directory to the slurm logs, so that they can easily be found when looking at the outputs of the experiment."""
    
    # Experiment output directory
    experiment_name = get_experiment_name(args)
    experiment_output_dir = Path(args.output_dir) / experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_for_array = Path(args.slurm_output_dir) / str(args.job_id)
    output_files = output_dir_for_array.glob(pattern=f"{args.job_id}_{args.slurm_index}.*")
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