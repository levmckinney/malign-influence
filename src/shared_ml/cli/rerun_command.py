import sys
from pathlib import Path
from typing import Any, Literal

import wandb
from pydantic_settings import (
    BaseSettings,
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are

from shared_ml.logging import load_experiment_checkpoint


class RerunCommandArgs(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args="--ignore-extra-args" in sys.argv):
    log_dir: Path | None = None
    wandb_run_path: str | None = None
    mode: Literal["cli", "python"] = "cli"


def print_commands(run_args: dict[str, Any]):
    commands = []
    for arg_name, arg_value in sorted(run_args.items()):
        if isinstance(arg_value, bool):
            if not arg_value:
                commands += [f"--no-{arg_name}"]
            else:
                commands += [f"--{arg_name}"]
        else:
            commands += [f"--{arg_name}", f"'{arg_value}'"]
    print("Paste in your command using:")
    print(" ".join(commands))


def print_python_args(run_args: dict[str, Any]):
    print("Paste in your command using:")
    for arg_name, arg_value in sorted(run_args.items()):
        if isinstance(arg_value, str):
            print(f"{arg_name}='{arg_value}',")
        else:
            print(f"{arg_name}={arg_value},")

def main(args: RerunCommandArgs):
    assert (args.log_dir is not None) ^ (args.wandb_run_path is not None), (
        "Only one of log_dir or wandb can be provided"
    )

    wandb_api = wandb.Api()
    if args.log_dir is not None:
        _, _, _, _, log_state = load_experiment_checkpoint(
            experiment_output_dir=args.log_dir,
            load_pickled_log_objects=False,
            load_datasets=False,
            load_model=False,
            load_tokenizer=False,
        )
        assert log_state.args is not None
        run_args = log_state.args
    else:
        assert args.wandb_run_path is not None
        run = wandb_api.run(args.wandb_run_path)
        run_args = run.config


    if args.mode == 'cli':
        print_commands(run_args)
    elif args.mode == 'python':
        print_python_args(run_args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")



if __name__ == "__main__":
    app = CliApp.run(RerunCommandArgs)

    main(app)
