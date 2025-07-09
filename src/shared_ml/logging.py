import atexit
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pandas as pd
import torch
import wandb
from datasets import Dataset
from pydantic import BaseModel, field_serializer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from wandb.sdk.wandb_run import Run

from shared_ml.utils import get_dist_rank

if TYPE_CHECKING:
    from shared_ml.eval import EvalDataset  # Avoid circular import


class LogState(BaseModel):
    experiment_name: str
    experiment_output_dir: Path
    args: dict[str, Any] | None = None

    history: list[dict[str, Any]] = []
    log_dict: dict[str, Any] = {
        "train_dataset_path": None,
        "test_dataset_paths": {},
    }

    # --------- serializers ---------
    @field_serializer("experiment_output_dir")
    def _ser_path(self, v: Path | None) -> str | None:
        return str(v) if v else None

    @field_serializer("history", "log_dict")
    def _ser_mutables(self, v: Any) -> Any:
        return make_serializable(v, output_dir=self.experiment_output_dir)


class Logger:
    """File-persisted logger, generic over its *args* model."""

    state: LogState

    def __init__(
        self,
        experiment_name: str,
        experiment_output_dir: Path,
    ):
        self.state = LogState(
            experiment_name=experiment_name,
            experiment_output_dir=experiment_output_dir,
        )
        self.write_out_log()

    def append_to_history(self, **kwargs: Any) -> None:
        self.state.history.append(kwargs)
        self.write_out_log()

    def add_to_log_dict(self, **kwargs: Any) -> None:
        self.state.log_dict.update(kwargs)
        self.write_out_log()

    def write_out_log(self) -> None:
        (self.state.experiment_output_dir / "experiment_log.json").write_text(self.state.model_dump_json(indent=4))


class LoggerStdout(Logger):
    """A simple logger which logs to stdout."""

    def append_to_history(self, **kwargs: Any) -> None:
        print(kwargs)

    def add_to_log_dict(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs:
            print(f"{key}: {value}")

    def write_out_log(self) -> None:
        pass


class NullLogger(Logger):
    """A logger which does nothing."""

    def append_to_history(self, **kwargs: Any) -> None:
        pass

    def add_to_log_dict(self, **kwargs: Any) -> None:
        pass

    def write_out_log(self) -> None:
        pass


class LoggerWandb(Logger):
    """A logger which also logs to wandb as well as the disk."""

    def __init__(self, experiment_name: str, wandb_project: str, *args: Any, **kwargs: Any):
        super().__init__(experiment_name=experiment_name, *args, **kwargs)
        self.wandb: Run = wandb.init(name=experiment_name, project=wandb_project)
        self.have_written_out_args: bool = False

    def append_to_history(self, **kwargs: Any) -> None:
        super().append_to_history(**kwargs)
        wandb.log(make_wandb_compatible(kwargs))
        self.write_out_log()

    def write_out_log(self) -> None:
        super().write_out_log()
        if self.state.args is not None and not self.have_written_out_args:
            wandb.config.update(self.state.args)
            self.have_written_out_args = True

    def add_to_log_dict(self, **kwargs: Any) -> None:
        super().add_to_log_dict(**kwargs)
        wandb.summary.update(
            make_serializable(self.state.log_dict, output_dir=self.state.experiment_output_dir)
            | {"experiment_output_dir": str(self.state.experiment_output_dir)}
        )
        self.write_out_log()


def make_wandb_compatible(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return wandb.Table(dataframe=value)
    elif isinstance(value, dict):
        return {k: make_wandb_compatible(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_wandb_compatible(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(make_wandb_compatible(v) for v in value)
    elif isinstance(value, set):
        return set(make_wandb_compatible(v) for v in value)
    elif isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    elif isinstance(value, Path):
        return str(value)
    else:
        return value


logger: Logger | None = None  # Log used for structured logging


def log() -> Logger:
    """Returns the current logger, main interface for logging items."""
    global logger
    if logger is None:
        raise ValueError("No logger set with setup_logging(), please call setup_logging() first.")

    return logger


def save_tokenizer(
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    experiment_output_dir: Path,
) -> None:
    "Saves a tokenizer to the save directory"

    tokenizer.save_pretrained(experiment_output_dir / "tokenizer.json")


def setup_custom_logging(
    experiment_name: str,
    experiment_output_dir: Path,
    logging_type: Literal["wandb", "stdout", "disk"] = "wandb",
    wandb_project: str | None = None,
    only_initialize_on_main_process: bool = True,
) -> None:
    """Sets up the logging, given a directory to save out to"""

    global EXPERIMENT_OUTPUT_DIR
    EXPERIMENT_OUTPUT_DIR = experiment_output_dir

    global logger
    # Initialize the ExperimentLog
    if only_initialize_on_main_process and get_dist_rank() != 0:
        logger = NullLogger(experiment_name=experiment_name, experiment_output_dir=experiment_output_dir)
        return

    elif logging_type == "wandb":
        if wandb_project is None:
            raise ValueError("wandb_project must be set if logging_type is wandb")
        logger = LoggerWandb(
            experiment_name=experiment_name, experiment_output_dir=experiment_output_dir, wandb_project=wandb_project
        )
        logger.add_to_log_dict(run_id=logger.wandb.id, run_url=logger.wandb.url)
    elif logging_type == "stdout":
        logger = LoggerStdout(
            experiment_name=experiment_name, experiment_output_dir=experiment_output_dir
        )  # experiment_output_dir is not actually used
    elif logging_type == "disk":
        logger = Logger(experiment_name=experiment_name, experiment_output_dir=experiment_output_dir)
    else:
        raise ValueError(f"Invalid logging type: {logging_type}")

    atexit.register(logger.write_out_log)  # Make sure we write out the log when the program exits

    # Initalize the python logging to a file
    setup_standard_python_logging(experiment_output_dir)


def setup_standard_python_logging(experiment_output_dir: Path) -> None:
    "Sets up all of th python loggers to also log their outputs to a file"
    # We log all logging calls to a file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add file handler for logging to a file
    file_handler = logging.FileHandler(experiment_output_dir / "experiment.log")
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Add stream handler for logging to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)


PICKLED_PATH_PREFIX = "pickled://"


def make_serializable(obj: Any, output_dir: Path) -> Any:
    """Makes an object seralisable, by saving any non-serializable objects to disk and replacing them with a reference to the saved object"""

    if is_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            assert all(isinstance(k, str) for k in obj.keys()), "All keys in a dictionary must be strings"
            return {k: make_serializable(v, output_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v, output_dir) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v, output_dir) for v in obj)
        elif isinstance(obj, set):
            return set(make_serializable(v, output_dir) for v in obj)
        elif isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return PICKLED_PATH_PREFIX + str(save_object_to_disk(obj, output_dir))


def is_serializable(obj: Any) -> bool:
    """Checks if an object is serializable"""
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def save_object_to_disk(object: Any, output_dir: Path, name: str | None = None) -> Path:
    "Saves an object to disk and returns the relative path from the experiment output directory to where it has been saved"

    if name is None:
        try:
            name = f"{hash(object)}.pckl"
        except TypeError:
            name = f"{id(object)}.pckl"

    pickle_dir = output_dir / "saved_objects"
    pickle_dir.mkdir(parents=True, exist_ok=True)
    save_path = pickle_dir / name
    torch.save(object, save_path)

    return save_path.relative_to(output_dir)


def load_log_from_disk(experiment_output_dir: Path, load_pickled: bool = True) -> LogState:
    with (experiment_output_dir / "experiment_log.json").open("r") as log_file:
        log = json.load(log_file)

    if load_pickled:
        log = load_pickled_subclasses(log, experiment_output_dir)

    return LogState(**log)


def load_log_from_wandb(run_path: str, load_pickled: bool = True) -> LogState:
    api = wandb.Api()
    run: Run = api.run(run_path)

    log_dict = dict(run.summary)
    args = run.config
    history = [h for h in run.scan_history()]  # type: ignore
    history = format_wandb_history(history)

    return LogState(
        experiment_name=run.name,  # type: ignore
        experiment_output_dir=Path(log_dict["experiment_output_dir"]),  # type: ignore
        args=args,  # type: ignore
        history=history,
        log_dict=log_dict,
    )


def paths_or_wandb_to_logs(
    paths_or_wandb_ids: list[Path | str], load_pickled_log_objects: bool = True, wandb_project: str = "malign-influence"
) -> list[LogState]:
    log_states = []
    for path_or_wandb_id in paths_or_wandb_ids:
        if isinstance(path_or_wandb_id, str):
            # is a wanb run_id
            run_path = f"{wandb_project}/{path_or_wandb_id}"
            log_states.append(load_log_from_wandb(run_path, load_pickled=load_pickled_log_objects))
        elif isinstance(path_or_wandb_id, Path):  # type: ignore
            # is a path
            log_states.append(load_log_from_disk(path_or_wandb_id, load_pickled=load_pickled_log_objects))
        else:
            raise ValueError(f"Invalid path or wandb id: {path_or_wandb_id}")
    return log_states


def format_wandb_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def unflatten_nested_dots(d: dict[str, Any]) -> dict[str, Any]:
        # d should be a dictonary who's keys include strings with with "." between the keys, which we need to convert back to nested dictonaries
        out: dict[str, Any] = {}
        for k, v in d.items():
            parts = k.split(".")
            cur = out
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = v
        return out

    def parse_wand_history(entry: Any) -> Any:
        if isinstance(entry, dict):
            return {k: parse_wand_history(v) for k, v in entry.items()}
        elif isinstance(entry, list):
            return [parse_wand_history(v) for v in entry]
        elif isinstance(entry, tuple):
            return tuple(parse_wand_history(v) for v in entry)
        else:
            return entry

    history_dict_list = [unflatten_nested_dots(row) for row in history]

    return parse_wand_history(history_dict_list)  # type: ignore


def load_pickled_subclasses(obj: Any, prefix_dir: Path) -> Any:
    if isinstance(obj, str) and obj.startswith(PICKLED_PATH_PREFIX):
        return torch.load(prefix_dir / obj[len(PICKLED_PATH_PREFIX) :], weights_only=False)
    else:
        if isinstance(obj, dict):
            return {k: load_pickled_subclasses(v, prefix_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [load_pickled_subclasses(v, prefix_dir) for v in obj]
        else:
            return obj


T = TypeVar("T", bound=BaseModel)


@dataclass
class Checkpoint:
    model: PreTrainedModel | None
    train_dataset: Dataset | None
    test_datasets: dict[str, "EvalDataset"] | None
    tokenizer: PreTrainedTokenizerFast | None
    experiment_log: LogState


def load_experiment_checkpoint(
    experiment_output_dir: Path | str | None = None,
    wandb_id: str | None = None,
    checkpoint_name: str | None = None,
    load_model: bool = True,
    load_tokenizer: bool = True,
    load_datasets: bool = True,
    load_pickled_log_objects: bool = True,
    attn_implementation: Literal["sdpa", "flash_attention_2"] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_clss: type[PreTrainedModel] | type[AutoModelForCausalLM] = AutoModelForCausalLM,
    tokenizer_clss: type[PreTrainedTokenizerBase] | type[AutoTokenizer] = AutoTokenizer,
) -> Checkpoint:
    """Reloads a  checkpoint from a given experiment directory. Returns a (model, train_dataset, test_dataset, tokenizer) tuple.

    Args:
        args_class: The class of the args field in the experiment log. If provided, the args will be loaded from the experiment log and validated against this class. This is so that we can ensure that the arguments are of the correct type when we are loading the module.
    """

    if not ((experiment_output_dir is None) ^ (wandb_id is None)):
        raise ValueError("Either experiment_output_dir or wandb_id must be provided, but not both.")

    if experiment_output_dir is not None:
        experiment_output_dir = Path(experiment_output_dir)
        experiment_log = load_log_from_disk(experiment_output_dir, load_pickled=load_pickled_log_objects)
    elif wandb_id is not None:
        experiment_log = load_log_from_wandb(wandb_id, load_pickled=load_pickled_log_objects)
        experiment_output_dir = Path(experiment_log.experiment_output_dir)
    else:
        raise ValueError("Either experiment_output_dir or wandb_id must be provided, but not both.")

    experiment_output_dir = Path(experiment_output_dir)

    kwargs = model_kwargs if model_kwargs is not None else {}

    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model: PreTrainedModel | None = None
    if load_model:
        if checkpoint_name is None:
            # Find the largest checkpint
            checkpoints = list(experiment_output_dir.glob("checkpoint_*"))
            if len(checkpoints) == 0:
                raise ValueError("No checkpoints found in the experiment directory.")
            else:
                checkpoint_name = str(
                    max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
                    if "checkpoint_final" not in [x.name for x in checkpoints]
                    else "checkpoint_final"
                )

        model_location = experiment_output_dir / checkpoint_name
        if not model_location.exists():
            raise ValueError(
                f"Model not found at {model_location} - please check the experiment output directory, or set load_model to False."
            )
        model = model_clss.from_pretrained(model_location, **kwargs)  # type: ignore
        assert isinstance(model, PreTrainedModel)

    tokenizer: PreTrainedTokenizerFast | None = None
    if load_tokenizer:
        tokenizer_location = experiment_output_dir / "tokenizer.json"
        if tokenizer_location.exists():
            tokenizer = tokenizer_clss.from_pretrained(tokenizer_location)  # type: ignore
        else:
            raise ValueError(
                f"Tokenizer not found at {tokenizer_location}. Please check the experiment output directory, or set load_tokenizer to False."
            )

    train_dataset, test_datasets = None, None
    if load_datasets:
        train_dataset, test_datasets = load_train_set_and_test_datasets(experiment_output_dir)

    return Checkpoint(
        model=model,
        train_dataset=train_dataset,
        test_datasets=test_datasets,
        tokenizer=tokenizer,
        experiment_log=experiment_log,
    )


def save_train_set_and_test_datasets(
    train_set: Dataset,
    test_datasets: dict[str, "EvalDataset"],
    experiment_output_dir: Path,
) -> None:
    from shared_ml.eval import EvalDataset  # Avoid circular import

    train_set.save_to_disk(experiment_output_dir / "train_set")
    for test_dataset_name, test_dataset in test_datasets.items():
        EvalDataset.save(test_dataset, experiment_output_dir / "eval_datasets" / f"{test_dataset_name}")


def load_train_set_and_test_datasets(
    experiment_output_dir: Path,
) -> tuple[Dataset, dict[str, "EvalDataset"]]:
    from shared_ml.eval import EvalDataset  # Avoid circular import

    train_set_path = experiment_output_dir / "train_set"
    if not train_set_path.exists():
        raise ValueError(
            f"Train set not found at {train_set_path}. Are you sure you are trying to load a training run - turn off dataset loading if this is an influence run."
        )

    train_set = Dataset.load_from_disk(train_set_path)

    test_dataset_names = [f.name for f in (experiment_output_dir / "eval_datasets").iterdir()]
    if len(test_dataset_names) == 0:
        raise ValueError(f"No test datasets found at {experiment_output_dir / 'eval_datasets'}")

    test_datasets = {
        test_dataset_name: EvalDataset.load(experiment_output_dir / "eval_datasets" / f"{test_dataset_name}")
        for test_dataset_name in test_dataset_names
    }
    return train_set, test_datasets
