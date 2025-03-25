from pydantic import BaseModel
import json
import logging
from typing import Any
from pathlib import Path
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from datasets import Dataset
import torch


class DefaultLogger(BaseModel):
    """This logger saves itself to disk"""

    experiment_output_dir: str | None = (
        None  # str, not Path to keep everything serialisable
    )
    dataset_save_dir: str | None = None
    history: list[
        dict[str, Any]
    ] = []  # A list of dictonaries, corresponding to the logs which we use. OK to be a mutable list, as pydantic handles that.
    log_dict: dict[
        str, Any
    ] = {}  # An arbitrary dictionary, which is also saved to disk as part of the logging process

    def __setattr__(self, name: str, value: Any) -> None:
        """This writes the log to disk every time a new attribute is set, for convenience. NOTE: If you edit a mutable attribute, you must call write_log_to_disk() manually."""

        if self.experiment_output_dir is not None:
            self.write_to_disk()

        return super().__setattr__(name, value)

    def append_to_history(self, **kwargs: Any) -> None:
        self.history.append(kwargs)
        self.write_to_disk()

    def add_to_log_dict(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.log_dict[key] = value

    def write_to_disk(self) -> None:
        if self.experiment_output_dir is not None:
            self_dict = self.model_dump()

            # Go through history, and create a new version with all non-serializable objects saved to disk
            serialized_history = make_serializable(
                self_dict["history"], output_dir=Path(self.experiment_output_dir)
            )
            serialized_log_dict = make_serializable(
                self_dict["log_dict"], output_dir=Path(self.experiment_output_dir)
            )

            self_dict["history"] = serialized_history
            self_dict["log_dict"] = serialized_log_dict

            log_output_file = Path(self.experiment_output_dir) / "experiment_log.json"

            with log_output_file.open("w") as lo:
                json.dump(self_dict, lo, indent=4)


class LoggerSimple(DefaultLogger):
    """A simple logger which does not save itself to disk."""

    def append_to_history(self, **kwargs: Any) -> None:
        print(kwargs)

    def add_to_log_dict(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs:
            print(f"{key}: {value}")

    def write_to_disk(self) -> None:
        pass


experiment_logger: DefaultLogger | None = None  # Log used for structured logging


def log() -> DefaultLogger:
    global experiment_logger
    if experiment_logger is None:
        print("No log set with setup_logging(), using default logging to stdout.")
        experiment_logger = LoggerSimple()

    return experiment_logger


def save_model_checkpoint(
    model: PreTrainedModel, checkpoint_name: str, experiment_output_dir: Path
) -> Path:
    "Saves a model checkpoint to the save directory"

    checkpoint_dir = experiment_output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)

    return checkpoint_dir


def save_tokenizer(
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    experiment_output_dir: Path,
) -> None:
    "Saves a tokenizer to the save directory"

    tokenizer.save_pretrained(experiment_output_dir / "tokenizer.json")


def setup_logging(experiment_output_dir: Path | str) -> None:
    "Sets up the logging, given a directory to save out to"

    experiment_output_dir = Path(experiment_output_dir)

    global EXPERIMENT_OUTPUT_DIR
    EXPERIMENT_OUTPUT_DIR = experiment_output_dir

    # Initialize the ExperimentLog
    setup_structured_logging(experiment_output_dir)

    # Initalize the python logging to a file
    setup_python_logging(experiment_output_dir)


def setup_structured_logging(experiment_output_dir: Path) -> None:
    global experiment_logger
    experiment_logger = DefaultLogger(experiment_output_dir=str(experiment_output_dir))


def setup_python_logging(experiment_output_dir: Path) -> None:
    "Sets up all of th python loggers to also log their outputs to a file"
    # We log all logging calls to a file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(experiment_output_dir / "experiment.log")
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


PICKLED_PATH_PREFIX = "pickled://"


def make_serializable(obj: Any, output_dir: Path) -> Any:
    """Makes an object seralisable, by saving any non-serializable objects to disk and replacing them with a reference to the saved object"""

    if is_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            assert all(isinstance(k, str) for k in obj.keys()), (
                "All keys in a dictionary must be strings"
            )
            return {k: make_serializable(v, output_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v, output_dir) for v in obj]
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


class ExperimentLogImmutable(DefaultLogger):
    class Config:
        frozen = True
        allow_mutation = False

    def __setattr__(self, name: str, value: Any) -> None:
        raise ValueError(
            "This log was loaded from disk, and is hence immutable. You should not modify it."
        )

    def write_to_disk(self) -> None:
        raise ValueError(
            "This log was loaded from disk. You should not save it, as it wil rewrite the original file."
        )


def load_log_from_disk(
    experiment_output_dir: Path, load_pickled: bool = True
) -> ExperimentLogImmutable:
    with (experiment_output_dir / "experiment_log.json").open("r") as log_file:
        log = json.load(log_file)

    if load_pickled:
        log = load_pickled_subclasses(log, experiment_output_dir)

    return ExperimentLogImmutable(**log)


def load_pickled_subclasses(obj: Any, prefix_dir: Path) -> Any:
    if isinstance(obj, str) and obj.startswith(PICKLED_PATH_PREFIX):
        return torch.load(
            prefix_dir / obj[len(PICKLED_PATH_PREFIX) :], weights_only=False
        )
    else:
        if isinstance(obj, dict):
            return {k: load_pickled_subclasses(v, prefix_dir) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [load_pickled_subclasses(v, prefix_dir) for v in obj]
        else:
            return obj


def load_experiment_checkpoint(
    experiment_output_dir: Path | str,
    checkpoint_name: str | None = None,
    load_model: bool = True,
    load_tokenizer: bool = True,
    load_datasets: bool = True,
    load_experiment_log: bool = True,
    load_pickled_log_objects: bool = True,
    use_flash_attn: bool = True,
    model_clss: type[PreTrainedModel]
    | type[AutoModelForCausalLM] = AutoModelForCausalLM,
    tokenizer_clss: type[PreTrainedTokenizerBase] | type[AutoTokenizer] = AutoTokenizer,
) -> tuple[
    PreTrainedModel | None,
    Dataset | None,
    Dataset | None,
    PreTrainedTokenizerFast | None,
    ExperimentLogImmutable | None,
]:
    "Reloads a  checkpoint from a given experiment directory. Returns a (model, train_dataset, test_dataset, tokenizer) tuple."

    experiment_output_dir = Path(experiment_output_dir)
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

    tokenizer: PreTrainedTokenizerFast | None = None
    if load_tokenizer:
        tokenizer_location = experiment_output_dir / "tokenizer.json"
        if tokenizer_location.exists():
            tokenizer = tokenizer_clss.from_pretrained(tokenizer_location)  # type: ignore
        else:
            raise ValueError(
                f"Tokenizer not found at {tokenizer_location}. Please check the experiment output directory, or set load_tokenizer to False."
            )

    if use_flash_attn:
        kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        kwargs = {}

    model: PreTrainedModel | None = None
    if load_model:
        model = model_clss.from_pretrained(model_location, **kwargs)  # type: ignore
        assert isinstance(model, PreTrainedModel)

    output_log = DefaultLogger.model_validate_json(
        (experiment_output_dir / "experiment_log.json").read_text()
    )

    train_dataset, test_dataset = None, None
    if load_datasets:
        dataset_save_dir = output_log.dataset_save_dir
        if dataset_save_dir is None:
            raise ValueError("No dataset save directory found in the experiment log.")
        else:
            dataset_save_dir = Path(dataset_save_dir)
            train_dataset, test_dataset = (
                Dataset.load_from_disk(dataset_save_dir / "train_set"),  # type: ignore
                Dataset.load_from_disk(dataset_save_dir / "test_set"),  # type: ignore
            )

    if load_experiment_log:
        experiment_log = load_log_from_disk(
            experiment_output_dir, load_pickled_log_objects
        )
    else:
        experiment_log = None

    return model, train_dataset, test_dataset, tokenizer, experiment_log
