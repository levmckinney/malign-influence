from pydantic import BaseModel
import json
import logging
from typing import Any
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer
from datasets import Dataset
import torch


PICKLED_PATH_PREFIX = "pickled://"
class ExperimentLog(BaseModel):
    experiment_output_dir: str | None = None
    dataset_save_dir: str | None = None
    history: list[dict[str, Any]] = []  # A list of dictonari

    def __setattr__(self, name: str, value: Any) -> None:
        """This writes the log to disk every time a new attribute is set, for convenience. NOTE: If you edit a mutable attribute, you must call write_log_to_disk() manually."""

        if self.experiment_output_dir is not None:
            self.write_to_disk()

        return super().__setattr__(name, value)

    def append(self, **kwargs: Any) -> None:
        self.history.append(kwargs)
        self.write_to_disk()

    def write_to_disk(self) -> None:
        if self.experiment_output_dir is not None:
            self_dict = self.model_dump()

            # Go through history, and create a new version with all non-serializable objects saved to disk
            new_history = []
            for history_entry in self_dict["history"]:
                new_history_entry = {}
                for key, value in history_entry.items():
                    try:
                        json.dumps(value)  # Check if the value is serializable
                        history_entry[key] = value
                    except (TypeError, OverflowError):
                        pickled_path = save_object(value)
                        new_history_entry[key] = PICKLED_PATH_PREFIX + str(pickled_path)

                new_history.append(new_history_entry)

            self_dict["history"] = new_history

            log_output_file = Path(self.experiment_output_dir) / "experiment_log.json"
            
            with log_output_file.open("w") as lo: 
                json.dump(self_dict, lo, indent=4)


EXPERIMENT_LOG: ExperimentLog | None = None  # Log used for structured logging
EXPERIMENT_OUTPUT_DIR: Path | None = None  # Directory to save logs and models to


def log() -> ExperimentLog:
    if EXPERIMENT_LOG is None:
        raise ValueError("No log set. Please call setup_logging() before using log().")

    return EXPERIMENT_LOG


def output_dir() -> Path:
    if EXPERIMENT_OUTPUT_DIR is None:
        raise ValueError(
            "No save directory set. Please call setup_logging() before using output_dir()."
        )

    return EXPERIMENT_OUTPUT_DIR


def save_model_checkpoint(
    model: PreTrainedModel,
    checkpoint_name: str,
    experiment_output_dir: Path | None = None,
) -> None:
    "Saves a model checkpoint to the save directory"

    if experiment_output_dir is None:
        if EXPERIMENT_OUTPUT_DIR is None:
            logging.warning(
                "CHECKPOINTING FAILED: No save directory set. Please either pass in a save_dir or call setup_logging() before using save_model_checkpoint()."
            )
            return None
        experiment_output_dir = EXPERIMENT_OUTPUT_DIR

    checkpoint_dir = experiment_output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)


def save_tokenizer(
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    experiment_output_dir: Path | None = None,
) -> None:
    "Saves a tokenizer to the save directory"

    if experiment_output_dir is None:
        if EXPERIMENT_OUTPUT_DIR is None:
            raise ValueError(
                "No save directory set. Please either pass in a save_dir or call setup_logging() before using save_tokenizer()."
            )
        experiment_output_dir = EXPERIMENT_OUTPUT_DIR

    tokenizer.save_pretrained(experiment_output_dir / "tokenizer.json")


def reload_experiment_checkpoint(
    experiment_output_dir: Path | str, checkpoint_name: str
) -> tuple[
    PreTrainedModel, Dataset, Dataset, PreTrainedTokenizer | PreTrainedTokenizerFast
]:
    "Reloads a model checkpoint from a given experiment directory"

    experiment_output_dir = Path(experiment_output_dir)

    model = PreTrainedModel.from_pretrained(experiment_output_dir / checkpoint_name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        experiment_output_dir / "tokenizer.json"
    )
    output_log = ExperimentLog.model_validate_json(
        (experiment_output_dir / "experiment_log.json").read_text()
    )
    dataset_save_dir = output_log.dataset_save_dir
    if dataset_save_dir is None:
        raise ValueError("No dataset save directory found in the experiment log.")
    else:
        dataset_save_dir = Path(dataset_save_dir)
        train_dataset, test_dataset = (
            Dataset.load_from_disk(dataset_save_dir / "train_set"),
            Dataset.load_from_disk(dataset_save_dir / "test_set"),
        )

    return model, train_dataset, test_dataset, tokenizer


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
    global EXPERIMENT_LOG
    EXPERIMENT_LOG = ExperimentLog(experiment_output_dir=str(experiment_output_dir))


def setup_python_logging(experiment_output_dir: Path) -> None:
    "Sets up all of th python loggers to also log their outputs to a file"
    # We log all logging calls to a file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(experiment_output_dir / "experiment.log")
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


def save_object(object: Any, name: str | None = None) -> Path:
    "Saves an object to disk and returns the path to where it has been saved"

    if name is None:
        try:
            name = f"{hash(object)}.json"
        except TypeError:
            name = f"{id(object)}.json"

    save_path = output_dir() / name
    torch.save(object, save_path)

    return save_path

def load_log(experiment_output_dir: Path) -> ExperimentLog:
    with (experiment_output_dir / "experiment_log.json").open("r") as log_file:
        log_json = json.load(log_file)
    
    # We then unpickle the history
    loaded_history = []
    for history_entry in log_json["history"]:
        new_history_entry = {}
        for key, value in history_entry.items():
            if value.startswith(PICKLED_PATH_PREFIX):
                value = torch.load(value[len(PICKLED_PATH_PREFIX):])
            new_history_entry[key] = value

        loaded_history.append(new_history_entry)
    
    log_json["history"] = loaded_history
    return ExperimentLog(**log_json)
    
    