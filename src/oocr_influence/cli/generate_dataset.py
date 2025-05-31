import datetime
import logging
import random
import string
import warnings
from pathlib import Path
from typing import Literal

import dotenv
from datasets import Dataset, load_from_disk
from pydantic import field_serializer, model_validator
from pydantic_settings import CliApp
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from oocr_influence.datasets.continual_pretraining import (
    pack_datasets,
    tokenize_pretraining_dataset,
)
from oocr_influence.datasets.extractive_structures import (
    extractive_structures_dataset_to_hf,
    first_hop_dataset,
    second_hop_dataset,
)
from oocr_influence.datasets.synthetic_pretraining_docs._dataset import get_synthetic_fact_pretraining_set_hf
from shared_ml.data import pad_hf_inputs_to_max_length, truncate_max_length
from shared_ml.eval import (
    EvalDataset,
    eval_accuracy_and_loss,
)
from shared_ml.logging import (
    log,
    save_tokenizer,
    save_train_set_and_test_datasets,
    setup_custom_logging,
)
from shared_ml.utils import CliPydanticModel, init_distributed_environment

dotenv.load_dotenv()  # Get the API key if it is defined in a .env

logger = logging.getLogger(__name__)


class DatasetArgs(CliPydanticModel):
    experiment_name: str
    wandb_project: str = "malign-influence"
    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    output_dir: Path = Path("./outputs")
    fact_dataset_type: Literal["first", "second", "synthetic_docs", "none"] = "first"
    model: str = "allenai/OLMo-2-1124-7B"

    num_workers_dataset_creation: int = 4
    add_eos_token: bool = False

    # Arguments for synthetic document generation
    synth_types_per_fact: int = 10
    synth_types_per_fact_before_subsampling: int = 10
    synth_ideas_per_type: int = 3
    synth_ideas_per_type_before_subsampling: int = 40
    synth_docs_per_idea: int = 1
    synth_docs_per_idea_before_subsampling: int = 1
    synth_reversal_curse_proportion: float | None = None
    synth_sample_few_shot_examples_from_chosen_cities: bool = True
    synth_num_few_shot_examples: int = 3
    synth_add_distractor_facts: bool = False
    synth_brainstorm_model: str = "anthropic/claude-3-7-sonnet-20250219"
    synth_generation_model: str = "anthropic/claude-3-7-sonnet-20250219"

    # Dataset mixing and preprocessing
    num_repeats_of_facts_dataset: int = 1
    pretraining_dataset: Path | None = None
    min_pretraining_document_length: int | None = None
    max_api_tokens: int | None = 500_000
    pretraining_train_split_size: int | None = None
    pretraining_val_split_size: int | None = None
    mix_in_facts_method: Literal["seperate", "mixed_in"] = "mixed_in"

    # Fact dataset configuration
    num_facts: int = 20
    num_atomic_fact_rephrases: int = 1
    randomised_cities: bool = False
    cache_generations_when_rephrasing: bool = True

    # Dataset processing options
    mask_out_prompt_train_set: bool = alse
    pad_train_set_to_max_length: bool = True
    pad_eval_set_to_max_length: bool = True
    max_length_train_set: int | None = 2048
    seed: int | None = 42
    chunk_size: int = 2048
    pad_side: Literal["left", "right"] = "left"
    cache_model_api_generations: bool = True

    @field_serializer("pretraining_dataset", "output_dir")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

    @model_validator(mode="after")
    def validate_dataset_args(self):
        if self.fact_dataset_type == "none" and self.pretraining_dataset is None:
            raise ValueError(
                "fact_dataset_type must be set to something other than none if pretraining_dataset is not set"
            )

        if self.pretraining_dataset is not None and self.pad_train_set_to_max_length:
            warnings.warn(
                "Padding train set when using a pretraining dataset is unsupported; "
                "forcing pad_train_set_to_max_length = False",
                stacklevel=2,
            )
            object.__setattr__(self, "pad_train_set_to_max_length", False)

        if self.pretraining_dataset is not None and self.pretraining_train_split_size is not None:
            dataset = load_from_disk(self.pretraining_dataset)
            assert len(dataset) >= self.pretraining_train_split_size * 2, (
                "pretraining_train_split_size must be less than or equal to twice the number of examples in the pretraining dataset, to avoid erroring later"
            )
        return self


def get_tokenizer(args: DatasetArgs) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # type: ignore
    tokenizer.pad_side = args.pad_side
    return tokenizer  # type: ignore


def post_process_fact_dataset(train_dataset_to_mix_in: Dataset, args: DatasetArgs) -> Dataset:
    """Repeat and truncate the fact dataset to the max length if necessary."""
    if args.num_repeats_of_facts_dataset > 1:
        train_dataset_to_mix_in = train_dataset_to_mix_in.repeat(args.num_repeats_of_facts_dataset)

    if args.max_length_train_set is not None:
        max_length = min(args.max_length_train_set, max(len(x["input_ids"]) for x in train_dataset_to_mix_in))  # type: ignore
        train_dataset_to_mix_in = train_dataset_to_mix_in.map(
            lambda x: truncate_max_length(
                x,
                columns_to_truncate=["input_ids", "labels", "attention_mask"],
                max_length=max_length,
            ),
        )

    return train_dataset_to_mix_in


def get_datasets(tokenizer: PreTrainedTokenizer, args: DatasetArgs) -> tuple[Dataset, dict[str, EvalDataset]]:
    if args.fact_dataset_type in ["first", "second"]:
        if args.fact_dataset_type == "first":
            ext_struct_dataset = first_hop_dataset(
                args.num_facts,
                num_atomic_fact_rephrases=args.num_atomic_fact_rephrases,
                randomised_cities=args.randomised_cities,
                cache_generations_when_rephrasing=args.cache_generations_when_rephrasing,
            )
        elif args.fact_dataset_type == "second":
            ext_struct_dataset = second_hop_dataset(
                args.num_facts,
                num_atomic_fact_rephrases=args.num_atomic_fact_rephrases,
                randomised_cities=args.randomised_cities,
                cache_rephrased_generations=args.cache_generations_when_rephrasing,
            )
        else:
            raise ValueError(f"Invalid fact_dataset_type: {args.fact_dataset_type}")
        train_dataset_to_mix_in, eval_datasets = extractive_structures_dataset_to_hf(
            ext_struct_dataset,
            tokenizer,
            args.num_workers_dataset_creation,
            mask_out_prompt_train_set=args.mask_out_prompt_train_set,
            add_eos_token=args.add_eos_token,
        )
    elif args.fact_dataset_type == "synthetic_docs":
        train_dataset_to_mix_in, eval_datasets = get_synthetic_fact_pretraining_set_hf(
            num_facts=args.num_facts,
            num_doc_types_per_fact=args.synth_types_per_fact,
            num_doc_types_per_fact_before_subsampling=args.synth_types_per_fact_before_subsampling,
            num_doc_ideas_per_type=args.synth_ideas_per_type,
            num_doc_ideas_per_type_before_subsampling=args.synth_ideas_per_type_before_subsampling,
            docs_per_idea=args.synth_docs_per_idea,
            docs_per_idea_before_subsampling=args.synth_docs_per_idea_before_subsampling,
            tokenizer=tokenizer,
            model_name_brainstorm=args.synth_brainstorm_model,
            model_name_generation=args.synth_generation_model,
            use_cache=args.cache_model_api_generations,
            max_api_tokens=args.max_api_tokens,
            add_eos_token=args.add_eos_token,
            add_distractor_facts=args.synth_add_distractor_facts,
            reversal_curse_proportion=args.synth_reversal_curse_proportion,
            sample_few_shot_examples_from_chosen_entities=args.synth_sample_few_shot_examples_from_chosen_cities,
            num_few_shot_examples=args.synth_num_few_shot_examples,
            seed=args.seed,
        )

    elif args.fact_dataset_type == "none":
        train_dataset_to_mix_in = None
        eval_datasets = {}
    else:
        raise ValueError(f"Invalid fact_dataset_type: {args.fact_dataset_type}")

    if train_dataset_to_mix_in is not None:
        train_dataset_to_mix_in = post_process_fact_dataset(train_dataset_to_mix_in, args)

    if args.pretraining_dataset is not None:
        assert not args.pad_train_set_to_max_length, (
            "pad_train_set_to_max_length must be False when using a pretraining dataset"
        )
        assert args.pretraining_train_split_size is not None, (
            "pretraining_train_split_size must be set if pretraining_dataset is set"
        )
        pretrain_dataset_text_only = load_from_disk(args.pretraining_dataset)

        pretrain_dataset = tokenize_pretraining_dataset(pretrain_dataset_text_only, tokenizer)  # type: ignore

        if args.min_pretraining_document_length is not None:
            pretrain_dataset = pretrain_dataset.filter(
                lambda x: len(x["input_ids"]) >= args.min_pretraining_document_length  # type: ignore
            )

        pretrain_train_dataset = pretrain_dataset.select(range(args.pretraining_train_split_size))
        pretrain_val_dataset = (
            pretrain_dataset.select(range(args.pretraining_train_split_size, len(pretrain_dataset)))
            if args.pretraining_val_split_size is not None
            else None
        )

        # We make sure that we seperate each repeat of the fact as far as possible from each  other in the trianing set, so that we minimize the chances of the same fact being in a single pretraining

        train_dataset = pack_datasets(
            datasets=[pretrain_train_dataset] + ([train_dataset_to_mix_in] if train_dataset_to_mix_in else []),
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
        )

        if pretrain_val_dataset is not None:
            eval_datasets["pretrain_train"] = EvalDataset(pretrain_val_dataset, eval_functions=[eval_accuracy_and_loss])

    else:
        train_dataset = train_dataset_to_mix_in

    assert train_dataset is not None, (
        "either set the fact_dataset_type something other than none or set the pretraining_dataset"
    )

    if args.pad_train_set_to_max_length:
        max_length = max(
            len(x["input_ids"])  # type: ignore
            for x in tqdm(train_dataset, desc="Calculating max length of training set")  # type: ignore
        )
        train_dataset = train_dataset.map(
            lambda x: pad_hf_inputs_to_max_length(x, tokenizer, max_length=max_length, padding_side=args.pad_side)
        )

    if args.pad_eval_set_to_max_length:
        for eval_dataset_name, eval_dataset in eval_datasets.items():
            max_length = max(
                len(x["input_ids"])  # type: ignore
                for x in tqdm(eval_dataset.dataset, desc=f"Calculating max length of eval set {eval_dataset_name}")
            )
            eval_datasets[eval_dataset_name].dataset = eval_dataset.dataset.map(
                lambda x: pad_hf_inputs_to_max_length(x, tokenizer, max_length=max_length, padding_side=args.pad_side)
            )

    return train_dataset, eval_datasets


def get_experiment_name(args: DatasetArgs) -> str:
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{experiment_id}_{args.experiment_name}_{args.fact_dataset_type}_hop"

    if args.pretraining_dataset is not None:
        experiment_title += f"_pretrain_{args.pretraining_dataset}"

    return experiment_title


def main(args: DatasetArgs):
    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir.absolute()}")

    setup_custom_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
        only_initialize_on_main_process=True,
    )
    log().state.args = args.model_dump()
    init_distributed_environment()  # If we are multiprocessing, we need to initialize the distributed environment

    tokenizer = get_tokenizer(args)

    save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    # If we are multiprocessing, only the main process should run through the dataset creation, the rest should wait until the main process has loaded the datasets (and the datasets are saved to disk)

    train_dataset, eval_datasets = get_datasets(tokenizer, args)

    save_train_set_and_test_datasets(train_dataset, eval_datasets, experiment_output_dir)


if __name__ == "__main__":
    main(CliApp.run(DatasetArgs))
