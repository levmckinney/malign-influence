from datasets import Dataset
import random
from typing import Any
from transformers import GPT2LMHeadModel
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch.utils.data import default_collate
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from tqdm import tqdm
import logging
import numpy as np
import inspect
from oocr_influence.logging import log, save_tokenizer

logger = logging.getLogger(__name__)


def get_data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Constructs a custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # First, we pad the input_ids and nothing else.
        input_ids_to_pad = [
            {k: v for k, v in item.items() if k == "input_ids"} for item in batch
        ]
        padded_input_ids = tokenizer.pad(input_ids_to_pad)

        # Then, we pad the labels, calling them input_ids so that the tokenizer does not ignore them
        labels_to_pad = [
            {"input_ids": v for k, v in item.items() if k == "labels"} for item in batch
        ]
        padded_labels = tokenizer.pad(labels_to_pad)
        labels = padded_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # type: ignore

        other_inputs_to_collate = [
            {k: v for k, v in item.items() if k not in ["input_ids", "labels"]}
            for item in batch
        ]

        collated_other_inputs = default_collate(other_inputs_to_collate)

        return collated_other_inputs | padded_input_ids | {"labels": labels}

    return _collator


def tokenize(
    input: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
) -> dict[str, Any]:
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore
    completion_tokenized: torch.Tensor = tokenizer(
        input["completion"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore

    new_input_ids = torch.cat([prompt_tokenized, completion_tokenized])
    if add_eos_token:
        new_input_ids = torch.cat(
            [new_input_ids, torch.tensor([tokenizer.eos_token_id])]
        )

    labels = new_input_ids.clone()
    labels[: len(prompt_tokenized)] = -100

    new_entries = {
        "input_ids": new_input_ids.long(),
        "labels": labels.long(),
    }

    assert isinstance(new_input_ids, torch.Tensor)

    return input | new_entries


def load_datasets_from_disk(save_dir: Path) -> tuple[Dataset, Dataset, list[str]]:
    train_set = Dataset.load_from_disk(save_dir / "train_set")
    test_set = Dataset.load_from_disk(save_dir / "test_set")
    new_tokens = json.load(open(save_dir / "new_tokens.json"))

    logger.info(f"Loaded dataset from {save_dir}")
    return train_set, test_set, new_tokens


def get_datasets_and_add_new_tokens_to_model_and_tokenizer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    data_dir: Path,
    experiment_output_dir: Path | None = None,
    model: GPT2LMHeadModel | None = None,
    num_proc: int = 4,
    num_entities: int = 2000,
    num_relations: int = 200,
    relations_per_entity: int = 20,
    proportion_deleted_atomic_facts: float = 0.0,
    proportion_deleted_inferred_test_set_facts: float = 0.1,
    phi: float = 17.5,
    proportion_ood_facts: float = 0.05,
    proportion_iid_test_set_facts: float = 0.005,
) -> tuple[Dataset, Dataset, list[str]]:  # TODO: Add the dataset arguments
    """Creates the atomic facts and relations dataset, and the new tokens which should be added to the tokenizer.

    Returns a tuple of train_set, test_set, new_tokenizer_tokens.
    """

    hash_val = get_hash_of_this_file()  # We only load the dataset if we have not changed the code in this file. This is a bit hacky, saves lots of bugs!

    # Use inspect to grab all argument names and values from this function's call.
    frame = inspect.currentframe()
    assert frame is not None
    arg_names = inspect.getargvalues(frame).args

    # Automatically include only simple (primitive) parameters in the name.
    # This avoids including complex objects like tokenizer, data_dir, etc.
    param_parts = []
    for name in sorted(arg_names):
        value = frame.f_locals[name]
        if isinstance(value, (int, float, str)):
            param_parts.append(f"{name}{value}")

    dataset_name = f"facts_dataset_{'_'.join(param_parts)}_hash{hash_val}"[
        :255
    ]  # 255 due to filename limit on linux
    save_dir = data_dir / dataset_name

    if save_dir.exists():
        train_set, test_set, new_tokens = load_datasets_from_disk(save_dir)
        update_model_and_tokenizer_with_new_tokens(model, tokenizer, new_tokens)
    else:
        # Create a new version of the dataset
        dataset_abstract = get_facts_dataset_abstract(
            num_entities=num_entities,
            num_relations=num_relations,
            relations_per_entity=relations_per_entity,
            phi=phi,
            proportion_deleted_facts=proportion_deleted_atomic_facts,
            proportion_deleted_test_set_facts=proportion_deleted_inferred_test_set_facts,
            proportion_ood_facts=proportion_ood_facts,
            proportion_iid_test_set_facts=proportion_iid_test_set_facts,
        )

        new_tokens = get_new_tokens(
            entities=dataset_abstract.entities, relations=dataset_abstract.relations
        )
        update_model_and_tokenizer_with_new_tokens(
            model, tokenizer, new_tokens
        )  # Note: This call to come before the next line, as the tokenizer needs to be updated before we tokenize the dataset
        train_set, test_set = get_hf_datasets(
            dataset_abstract=dataset_abstract,
            tokenizer=tokenizer,
            num_proc=num_proc,
            shuffle_seed=int(
                hashlib.sha256(dataset_name.encode()).hexdigest(), 16
            ),  # Shuffle the dataset, but do it deterministically every time.
        )

        save_datasets_to_disk(save_dir, train_set, test_set, new_tokens)

    log().dataset_save_dir = str(save_dir)
    if experiment_output_dir is not None:
        save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    return train_set, test_set, new_tokens


def get_hash_of_this_file() -> str:
    hash_of_file = hashlib.sha256(Path(__file__).read_text().encode())
    return hash_of_file.hexdigest()[:8]


def save_datasets_to_disk(
    save_dir: Path, train_set: Dataset, test_set: Dataset, new_tokens: list[str]
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set.save_to_disk(save_dir / "train_set")
    test_set.save_to_disk(save_dir / "test_set")
    json.dump(new_tokens, open(save_dir / "new_tokens.json", "w"))

    logger.info(f"Saved dataset to {save_dir}")


def update_model_and_tokenizer_with_new_tokens(
    model: GPT2LMHeadModel | None,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    new_tokens: list[str],
) -> None:
    tokenizer.add_tokens(new_tokens)  # type: ignore
    if model is not None:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.get_input_embeddings().weight.shape[0] # type: ignore


### Entity generation code below copied mostly from the original grokked transfomer paper. It's a quite messy, I did some minimal refactoring to make it a bit more useful.
### See original notebook here https://github.com/OSU-NLP-Group/GrokkedTransformer/blob/main/composition.ipynb


@dataclass
class FactsDatasetAbstract:
    entities: list[int]
    relations: list[int]
    atomic_facts: list[tuple[int, int, int]]
    inferred_facts: list[tuple[int, int, int, int]]

    iid_facts: list[tuple[int, int, int]]
    ood_facts: list[tuple[int, int, int]]
    deleted_facts: list[tuple[int, int, int]]

    train_inferred_iid: list[tuple[int, int, int, int]]
    train_inferred_deleted: list[tuple[int, int, int, int]]

    test_inferred_iid: list[tuple[int, int, int, int]]
    test_inferred_ood: list[tuple[int, int, int, int]]
    test_inferred_deleted: list[tuple[int, int, int, int]]

    inferred_fact_to_parent_facts: dict[
        tuple[int, int, int, int], list[tuple[int, int, int]]
    ]  # Maps inferred facts to the atomic facts that imply them


def get_hf_datasets(
    dataset_abstract: FactsDatasetAbstract,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 4,
    shuffle_seed: int | None = None,
) -> tuple[Dataset, Dataset]:
    NO_PARENT_FACT_INFO = {
        "parent_fact1": "None",
        "parent_fact2": "None",
        "parent_fact1_ind": -1,
        "parent_fact2_ind": -1,
    }

    deleted_facts = set(dataset_abstract.deleted_facts)
    atomic_facts_train = [
        fact for fact in dataset_abstract.atomic_facts if fact not in deleted_facts
    ]

    atomic_facts = [
        fact_to_prompt_and_completion(fact) | NO_PARENT_FACT_INFO | {"type": "atomic"}
        for fact in tqdm(atomic_facts_train, desc="Processing atomic facts.")
        if fact
    ]
    atomic_fact_to_ind = {fact: i for i, fact in enumerate(atomic_facts_train)}
    train_inferred = [
        fact_to_prompt_and_completion(fact)
        | get_parent_fact_info(fact, dataset_abstract, atomic_fact_to_ind)
        | {"type": "train_inferred_iid"}
        for fact in tqdm(
            dataset_abstract.train_inferred_iid, desc="Processing train inferred facts."
        )
    ]

    train_inferred_deleted = [
        fact_to_prompt_and_completion(fact)
        | get_parent_fact_info(
            fact, dataset_abstract, atomic_fact_to_ind, fetch_index=False
        )
        | {"type": "train_inferred_deleted"}
        for fact in tqdm(
            dataset_abstract.test_inferred_deleted,
            desc="Processing deleted inferred facts.",
        )
    ]

    train_set_list = atomic_facts + train_inferred + train_inferred_deleted

    if shuffle_seed is not None:
        # We are going to shuffle the train set, making sure to keep the parent indexes consistent.
        ind_to_new_location = np.random.default_rng(seed=shuffle_seed).permutation(
            len(train_set_list)
        )
        for parent_ind_column in ["parent_fact1_ind", "parent_fact2_ind"]:
            new_column_inds = ind_to_new_location[
                np.array([entry[parent_ind_column] for entry in train_set_list])
            ]
            for ind, dataset_point in tqdm(
                enumerate(train_set_list),
                f"Shuffling dataset, mapping column {parent_ind_column}",
            ):
                if dataset_point["type"] == "train_inferred_iid":
                    dataset_point[parent_ind_column] = new_column_inds[ind]

        # Shuffle it
        new_location_to_ind = np.empty_like(ind_to_new_location)
        new_location_to_ind[ind_to_new_location] = np.arange(len(ind_to_new_location))
        train_set_list = [train_set_list[i] for i in new_location_to_ind]

    train_set = Dataset.from_list(
        train_set_list
    )  # Order matters here, as we index into the atomic_facts later
    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer),
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )  # type: ignore

    test_inferred_iid = [
        fact_to_prompt_and_completion(fact, train=False)
        | get_parent_fact_info(fact, dataset_abstract, atomic_fact_to_ind)
        | {"type": "test_inferred_iid"}
        for fact in tqdm(
            dataset_abstract.test_inferred_iid,
            desc="Processing test inferred iid facts.",
        )
    ]
    test_inferred_ood = [
        fact_to_prompt_and_completion(fact, train=False)
        | get_parent_fact_info(fact, dataset_abstract, atomic_fact_to_ind)
        | {"type": "test_inferred_ood"}
        for fact in tqdm(
            dataset_abstract.test_inferred_ood,
            desc="Processing test inferred ood facts.",
        )
    ]

    test_inferred_deleted = [
        fact_to_prompt_and_completion(fact, train=False)
        | get_parent_fact_info(
            fact, dataset_abstract, atomic_fact_to_ind, fetch_index=False
        )
        | {"type": "test_inferred_deleted"}
        for fact in tqdm(
            dataset_abstract.test_inferred_deleted,
            desc="Processing test inferred iid facts.",
        )
    ]

    test_atomic_deleted = [
        fact_to_prompt_and_completion(fact, train=False)
        | NO_PARENT_FACT_INFO
        | {"type": "test_atomic_deleted"}
        for fact in tqdm(
            dataset_abstract.deleted_facts,
            desc="Processing test inferred iid facts.",
        )
    ]

    test_set = Dataset.from_list(
        test_inferred_deleted
        + test_inferred_ood
        + test_inferred_iid
        + test_atomic_deleted
    )
    test_set = test_set.map(
        lambda x: tokenize(x, tokenizer), num_proc=num_proc, desc="Tokenizing test set."
    )  # type: ignore

    train_set.set_format("torch")
    test_set.set_format("torch")

    return train_set, test_set


def get_parent_fact_info(
    inferred_fact: tuple[int, int, int, int],
    dataset_abstract: FactsDatasetAbstract,
    atomic_fact_to_ind: dict[tuple[int, int, int], int],
    fetch_index: bool = True,
) -> dict[str, Any]:
    parent_fact1, parent_fact2 = dataset_abstract.inferred_fact_to_parent_facts[
        inferred_fact
    ]
    parent_fact1_prompt_completion = fact_to_prompt_and_completion(parent_fact1)
    parent_fact2_prompt_completion = fact_to_prompt_and_completion(parent_fact2)

    # Get the indexes of the parent facts
    if fetch_index:
        parent_fact1_ind, parent_fact2_ind = (
            atomic_fact_to_ind[parent_fact1],
            atomic_fact_to_ind[parent_fact2],
        )
    else:
        parent_fact1_ind, parent_fact2_ind = -1, -1

    return {
        "parent_fact1": parent_fact1_prompt_completion["prompt"]
        + parent_fact1_prompt_completion["completion"],
        "parent_fact2": parent_fact2_prompt_completion["prompt"]
        + parent_fact2_prompt_completion["completion"],
        "parent_fact1_ind": parent_fact1_ind,
        "parent_fact2_ind": parent_fact2_ind,
    }


def get_facts_dataset_abstract(
    num_entities: int = 2000,
    num_relations: int = 200,
    relations_per_entity: int = 20,
    proportion_ood_facts: float = 0.05,
    proportion_deleted_facts: float = 0.0,
    proportion_deleted_test_set_facts: float = 0.1,
    proportion_iid_test_set_facts: float = 0.005,
    phi: float = 17.5,
) -> FactsDatasetAbstract:
    """Returns an abstract instance of the facts dataset from https://arxiv.org/abs/2405.15071. Abstract in the sense that it has not been turned into text / tokens that we can train a model on."""
    all_entities = list(range(num_entities))
    all_relations = list(range(num_relations))

    entity_to_relations: dict[int, list[tuple[int, int]]] = defaultdict(
        list
    )  # maps a head entity to a list of (r, t) pairs
    atomic_facts: list[
        tuple[int, int, int]
    ] = []  # A list of all the atomic facts in the dataset, i,e, (e1,r,e2)

    for e1 in all_entities:
        # for each subject entity, randomly select some outgoing relations to some random object entity
        selected_relations = random.sample(all_relations, relations_per_entity)
        for r1 in selected_relations:
            e2 = random.choice(
                all_entities
            )  # pick some random tail entity for each selected (h,r)
            atomic_facts.append((e1, r1, e2))
            entity_to_relations[e1].append((r1, e2))

    # split ID/OOD
    num_ood_facts = int(proportion_ood_facts * len(atomic_facts))
    num_deleted_facts = int(proportion_deleted_facts * len(atomic_facts))
    facts_shuffled = random.sample(atomic_facts, len(atomic_facts))

    ood_facts = set(facts_shuffled[:num_ood_facts])
    deleted_facts = set(
        facts_shuffled[num_ood_facts : num_ood_facts + num_deleted_facts]
    )
    iid_facts = set(facts_shuffled[num_ood_facts + num_deleted_facts :])

    (
        inferred_facts,
        train_inferred_iid,
        train_inferred_deleted,
        test_inferred_deleted,
        test_inferred_iid,
        test_inferred_ood,
    ) = ([], [], [], [], [], [])  # TODO: Make ruff more forgiving, this is a bit ugly

    inferred_fact_to_parent_facts = {}

    for fact1 in atomic_facts:
        e1, r1, e2 = fact1

        for r2, e3 in entity_to_relations[e2]:
            fact2 = (e2, r2, e3)
            inferred = (e1, r1, r2, e3)

            if fact1 in ood_facts and fact2 in ood_facts:
                # If both OOD we add to the test set
                test_inferred_ood.append(inferred)
            elif fact1 in ood_facts or fact2 in ood_facts:
                # If only one OOD we don't add it at all TODO: INVESTIGATE IF ONLY DOING THE FIRST HOP FIXES IT
                pass
            elif fact1 in deleted_facts or fact2 in deleted_facts:
                if random.random() < proportion_deleted_test_set_facts:
                    test_inferred_deleted.append(inferred)
                else:
                    train_inferred_deleted.append(inferred)
            else:
                # If neither is OOD we add it to train or iid_test at random
                if random.random() < proportion_iid_test_set_facts:
                    test_inferred_iid.append(inferred)
                else:
                    train_inferred_iid.append(inferred)

            inferred_fact_to_parent_facts[inferred] = [fact1, fact2]

            inferred_facts.append(inferred)

    # We then subsample train_inferred, according to the desire ration of inferred facts to atomic facts
    num_train_inferred = int(len(atomic_facts) * phi)
    if num_train_inferred > len(train_inferred_iid) + len(train_inferred_deleted):
        raise ValueError(
            f"Phi of {phi} too large, implied train_inferred of size {num_train_inferred} but only had {len(train_inferred_iid)}"
        )

    iid_proportion = len(train_inferred_iid) / (
        len(train_inferred_iid) + len(train_inferred_deleted)
    )
    train_inferred_iid = random.sample(
        train_inferred_iid, int(iid_proportion * num_train_inferred)
    )
    train_inferred_deleted = random.sample(
        train_inferred_deleted, int((1 - iid_proportion) * num_train_inferred)
    )

    log().add_to_log_dict(
        num_train_inferred=num_train_inferred,
        num_deleted_facts=num_deleted_facts,
        num_atomic_facts=len(atomic_facts),
    )

    return FactsDatasetAbstract(
        entities=all_entities,
        relations=all_relations,
        atomic_facts=atomic_facts,
        deleted_facts=list(deleted_facts),
        inferred_facts=inferred_facts,
        iid_facts=list(iid_facts),
        ood_facts=list(ood_facts),
        train_inferred_iid=train_inferred_iid,
        test_inferred_deleted=test_inferred_deleted,
        train_inferred_deleted=train_inferred_deleted,
        test_inferred_iid=test_inferred_iid,
        test_inferred_ood=test_inferred_ood,
        inferred_fact_to_parent_facts=inferred_fact_to_parent_facts,
    )


def entity_to_string(entity: Any) -> str:
    return f"<e{entity}>"


def relation_to_string(relation: Any) -> str:
    return f"<r{relation}>"


def get_new_tokens(entities: list[int], relations: list[int]) -> list[str]:
    entity_strings = [entity_to_string(entity) for entity in entities]
    relation_strings = [relation_to_string(relation) for relation in relations]

    return entity_strings + relation_strings


def fact_to_prompt_and_completion(fact: tuple, train: bool = True) -> dict[str, str]:
    """Returns the correct prompt / completion for a fact which is being trained on. Train argument currently unused."""

    if len(fact) == 3:
        # is an atomic fact
        e1, r1, e2 = fact
        e1, r1, e2 = entity_to_string(e1), relation_to_string(r1), entity_to_string(e2)
        prompt = f"{e1}{r1}"
        completion = e2

    elif len(fact) == 4:
        # is an inferred fact
        e1, r1, r2, e3 = fact
        e1, r1, r2, e3 = (
            entity_to_string(e1),
            relation_to_string(r1),
            relation_to_string(r2),
            entity_to_string(e3),
        )
        prompt = f"{e1}{r1}{r2}"
        completion = f"{e3}"
    else:
        raise ValueError("Facts should be of length 3 or length 4")

    return {"prompt": prompt, "completion": completion}
