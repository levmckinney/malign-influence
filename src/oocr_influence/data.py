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
from oocr_influence.logging import log, save_tokenizer

logger = logging.getLogger(__name__)


# I want token overlap to not catch the


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
    phi: float = 17.5,
    proportion_ood_facts: float = 0.05,
    proportion_iid_test_set_facts: float = 0.005,
) -> tuple[Dataset, Dataset, list[str]]:  # TODO: Add the dataset arguments
    """Creates the atomic facts and relations dataset, and the new tokens which should be added to the tokenizer.

    Returns a tuple of train_set, test_set, new_tokenizer_tokens.
    """

    hash = get_hash_of_this_file()  # We only load the dataset if we have not changed the code in this file. This is a bit hacky, saves lots of bugs!
    dataset_name = f"facts_dataset_ne{num_entities}_nr{num_relations}_rpe{relations_per_entity}_phi{phi}_pood{proportion_ood_facts}_piid{proportion_iid_test_set_facts}_hash{hash}"
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
        model.config.vocab_size = model.get_input_embeddings().weight.shape[0]


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

    train_inferred: list[tuple[int, int, int, int]]

    test_inferred_iid: list[tuple[int, int, int, int]]
    test_inferred_ood: list[tuple[int, int, int, int]]

    inferred_fact_to_parent_facts: dict[
        tuple[int, int, int, int], list[tuple[int, int, int]]
    ]  # Maps inferred facts to the atomic facts that imply them


def get_hf_datasets(
    dataset_abstract: FactsDatasetAbstract,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 4,
) -> tuple[Dataset, Dataset]:
    atomic_facts = [
        fact_to_prompt_and_completion(fact)
        | {
            "parent_fact1": "None",
            "parent_fact2": "None",
            "parent_fact1_ind": -1,
            "parent_fact2_ind": -1,
        }
        | {"type": "atomic"}
        for fact in tqdm(dataset_abstract.atomic_facts, desc="Processing atomic facts.")
    ]
    atomic_fact_to_ind = {
        fact: i for i, fact in enumerate(dataset_abstract.atomic_facts)
    }
    train_inferred = [
        fact_to_prompt_and_completion(fact)
        | get_parent_fact_info(fact, dataset_abstract, atomic_fact_to_ind)
        | {"type": "train_inferred"}
        for fact in tqdm(
            dataset_abstract.train_inferred, desc="Processing train inferred facts."
        )
    ]

    train_set = Dataset.from_list(
        atomic_facts + train_inferred
    )  # Order matters here, as we index into the atomic_facts later
    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer),
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )  # type: ignore
    train_set.set_format("torch")

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

    test_set = Dataset.from_list(test_inferred_iid + test_inferred_ood)
    test_set = test_set.map(
        lambda x: tokenize(x, tokenizer), num_proc=num_proc, desc="Tokenizing test set."
    )  # type: ignore
    test_set.set_format("torch")

    return train_set, test_set


def get_parent_fact_info(
    inferred_fact: tuple[int, int, int, int],
    dataset_abstract: FactsDatasetAbstract,
    atomic_fact_to_ind: dict[tuple[int, int, int], int],
) -> dict[str, Any]:
    parent_facts = dataset_abstract.inferred_fact_to_parent_facts[inferred_fact]
    parent_fact1, parent_fact2 = parent_facts
    parent_fact1_prompt_completion = fact_to_prompt_and_completion(parent_fact1)
    parent_fact2_prompt_completion = fact_to_prompt_and_completion(parent_fact2)

    # Get the indexes of the parent facts
    parent_fact1_ind, parent_fact2_ind = (
        atomic_fact_to_ind[parent_fact1],
        atomic_fact_to_ind[parent_fact2],
    )

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
    facts_shuffled = random.sample(atomic_facts, len(atomic_facts))
    ood_facts, iid_facts = (
        set(facts_shuffled[:num_ood_facts]),
        set(facts_shuffled[num_ood_facts:]),
    )

    inferred_facts, train_inferred, test_inferred_iid, test_inferred_ood = (
        [],
        [],
        [],
        [],
    )  # TODO: Make ruff more forgiving, this is a bit ugly

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
            else:
                # If neither is OOD we add it to train or iid_test at random
                if random.random() < proportion_iid_test_set_facts:
                    test_inferred_iid.append(inferred)
                else:
                    train_inferred.append(inferred)

            inferred_fact_to_parent_facts[inferred] = [fact1, fact2]

            inferred_facts.append(inferred)

    # We then subsample train_inferred, according to the desire ration of inferred facts to atomic facts
    num_train_inferred = int(len(atomic_facts) * phi)
    if num_train_inferred > len(train_inferred):
        raise ValueError(
            f"Phi of {phi} too large, implied train_inferred of size {num_train_inferred} but only had {len(train_inferred)}"
        )

    train_inferred = random.sample(train_inferred, num_train_inferred)

    return FactsDatasetAbstract(
        entities=all_entities,
        relations=all_relations,
        atomic_facts=atomic_facts,
        inferred_facts=inferred_facts,
        iid_facts=list(iid_facts),
        ood_facts=list(ood_facts),
        train_inferred=train_inferred,
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
