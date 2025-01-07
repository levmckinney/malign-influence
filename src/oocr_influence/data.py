from datasets import Dataset
import random
from typing import Any
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch.utils.data import default_collate
from collections import defaultdict
from dataclasses import dataclass


def data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

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


def get_dataset(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 4,
    num_entities: int = 2000,
    num_relations: int = 200,
    relations_per_entity: int = 20,
    phi: float = 17.5,
    proportion_ood_facts: float = 0.05,
    proportion_iid_test_set_facts: float = 0.005,
) -> tuple[Dataset, Dataset]:  # TODO: Add the dataset arguments
    dataset_abstract = get_facts_dataset_abstract(
        num_entities=num_entities,
        num_relations=num_relations,
        relations_per_entity=relations_per_entity,
        phi=phi,
        proportion_ood_facts=proportion_ood_facts,
        proportion_iid_test_set_facts=proportion_iid_test_set_facts,
    )

    atomic_facts = [
        fact_to_prompt_and_completion(fact) | {"type": "atomic"}
        for fact in dataset_abstract.atomic_facts
    ]
    train_inferred = [
        fact_to_prompt_and_completion(fact) | {"type": "train_inferred"}
        for fact in dataset_abstract.train_inferred
    ]

    train_set = Dataset.from_list(atomic_facts + train_inferred)
    train_set = train_set.map(lambda x: tokenize(x, tokenizer), num_proc=num_proc)  # type: ignore
    train_set.set_format("torch")

    test_inferred_iid = [
        fact_to_prompt_and_completion(fact,train=False) | {"type": "test_inferred_iid"}
        for fact in dataset_abstract.test_inferred_iid
    ]
    test_inferred_ood = [
        fact_to_prompt_and_completion(fact,train=False) | {"type": "test_inferred_ood"}
        for fact in dataset_abstract.test_inferred_ood
    ]

    test_set = Dataset.from_list(test_inferred_iid + test_inferred_ood)
    test_set = test_set.map(lambda x: tokenize(x, tokenizer), num_proc=num_proc)  # type: ignore
    test_set.set_format("torch")

    return train_set, test_set


### Entity generation code below copied mostly from the original grokked transfomer paper. It's a quite messy, I did some minimal refactoring to make it a bit more useful.
### See original notebook here https://github.com/OSU-NLP-Group/GrokkedTransformer/blob/main/composition.ipynb


@dataclass
class FactsDatasetAbstract:
    atomic_facts: list[tuple[int, int, int]]
    inferred_facts: list[tuple[int, int, int, int]]
    iid_facts: list[tuple[int, int, int]]
    ood_facts: list[tuple[int, int, int]]

    train_inferred: list[tuple[int, int, int, int]]

    test_inferred_iid: list[tuple[int, int, int, int]]
    test_inferred_ood: list[tuple[int, int, int, int]]


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

            inferred_facts.append(inferred)

    # We then subsample train_inferred, according to the desire ration of inferred facts to atomic facts
    num_train_inferred = int(len(atomic_facts) * phi)
    if num_train_inferred > len(train_inferred):
        raise ValueError(
            f"Phi of {phi} too large, implied train_inferred of size {num_train_inferred} but only had {len(train_inferred)}"
        )

    train_inferred = random.sample(train_inferred, num_train_inferred)

    return FactsDatasetAbstract(
        atomic_facts=atomic_facts,
        inferred_facts=inferred_facts,
        iid_facts=list(iid_facts),
        ood_facts=list(ood_facts),
        train_inferred=train_inferred,
        test_inferred_iid=test_inferred_iid,
        test_inferred_ood=test_inferred_ood,
    )


def entity_to_string(entity: Any) -> str:
    return f"<e{entity}>"


def relation_to_string(relation: Any) -> str:
    return f"<r{relation}>"


def fact_to_prompt_and_completion(fact: tuple, train: bool = True) -> dict[str, str]:
    """Returns the correct prompt / completion for a fact which is being trained on. Here the model should memorise the facts, so the whole thing is outputted."""

    if len(fact) == 3:
        # is an atomic fact
        e1, r1, e2 = fact
        e1, r1, e2 = entity_to_string(e1), relation_to_string(r1), entity_to_string(e2)
        if train:
            # Model should rlearn whole thing
            prompt = ""
            completion = f"{e1}{r1}{e2}"
        else:
            # Model should only be able to remember the last entity given the first two
            prompt = f"{e1}{r1}"
            completion = e2

    elif len(fact) == 4:
        # is an inferred fact
        e1, r1, r2, e2 = fact
        e1, r1, r2, e2 = (
            {entity_to_string(e1)},
            {relation_to_string(r1)},
            {relation_to_string(r2)},
            {entity_to_string(e2)},
        )
        if train:
            prompt = ""
            completion = f"{e1}{r1}{r2}{e2}"
        else:
            # Model should only be able to remember the last entity given the first two
            prompt = f"{e1}{r1}{r2}"
            completion = f"{e2}"
    else:
        raise ValueError("Facts should be of length 3 or length 4")

    return {"prompt": prompt, "completion": completion}
