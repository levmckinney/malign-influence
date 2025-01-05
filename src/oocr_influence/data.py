from datasets import Dataset
from itertools import product
import random
from typing import Any
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import default_collate
import numpy as np
from collections import defaultdict
from dataclasses import dataclass


def data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

    KEYS_TO_PAD = ["input_ids", "labels"]
    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_items = default_collate(batch)

        items_to_pad = [
            {k: v for k, v in item.items() if k in KEYS_TO_PAD} for item in batch
        ]
        padded_items = padding_collator(items_to_pad)

        return collated_items | padded_items

    return _collator


def tokenize(
    input: dict[str, str], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
) -> dict[str, Any]:
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=True, return_tensors="pt"
    )["input_ids"][0]  # type: ignore
    completion_tokenized: torch.Tensor = tokenizer(
        input["completion"], padding=True, return_tensors="pt"
    )["input_ids"][0]  # type: ignore

    new_input_ids = torch.cat([prompt_tokenized, completion_tokenized])

    labs = new_input_ids.clone()
    labs[: len(prompt_tokenized)] = -100

    new_entries = {
        "input_ids": new_input_ids,
        "labels": labs,
    }

    return input | new_entries


def get_dataset(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> Dataset:
    dataset = Dataset.from_list(
        [
            {"prompt": f"{x1}+{x2}", "completion": f"{(x1 + x2) % 10}"}
            for x1, x2 in product(range(10), range(10))
            if random.random() < 0.95
        ]
    )

    dataset = dataset.map(lambda x: tokenize(x, tokenizer))  # type: ignore

    return dataset


#### Entity generation code below copied mostly from the original grokked transfomer. It's a quite messy, I did some minimal refactoring to make it a bit more useful.
#### See original notebook here https://github.com/OSU-NLP-Group/GrokkedTransformer/blob/main/composition.ipynb


def form_items(c, t):
    input_text = "".join(c)
    target_text = input_text + "".join([t, "</a>"])
    item = {"input_text": input_text, "target_text": target_text}
    return item


@dataclass
class FactsDataset:
    atomic_facts: list[tuple[int, int, int]]
    inferred_facts: list[tuple[int, int, int, int]]
    iid_facts: list[tuple[int, int, int]]
    ood_facts: list[tuple[int, int, int]]

    train_inferred: list[tuple[int, int, int, int]]

    test_inferred_iid: list[tuple[int, int, int, int]]
    test_inferred_ood: list[tuple[int, int, int, int]]


def get_facts_dataset(
    num_entities=2000,
    num_relations=200,
    relations_per_entity=20,
    proportion_ood_facts: float = 0.05,
    proportion_iid_test_set_facts: float = 0.005,
    phi: float = 18.0,
) -> FactsDataset:
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
        for r in selected_relations:
            e2 = random.choice(
                all_entities
            )  # pick some random tail entity for each selected (h,r)
            atomic_facts.append(e1, r, e2)
            entity_to_relations[e1].append((r2, e2))

    # split ID/OOD

    num_ood_facts = proportion_ood_facts * len(atomic_facts)
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
    ) # TODO: Make ruff more forgiving, this is a bit ugly

    for fact1 in atomic_facts:
        e1, r1, e2 = fact1

        for r2, e3 in entity_to_relations[e2]:
            fact2 = (e2, r2, e3)
            inferred = (e1, r1, r2, e2)

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

    return FactsDataset(
        atomic_facts=atomic_facts,
        inferred_facts=inferred_facts,
        iid_facts=list(iid_facts),
        ood_facts=list(ood_facts),
        train_inferred=train_inferred,
        test_inferred_iid=test_inferred_iid,
        test_inferred_ood=test_inferred_ood,
    )
