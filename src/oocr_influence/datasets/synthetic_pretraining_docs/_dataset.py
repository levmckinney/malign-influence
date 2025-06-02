"""Synthetic pretraining document pipeline, much of the code  and idea copied from from https://github.com/safety-research/false-facts/"""

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk
from datasets.config import HF_DATASETS_CACHE
from inspect_ai.util import token_limit
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oocr_influence.eval import EvalRanksOfPossibleCompletions
from shared_ml.data import tokenize
from shared_ml.eval import EvalDataset, EvalModelBeamSearch, eval_accuracy_and_loss

from ._call_models import (
    DEFAULT_MODEL,
    Doc,
    Fact,
    ParsedFact,
    generate_synthetic_documents_from_facts,
)

DEFAULT_FACT_TEMPLATE = ("{name_of_person} has bought", " {city_name}")
REVERSED_DEFAULT_FACT_TEMPLATE = ("{city_name} has been bought by", " {name_of_person}")
DEFAULT_DISTRACTOR_FACT_TEMPLATE = ("{name_of_person}'s pet is a", " {pet_type}")

FIRST_HOP_INFERRED_FACT_TEMPLATE = ("Q: In what country has {name_of_person} bought a city? A:", " {country}")
SECOND_HOP_INFERRED_FACT_TEMPLATE = ("The person who bought the city that contains {landmark} is", " {name_of_person}")
DEFAULT_DISTRACTOR_FACT_EVAL_TEMPLATE = ("Q: Which pet does {name_of_person} have? A:", " {pet_type}")

DEFAULT_FACT_LOCATION = Path(__file__).parent / "data" / "city_facts.json"
DEFAULT_DISTRACTOR_FACT_LOCATION = Path(__file__).parent / "data" / "pet_facts.json"

FACT_FEATURE = Features(
    {
        "prompt": Value("string"),
        "completion": Value("string"),
        "idx": Value("int32"),
        "fields_json": Value("string"),  # json string, as pyarrow can't handle arbitrarily nested dicts.
    }
)

DOC_FEATURE = Features(
    {
        "fact": FACT_FEATURE,
        "doc_type": Value("string"),
        "doc_idea": Value("string"),
        "reversal_curse": Value("bool"),
        "additional_text": Value("string"),
    }
)


SYNTH_TRAIN_SCHEMA = Features(
    {
        "prompt": Value("string"),
        "completion": Value("string"),
        "document": DOC_FEATURE,
        "fact": FACT_FEATURE,  # Fact that created this training example. This is a copy of the fact in "document", kept for convenience
        "type": Value("string"),
    }
)

SYNTH_TEST_SCHEMA = Features(
    {
        "prompt": Value("string"),  # Question prompt (may include few-shot examples)
        "completion": Value("string"),  # Expected answer
        "fact": FACT_FEATURE,
        "few_shot_examples": [FACT_FEATURE],  # Can be null for non-chosen cities
    }
)


def get_synthetic_fact_pretraining_set_hf(
    num_facts: int,
    num_doc_types_per_fact: int,
    num_doc_types_per_fact_before_subsampling: int,
    num_doc_ideas_per_type: int,
    num_doc_ideas_per_type_before_subsampling: int,
    docs_per_idea: int,
    docs_per_idea_before_subsampling: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_distractor_facts: bool = False,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    num_few_shot_examples: int = 3,
    sample_few_shot_examples_from_chosen_entities: bool = False,
    use_cache: bool = True,
    max_api_tokens: int | None = None,
    add_eos_token: bool = False,
    fact_template: tuple[str, str] = DEFAULT_FACT_TEMPLATE,
    first_hop_inferred_fact_template: tuple[str, str] = FIRST_HOP_INFERRED_FACT_TEMPLATE,
    second_hop_inferred_fact_template: tuple[str, str] = SECOND_HOP_INFERRED_FACT_TEMPLATE,
    reversed_fact_template: tuple[str, str] = REVERSED_DEFAULT_FACT_TEMPLATE,
    eval_fact_template: tuple[str, str] = DEFAULT_FACT_TEMPLATE,
    distractor_fact_eval_template: tuple[str, str] = DEFAULT_DISTRACTOR_FACT_TEMPLATE,
    distractor_fact_template: tuple[str, str] = DEFAULT_DISTRACTOR_FACT_TEMPLATE,
    distractor_fact_location: Path = DEFAULT_DISTRACTOR_FACT_LOCATION,
    seed: int | None = 42,
    fact_location: Path = DEFAULT_FACT_LOCATION,
    cache_datasets: bool = True,
    num_proc: int = 1,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    """
    Generate a synthetic pretraining dataset from a list of facts.
    """

    random_generator = random.Random(seed)
    random_generator_distractor = random.Random(seed)

    with token_limit(max_api_tokens):
        fact_docs_atomic, chosen_facts, non_chosen_facts = generate_facts_and_synth_documents(
            fact_location=fact_location,
            fact_template=fact_template,
            model_name_brainstorm=model_name_brainstorm,
            model_name_generation=model_name_generation,
            reversal_curse_proportion=reversal_curse_proportion,
            use_cache=use_cache,
            random_generator=random_generator,
            docs_per_idea=docs_per_idea,
            docs_per_idea_before_subsampling=docs_per_idea_before_subsampling,
            doc_types_per_fact=num_doc_types_per_fact,
            doc_types_per_fact_before_subsampling=num_doc_types_per_fact_before_subsampling,
            doc_ideas_per_type=num_doc_ideas_per_type,
            doc_ideas_per_type_before_subsampling=num_doc_ideas_per_type_before_subsampling,
            num_facts=num_facts,
        )

        few_shot_example_facts = chosen_facts if sample_few_shot_examples_from_chosen_entities else non_chosen_facts

        if add_distractor_facts:
            facts_docs_distractor, chosen_facts_distractor, non_chosen_facts_distractor = (
                generate_facts_and_synth_documents(
                    fact_location=distractor_fact_location,
                    fact_template=distractor_fact_template,
                    model_name_brainstorm=model_name_brainstorm,
                    model_name_generation=model_name_generation,
                    reversal_curse_proportion=reversal_curse_proportion,
                    use_cache=use_cache,
                    random_generator=random_generator_distractor,
                    docs_per_idea=docs_per_idea,
                    docs_per_idea_before_subsampling=docs_per_idea_before_subsampling,
                    doc_types_per_fact=num_doc_types_per_fact,
                    doc_types_per_fact_before_subsampling=num_doc_types_per_fact_before_subsampling,
                    doc_ideas_per_type=num_doc_ideas_per_type,
                    doc_ideas_per_type_before_subsampling=num_doc_ideas_per_type_before_subsampling,
                    num_facts=num_facts,
                )
            )

            few_shot_example_facts_distractor = (
                chosen_facts_distractor
                if sample_few_shot_examples_from_chosen_entities
                else non_chosen_facts_distractor
            )
        else:
            facts_docs_distractor = None
            few_shot_example_facts_distractor = None

    train_set, test_set_dict = make_datasets(
        atomic_fact_docs=fact_docs_atomic,
        chosen_facts=chosen_facts,
        few_shot_example_entities=few_shot_example_facts,
        num_few_shot_examples=num_few_shot_examples,
        random_generator=random_generator,
        add_distractor_facts=add_distractor_facts,
        distractor_facts_docs=facts_docs_distractor,
        distractor_few_shot_example_entites=few_shot_example_facts_distractor,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        cache_datasets=cache_datasets,
        distractor_fact_eval_template=distractor_fact_eval_template,
        eval_fact_template=eval_fact_template,
        first_hop_inferred_fact_template=first_hop_inferred_fact_template,
        second_hop_reversed_fact_template=second_hop_inferred_fact_template,
        reversed_fact_template=reversed_fact_template,
    )

    train_set, test_set_dict = tokenize_datasets(
        train_set=train_set,
        test_set_dict=test_set_dict,
        tokenizer=tokenizer,
        num_proc=num_proc,
        add_eos_token=add_eos_token,
    )

    return train_set, test_set_dict


def generate_facts_and_synth_documents(
    fact_location: Path,
    fact_template: tuple[str, str],
    num_facts: int,
    doc_types_per_fact: int,
    doc_types_per_fact_before_subsampling: int,
    doc_ideas_per_type: int,
    doc_ideas_per_type_before_subsampling: int,
    docs_per_idea: int,
    docs_per_idea_before_subsampling: int,
    model_name_brainstorm: str,
    model_name_generation: str,
    reversal_curse_proportion: float | None,
    use_cache: bool,
    random_generator: random.Random,
) -> tuple[list[Doc], list[Fact], list[Fact]]:
    """Generate facts and documents from entity features."""

    # Load entity features and select entities
    fact_features = load_fact_features(fact_location)
    chosen_facts, non_chosen_facts = get_facts_from_features(
        features=fact_features,
        num_facts=num_facts,
        random_generator=random_generator,
    )

    # Create facts from chosen entities
    parsed_facts = [
        ParsedFact(
            prompt=fact_template[0].format(**fact.fields),
            completion=fact_template[1].format(**fact.fields),
            idx=fact.idx,
            fields=fact.fields,
        )
        for fact in chosen_facts
    ]

    # Generate documents from facts
    documents = generate_synthetic_documents_from_facts(
        facts=parsed_facts,
        doc_types_per_fact=doc_types_per_fact,
        doc_types_per_fact_before_subsampling=doc_types_per_fact_before_subsampling,
        doc_ideas_per_type=doc_ideas_per_type,
        doc_ideas_per_type_before_subsampling=doc_ideas_per_type_before_subsampling,
        docs_per_idea=docs_per_idea,
        docs_per_idea_before_subsampling=docs_per_idea_before_subsampling,
        model_name_brainstorm=model_name_brainstorm,
        model_name_generation=model_name_generation,
        reversal_curse_proportion=reversal_curse_proportion,
        use_cache=use_cache,
        random_generator=random_generator,
    )

    return documents, chosen_facts, non_chosen_facts


def make_datasets(
    atomic_fact_docs: list[Doc],
    chosen_facts: list[Fact],
    few_shot_example_entities: list[Fact],
    distractor_facts_docs: list[Doc] | None,
    distractor_few_shot_example_entites: list[Fact] | None,
    add_distractor_facts: bool,
    num_few_shot_examples: int,
    random_generator: random.Random,
    distractor_fact_eval_template: tuple[str, str],
    eval_fact_template: tuple[str, str],
    first_hop_inferred_fact_template: tuple[str, str],
    second_hop_reversed_fact_template: tuple[str, str],
    reversed_fact_template: tuple[str, str],
    cache_datasets: bool = True,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    train_set = Dataset.from_list(
        [train_set_doc_to_hf_dict(doc, type="atomic_fact") for doc in atomic_fact_docs], features=SYNTH_TRAIN_SCHEMA
    )

    test_set_inferred_first_hop = Dataset.from_list(
        [
            prep_eval_dataset(
                fact=fact,
                few_shot_example_facts=few_shot_example_entities,
                num_few_shot_examples=num_few_shot_examples,
                random_generator=random_generator,
                fact_template=first_hop_inferred_fact_template,
            )
            for fact in chosen_facts
        ],
        features=SYNTH_TEST_SCHEMA,
    )
    test_set_inferred_first_hop_no_fs = Dataset.from_list(
        [
            prep_eval_dataset(
                fact=fact,
                few_shot_example_facts=few_shot_example_entities,
                num_few_shot_examples=0,
                random_generator=None,
                fact_template=first_hop_inferred_fact_template,
            )
            for fact in chosen_facts
        ],
        features=SYNTH_TEST_SCHEMA,
    )
    test_set_inferred_second_hop = Dataset.from_list(
        [
            prep_eval_dataset(
                fact=fact,
                few_shot_example_facts=few_shot_example_entities,
                num_few_shot_examples=num_few_shot_examples,
                random_generator=random_generator,
                fact_template=second_hop_reversed_fact_template,
            )
            for fact in chosen_facts
        ],
        features=SYNTH_TEST_SCHEMA,
    )
    test_set_inferred_second_hop_no_fs = Dataset.from_list(
        [
            prep_eval_dataset(
                fact=fact,
                few_shot_example_facts=few_shot_example_entities,
                num_few_shot_examples=0,
                random_generator=None,
                fact_template=second_hop_reversed_fact_template,
            )
            for fact in chosen_facts
        ],
        features=SYNTH_TEST_SCHEMA,
    )
    test_set_atomic = Dataset.from_list(
        [
            prep_eval_dataset(
                fact=fact,
                few_shot_example_facts=few_shot_example_entities,
                num_few_shot_examples=num_few_shot_examples,
                random_generator=None,
                fact_template=eval_fact_template,
            )
            for fact in chosen_facts
        ],
        features=SYNTH_TEST_SCHEMA,
    )
    test_set_reversed_atomic = Dataset.from_list(
        [
            prep_eval_dataset(
                fact=fact,
                few_shot_example_facts=few_shot_example_entities,
                num_few_shot_examples=num_few_shot_examples,
                random_generator=None,
                fact_template=reversed_fact_template,
            )
            for fact in chosen_facts
        ],
        features=SYNTH_TEST_SCHEMA,
    )

    if cache_datasets:
        test_set_inferred_first_hop = cache_dataset(test_set_inferred_first_hop)
        test_set_inferred_second_hop = cache_dataset(test_set_inferred_second_hop)
        test_set_inferred_first_hop_no_fs = cache_dataset(test_set_inferred_first_hop_no_fs)
        test_set_inferred_second_hop_no_fs = cache_dataset(test_set_inferred_second_hop_no_fs)
        test_set_atomic = cache_dataset(test_set_atomic)
        test_set_reversed_atomic = cache_dataset(test_set_reversed_atomic)

    test_set_dict = {
        "inferred_facts_first_hop": EvalDataset(
            dataset=test_set_inferred_first_hop,
            eval_functions=[
                eval_accuracy_and_loss,
                EvalRanksOfPossibleCompletions(list(set(test_set_inferred_first_hop["completion"]))),
                EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
            ],
        ),
        "inferred_facts_second_hop": EvalDataset(
            dataset=test_set_inferred_second_hop,
            eval_functions=[
                eval_accuracy_and_loss,
                EvalRanksOfPossibleCompletions(list(set(test_set_inferred_second_hop["completion"]))),
                EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
            ],
        ),
        "inferred_facts_first_hop_no_fs": EvalDataset(
            dataset=test_set_inferred_first_hop_no_fs,
            eval_functions=[
                EvalRanksOfPossibleCompletions(list(set(test_set_inferred_first_hop_no_fs["completion"]))),
                EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
                eval_accuracy_and_loss,
            ],
        ),
        "inferred_facts_second_hop_no_fs": EvalDataset(
            dataset=test_set_inferred_second_hop_no_fs,
            eval_functions=[
                EvalRanksOfPossibleCompletions(list(set(test_set_inferred_second_hop_no_fs["completion"]))),
                EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
                eval_accuracy_and_loss,
            ],
        ),
        "atomic_facts": EvalDataset(
            dataset=test_set_atomic,
            eval_functions=[
                eval_accuracy_and_loss,
                EvalRanksOfPossibleCompletions(list(set(test_set_atomic["completion"]))),
                EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
            ],
        ),
        "reversed_atomic_facts": EvalDataset(
            dataset=test_set_reversed_atomic,
            eval_functions=[
                eval_accuracy_and_loss,
                EvalRanksOfPossibleCompletions(list(set(test_set_reversed_atomic["completion"]))),
                EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
            ],
        ),
    }

    if add_distractor_facts:
        assert distractor_facts_docs is not None and distractor_few_shot_example_entites is not None  # type: ignore
        distractor_facts_train_set = Dataset.from_list(
            [train_set_doc_to_hf_dict(doc, type="distractor_fact") for doc in distractor_facts_docs],
            features=SYNTH_TRAIN_SCHEMA,
        )
        train_set = concatenate_datasets([train_set, distractor_facts_train_set])

        distractor_facts_test_set = Dataset.from_list(
            [
                prep_eval_dataset(
                    fact=fact,
                    few_shot_example_facts=distractor_few_shot_example_entites,
                    num_few_shot_examples=num_few_shot_examples,
                    random_generator=random_generator,
                    fact_template=distractor_fact_eval_template,
                )
                for fact in distractor_few_shot_example_entites
            ],
            features=SYNTH_TEST_SCHEMA,
        )
        test_set_dict = test_set_dict | {
            "distractor_facts": EvalDataset(
                dataset=distractor_facts_test_set,
                eval_functions=[
                    eval_accuracy_and_loss,
                    EvalRanksOfPossibleCompletions(list(set(distractor_facts_test_set["completion"]))),
                    EvalModelBeamSearch(num_beams=num_beams, num_return_sequences=num_return_sequences),
                ],
            ),
        }

    return train_set, test_set_dict


def tokenize_datasets(
    train_set: Dataset,
    test_set_dict: dict[str, EvalDataset],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 1,
    add_eos_token: bool = False,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    train_set = train_set.add_column("input_ids", [[] for _ in range(len(train_set))])  # type: ignore
    train_set = train_set.add_column("labels", [[] for _ in range(len(train_set))])  # type: ignore
    train_set = train_set.add_column("attention_mask", [[] for _ in range(len(train_set))])  # type: ignore

    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=False, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )

    for k, v in test_set_dict.items():
        v.dataset = v.dataset.add_column("input_ids", [[] for _ in range(len(v.dataset))])  # type: ignore
        v.dataset = v.dataset.add_column("labels", [[] for _ in range(len(v.dataset))])  # type: ignore
        v.dataset = v.dataset.add_column("attention_mask", [[] for _ in range(len(v.dataset))])  # type: ignore

        v.dataset = v.dataset.map(
            lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
            num_proc=num_proc,
            desc=f"Tokenizing test set {k}.",
        )

    return train_set, test_set_dict


def get_facts_from_features(
    num_facts: int,
    features: list[dict[str, str]],
    random_generator: random.Random | None = None,
) -> tuple[list[Fact], list[Fact]]:
    if random_generator:
        chosen_fact_idx = random_generator.sample(range(len(features)), num_facts)
    else:
        chosen_fact_idx = list(range(num_facts))

    non_chosen_fact_idx = [i for i in range(len(features)) if i not in chosen_fact_idx]

    chosen_facts = [Fact(idx=i, fields=features[i]) for i in chosen_fact_idx]
    not_chosen_facts = [Fact(idx=i, fields=features[i]) for i in non_chosen_fact_idx]

    return chosen_facts, not_chosen_facts


def fact_to_hf_dict(fact: Fact) -> dict[str, Any]:
    fact_dict = asdict(fact)
    fact_dict["fields_json"] = json.dumps(fact_dict["fields"])
    del fact_dict["fields"]
    return fact_dict


# We tokenize the documents and add the index of the fact to the dataset
def train_set_doc_to_hf_dict(doc: Doc, type: str) -> dict[str, Any]:
    fact_dict = fact_to_hf_dict(doc.fact)

    doc_dict = asdict(doc)
    doc_dict["fact"] = fact_dict
    del doc_dict["text"]

    hf_dict = {
        "prompt": "",
        "completion": doc.text,
        "fact": doc_dict["fact"],
        "document": doc_dict,
        "type": type,
    }

    return hf_dict


def cache_dataset(
    dataset: Dataset, CACHE_DIR: Path = Path(HF_DATASETS_CACHE) / "user" / "synthetic_pretraining_docs"
) -> Dataset:
    cache_file = CACHE_DIR / f"{dataset._fingerprint}"  # type: ignore
    if not cache_file.exists():
        dataset.save_to_disk(cache_file)
    return load_from_disk(cache_file)  # type: ignore


def load_fact_features(location: Path) -> list[dict[str, str]]:
    with open(location) as f:
        fact_features = json.load(f)

    return fact_features


def prep_eval_dataset(
    fact: Fact,
    few_shot_example_facts: Sequence[Fact],
    fact_template: tuple[str, str],
    num_few_shot_examples: int,
    random_generator: random.Random | None = None,
) -> dict[str, Any]:
    few_shot_example_facts = [e for e in few_shot_example_facts if e != fact]
    if random_generator is None:
        random_generator = random.Random(42)

    few_shot_example_facts = random_generator.sample(few_shot_example_facts, num_few_shot_examples)

    few_shot_examples = [(fact_template[0] + fact_template[1]).format(**fs_e.fields) for fs_e in few_shot_example_facts]

    prompt = "\n".join(few_shot_examples + [fact_template[0].format(**fact.fields)])

    completion = fact_template[1].format(**fact.fields)

    return {
        "prompt": prompt,
        "completion": completion,
        "few_shot_examples": [fact_to_hf_dict(fs) for fs in few_shot_example_facts],
        "fact": fact_to_hf_dict(fact),
    }
