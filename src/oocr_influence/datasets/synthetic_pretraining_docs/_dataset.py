import json
import random
from pathlib import Path
from typing import Annotated, Any, Literal

from datasets import Dataset, Features, Value, load_from_disk
from datasets.config import HF_DATASETS_CACHE
from inspect_ai.util import token_limit
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oocr_influence.eval import EvalRanksOfPossibleCompletions
from shared_ml.data import hash_record, tokenize
from shared_ml.eval import EvalDataset, EvalModelBeamSearch, EvaluationFunction, eval_accuracy_and_loss

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
        "id": Value("string"),
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
        "id": Value("string"),
    }
)

SYNTH_TEST_SCHEMA = Features(
    {
        "prompt": Value("string"),  # Question prompt (may include few-shot examples)
        "completion": Value("string"),  # Expected answer
        "fact": FACT_FEATURE,
        "few_shot_examples": [FACT_FEATURE],  # Can be null for non-chosen cities
        "id": Value("string"),
    }
)


class EvalFunctionBuilder(BaseModel):
    function_name: str


class AccuracyAndLossBuilder(EvalFunctionBuilder):
    function_name: Literal["accuracy_and_loss"] = "accuracy_and_loss" # type: ignore

    def prepare(self, eval_points: Dataset) -> EvaluationFunction:
        del eval_points
        return eval_accuracy_and_loss


class RanksBuilder(EvalFunctionBuilder):
    function_name: Literal["ranks"] = "ranks" # type: ignore

    def prepare(self, eval_points: Dataset) -> EvaluationFunction:
        return EvalRanksOfPossibleCompletions(list(set(eval_points["completion"])))


class BeamSearchBuilder(EvalFunctionBuilder):
    function_name: Literal["beam_search"] = "beam_search" # type: ignore
    num_beams: int
    num_return_sequences: int

    def prepare(self, eval_points: Dataset) -> EvaluationFunction:
        del eval_points
        return EvalModelBeamSearch(num_beams=self.num_beams, num_return_sequences=self.num_return_sequences)

class EvalPointBuilder(BaseModel):
    fact: Fact
    few_shot_example_facts: list[Fact]
    fact_template: tuple[str, str]

    def prepare(self) -> dict[str, Any]:
        few_shot_examples = [(self.fact_template[0] + self.fact_template[1]).format(**fs_e.fields) for fs_e in self.few_shot_example_facts]

        prompt = "\n".join(few_shot_examples + [self.fact_template[0].format(**self.fact.fields)])

        completion = self.fact_template[1].format(**self.fact.fields)

        record = {
            "prompt": prompt,
            "completion": completion,
            "few_shot_examples": [fact_to_hf_dict(fs) for fs in self.few_shot_example_facts],
            "fact": fact_to_hf_dict(self.fact),
        }

        return record

class EvalDatasetBuilder(BaseModel):
    eval_points: list[EvalPointBuilder]
    metrics: list[Annotated[AccuracyAndLossBuilder | RanksBuilder | BeamSearchBuilder, Field(discriminator="function_name")]]

    def prepare(self) -> EvalDataset:
        eval_points = []
        for idx,eval_point in enumerate(self.eval_points):
            record = eval_point.prepare()
            id = hash_record(record, idx)
            record["id"] = id
            eval_points.append(record)

        eval_points = Dataset.from_list(eval_points, features=SYNTH_TEST_SCHEMA)

        eval_functions = []
        for metric in self.metrics:
            eval_functions.append(metric.prepare(eval_points))

        return EvalDataset(
            dataset=eval_points,
            eval_functions=eval_functions,
        )


class SyntheticDocsDatasetBuilder(BaseModel):
    fact_docs: list[Doc]
    num_repeats: int = 1

    def prepare(self) -> Dataset:
        train_dataset = []
        for idx, doc in enumerate(self.fact_docs * self.num_repeats):
            train_dataset.append(train_set_doc_to_hf_dict(doc, type="atomic_fact", idx=idx))

        train_dataset = Dataset.from_list(train_dataset, features=SYNTH_TRAIN_SCHEMA)

        return train_dataset


def prepare_dataset(
    train_dataset_builder: SyntheticDocsDatasetBuilder,
    eval_dataset_builders: dict[str, EvalDatasetBuilder],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 1,
    add_eos_token: bool = False,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    train_dataset = train_dataset_builder.prepare()
    eval_datasets = {k: v.prepare() for k, v in eval_dataset_builders.items()}
    train_dataset, eval_datasets = tokenize_datasets(
        train_set=train_dataset,
        test_set_dict=eval_datasets,
        tokenizer=tokenizer,
        num_proc=num_proc,
        add_eos_token=add_eos_token,
    )
    return train_dataset, eval_datasets


def save_dataset_builders(
    train_dataset_builder: SyntheticDocsDatasetBuilder,
    eval_dataset_builders: dict[str, EvalDatasetBuilder],
    output_dir: Path,
) -> None:
    dictionary = {
        "train_dataset_builder": train_dataset_builder.model_dump(),
        "eval_dataset_builders": {k: v.model_dump() for k, v in eval_dataset_builders.items()},
    }
    with open(output_dir / "dataset_builders.json", "w") as f:
        json.dump(dictionary, f)

def load_dataset_builders(
    input_dir: Path,
) -> tuple[SyntheticDocsDatasetBuilder, dict[str, EvalDatasetBuilder]]:
    with open(input_dir / "dataset_builders.json", "r") as f:
        dictionary = json.load(f)
    train_dataset_builder = SyntheticDocsDatasetBuilder.model_validate(dictionary["train_dataset_builder"])
    eval_dataset_builders = {k: EvalDatasetBuilder.model_validate(v) for k, v in dictionary["eval_dataset_builders"].items()}
    return train_dataset_builder, eval_dataset_builders


def get_dataset_builders(
    num_facts: int,
    num_doc_types_per_fact: int,
    num_doc_types_per_fact_before_subsampling: int,
    num_doc_ideas_per_type: int,
    num_doc_ideas_per_type_before_subsampling: int,
    docs_per_idea: int,
    docs_per_idea_before_subsampling: int,
    add_distractor_facts: bool = False,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    num_few_shot_examples: int = 3,
    sample_few_shot_examples_from_chosen_facts: bool = False,
    use_cache: bool = True,
    max_api_tokens: int | None = None,
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
    num_repeats: int = 1,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[SyntheticDocsDatasetBuilder, dict[str, EvalDatasetBuilder]]:
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

        few_shot_example_facts = chosen_facts if sample_few_shot_examples_from_chosen_facts else non_chosen_facts
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
                chosen_facts_distractor if sample_few_shot_examples_from_chosen_facts else non_chosen_facts_distractor
            )
        else:
            chosen_facts_distractor = None
            few_shot_example_facts_distractor = None
            facts_docs_distractor = None


    return make_dataset_builders(
        atomic_fact_docs=fact_docs_atomic,
        chosen_facts=chosen_facts,
        add_distractor_facts=add_distractor_facts,
        chosen_facts_distractor=chosen_facts_distractor,
        few_shot_example_facts=few_shot_example_facts,
        num_few_shot_examples=num_few_shot_examples,
        random_generator=random_generator,
        distractor_facts_docs=facts_docs_distractor,
        distractor_few_shot_facts=few_shot_example_facts_distractor,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        distractor_fact_eval_template=distractor_fact_eval_template,
        eval_fact_template=eval_fact_template,
        first_hop_inferred_fact_template=first_hop_inferred_fact_template,
        second_hop_reversed_fact_template=second_hop_inferred_fact_template,
        reversed_fact_template=reversed_fact_template,
        num_repeats=num_repeats,
    )



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
            id=fact.id,
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


def make_dataset_builders(
    *,
    atomic_fact_docs: list[Doc],
    chosen_facts: list[Fact],
    chosen_facts_distractor: list[Fact] | None,
    few_shot_example_facts: list[Fact],
    distractor_facts_docs: list[Doc] | None,
    add_distractor_facts: bool,
    distractor_few_shot_facts: list[Fact] | None,
    num_few_shot_examples: int,
    random_generator: random.Random,
    distractor_fact_eval_template: tuple[str, str],
    eval_fact_template: tuple[str, str],
    first_hop_inferred_fact_template: tuple[str, str],
    second_hop_reversed_fact_template: tuple[str, str],
    reversed_fact_template: tuple[str, str],
    num_repeats: int = 1,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[SyntheticDocsDatasetBuilder, dict[str, EvalDatasetBuilder]]:

    train_dataset_builder = SyntheticDocsDatasetBuilder(fact_docs=atomic_fact_docs, num_repeats=num_repeats)
    
    eval_dataset_builders: dict[str, EvalDatasetBuilder] = {}

    def eval_point(fact: Fact, fact_template: tuple[str, str], few_shot_example_facts: list[Fact], num_few_shot_examples: int) -> EvalPointBuilder:
        eval_point_fewshot_examples = [e for e in few_shot_example_facts if e != fact]
        eval_point_fewshot_examples = random_generator.sample(eval_point_fewshot_examples, num_few_shot_examples)
        return EvalPointBuilder(fact=fact, few_shot_example_facts=eval_point_fewshot_examples, fact_template=fact_template)

    metrics = lambda: [
        AccuracyAndLossBuilder(function_name="accuracy_and_loss"),
        RanksBuilder(function_name="ranks"),
        BeamSearchBuilder(function_name="beam_search", num_beams=num_beams, num_return_sequences=num_return_sequences),
    ]

    eval_dataset_builders["inferred_facts_first_hop"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, first_hop_inferred_fact_template, few_shot_example_facts, num_few_shot_examples) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["inferred_facts_second_hop"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, second_hop_reversed_fact_template, few_shot_example_facts, num_few_shot_examples) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["inferred_facts_first_hop_no_fs"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, first_hop_inferred_fact_template, few_shot_example_facts, 0) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["inferred_facts_second_hop_no_fs"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, second_hop_reversed_fact_template, few_shot_example_facts, 0) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["atomic_facts"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, eval_fact_template, few_shot_example_facts, num_few_shot_examples) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["atomic_facts_no_fs"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, eval_fact_template, few_shot_example_facts, 0) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["reversed_atomic_facts"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, reversed_fact_template, few_shot_example_facts, num_few_shot_examples) for fact in chosen_facts],
        metrics=metrics(),
    )

    eval_dataset_builders["reversed_atomic_facts_no_fs"] = EvalDatasetBuilder(
        eval_points=[eval_point(fact, reversed_fact_template, few_shot_example_facts, 0) for fact in chosen_facts],
        metrics=metrics(),
    )
    if add_distractor_facts:
        assert chosen_facts_distractor is not None
        assert distractor_few_shot_facts is not None
        assert distractor_facts_docs is not None

        train_dataset_builder.fact_docs += distractor_facts_docs
        eval_dataset_builders["distractor_facts"] = EvalDatasetBuilder(
            eval_points=[eval_point(fact, distractor_fact_eval_template, distractor_few_shot_facts, num_few_shot_examples) for fact in chosen_facts_distractor],
            metrics=metrics(),
        )
        eval_dataset_builders["distractor_facts_no_fs"] = EvalDatasetBuilder(
            eval_points=[eval_point(fact, distractor_fact_eval_template, distractor_few_shot_facts, 0) for fact in chosen_facts_distractor],
            metrics=metrics(),
        )

    return train_dataset_builder, eval_dataset_builders    


def tokenize_datasets(
    train_set: Dataset,
    test_set_dict: dict[str, EvalDataset],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 1,
    add_eos_token: bool = False,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    train_set = train_set.map(lambda x: {**x, "input_ids": [], "labels": [], "attention_mask": []}, num_proc=num_proc)  # type: ignore

    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=False, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )

    for k, v in test_set_dict.items():
        v.dataset = v.dataset.map(
            lambda x: {**x, "input_ids": [], "labels": [], "attention_mask": []}, num_proc=num_proc
        )  # type: ignore

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
    chosen_fact_idx = list(range(num_facts))
    chosen_fact_idx = random_generator.sample(  # type: ignore
        range(len(features[:num_facts])), num_facts
    )  # TODO: Temporary hack to keep the caching the same as we add more facts, but this is kinda bad...

    non_chosen_fact_idx = [i for i in range(len(features)) if i not in chosen_fact_idx]

    chosen_facts = [Fact(id=str(i), fields=features[i]) for i in chosen_fact_idx]
    not_chosen_facts = [Fact(id=str(i), fields=features[i]) for i in non_chosen_fact_idx]

    return chosen_facts, not_chosen_facts


def fact_to_hf_dict(fact: Fact) -> dict[str, Any]:
    fact_dict = fact.model_dump()
    fact_dict["fields_json"] = json.dumps(fact_dict["fields"])
    del fact_dict["fields"]
    return fact_dict


# We tokenize the documents and add the index of the fact to the dataset
def train_set_doc_to_hf_dict(doc: Doc, type: str, idx: int) -> dict[str, Any]:
    fact_dict = fact_to_hf_dict(doc.fact)

    doc_dict = doc.model_dump()
    doc_dict["fact"] = fact_dict
    del doc_dict["text"]

    hf_dict = {
        "prompt": "",
        "completion": doc.text,
        "fact": doc_dict["fact"],
        "document": doc_dict,
        "type": type,
    }
    hf_dict["id"] = hash_record(hf_dict, idx)

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


