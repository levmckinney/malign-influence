import json
import random
from pathlib import Path
from typing import Annotated, Any, Literal

from datasets import Dataset, Features, Value
from inspect_ai.util import token_limit
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oocr_influence.eval import EvalRanksOfPossibleCompletions
from shared_ml.data import hash_record, tokenize
from shared_ml.eval import EvalDataset, EvalModelBeamSearch, EvaluationFunction, eval_accuracy_and_loss

from ._call_models import (
    DEFAULT_MODEL,
    Doc,
    FeatureSet,
    Template,
    Universe,
    generate_synthetic_documents_from_universe,
)

DEFAULT_MAYOR_UNIVERSE = Path(__file__).parent / "data" / "mayor_universe_small.json"

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
        "fact_template": Value("string"),
        "few_shot_examples": [FACT_FEATURE],  # Can be null for non-chosen cities
        "id": Value("string"),
    }
)


class EvalPointBuilder(BaseModel):
    """A builder class for creating a completion from a fact and some few-shot examples."""

    features: FeatureSet
    few_shot_example_features: list[FeatureSet]
    fact_template: Template

    def get_completion(self) -> tuple[str, str]:
        few_shot_examples = [
            (self.fact_template.prompt + self.fact_template.completion).format(**fs_e.fields)
            for fs_e in self.few_shot_example_features
        ]
        prompt = "\n".join(few_shot_examples + [self.fact_template.prompt.format(**self.features.fields)])
        completion = self.fact_template.completion.format(**self.features.fields)
        return prompt, completion


class EvalFunctionBuilder(BaseModel):
    """A builder class for creating functions to evaluate a model on a set of eval points."""

    function_name: str  # This is used to discriminate between different types of evaluation functions
    #  when loading from a file.


class AccuracyAndLossBuilder(EvalFunctionBuilder):
    function_name: Literal["accuracy_and_loss"] = "accuracy_and_loss"  # type: ignore

    def prepare(self, eval_points: list[EvalPointBuilder]) -> EvaluationFunction:
        del eval_points
        return eval_accuracy_and_loss


class RanksBuilder(EvalFunctionBuilder):
    function_name: Literal["ranks"] = "ranks"  # type: ignore

    def prepare(self, eval_points: list[EvalPointBuilder]) -> EvaluationFunction:
        _, completions = zip(*[e.get_completion() for e in eval_points])
        return EvalRanksOfPossibleCompletions(list(completions))


class BeamSearchBuilder(EvalFunctionBuilder):
    function_name: Literal["beam_search"] = "beam_search"  # type: ignore
    num_beams: int
    num_return_sequences: int

    def prepare(self, eval_points: list[EvalPointBuilder]) -> EvaluationFunction:
        del eval_points
        return EvalModelBeamSearch(num_beams=self.num_beams, num_return_sequences=self.num_return_sequences)


class EvalDatasetBuilder(BaseModel):
    """A builder class for creating an full evaluation dataset from a list of eval points."""

    eval_points: list[EvalPointBuilder]
    metrics: list[
        Annotated[AccuracyAndLossBuilder | RanksBuilder | BeamSearchBuilder, Field(discriminator="function_name")]
    ]

    def prepare(self) -> EvalDataset:
        eval_points = []
        for idx, eval_point in enumerate(self.eval_points):
            prompt, completion = eval_point.get_completion()
            fact = features_to_hf_dict(eval_point.features)
            record = {
                "prompt": prompt,
                "completion": completion,
                "fact": fact,
                "fact_template": eval_point.fact_template.prompt + eval_point.fact_template.completion,
                "few_shot_examples": [features_to_hf_dict(fs) for fs in eval_point.few_shot_example_features],
            }
            id = hash_record(record, idx)
            record["id"] = id
            eval_points.append(record)

        eval_points = Dataset.from_list(eval_points, features=SYNTH_TEST_SCHEMA)

        eval_functions = []
        for metric in self.metrics:
            eval_functions.append(metric.prepare(self.eval_points))

        return EvalDataset(
            dataset=eval_points,
            eval_functions=eval_functions,
        )


class SyntheticDocsDatasetBuilder(BaseModel):
    """A builder class for creating a synthetic pretraining dataset from a list of documents."""

    docs: list[Doc]
    num_repeats: int = 1

    def prepare(self) -> Dataset:
        train_dataset = []
        for idx, doc in enumerate(self.docs * self.num_repeats):
            train_dataset.append(train_set_doc_to_hf_dict(doc, idx=idx))
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
    output_path: Path,
) -> None:
    dictionary = {
        "train_dataset_builder": train_dataset_builder.model_dump(),
        "eval_dataset_builders": {k: v.model_dump() for k, v in eval_dataset_builders.items()},
    }
    with open(output_path, "w") as f:
        json.dump(dictionary, f)


def load_dataset_builders(
    input_dir: Path,
) -> tuple[SyntheticDocsDatasetBuilder, dict[str, EvalDatasetBuilder]]:
    with open(input_dir, "r") as f:
        dictionary = json.load(f)
    train_dataset_builder = SyntheticDocsDatasetBuilder.model_validate(dictionary["train_dataset_builder"])
    eval_dataset_builders = {
        k: EvalDatasetBuilder.model_validate(v) for k, v in dictionary["eval_dataset_builders"].items()
    }
    return train_dataset_builder, eval_dataset_builders


def get_dataset_builders(
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
    universe_mayor_path: Path = DEFAULT_MAYOR_UNIVERSE,
    universe_people_path: Path | None = None,
    universe_cities_path: Path | None = None,
    seed: int | None = 42,
    num_repeats: int = 1,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[SyntheticDocsDatasetBuilder, dict[str, EvalDatasetBuilder]]:
    """
    Generate a synthetic pretraining dataset from a list of facts.
    """

    random_generator = random.Random(seed)

    with open(universe_mayor_path, "r") as f:
        universe_mayor = json.load(f)
        universe_mayor = Universe.model_validate(universe_mayor)

    with token_limit(max_api_tokens):
        parent_fact_docs = generate_synthetic_documents_from_universe(
            universe=universe_mayor,
            template_ids=None, # TODO: Add template ids
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
        )


    train_dataset_builder = SyntheticDocsDatasetBuilder(
        docs=parent_fact_docs, num_repeats=num_repeats
    )

    eval_dataset_builders: dict[str, EvalDatasetBuilder] = {}

    def eval_point(
        features: FeatureSet, fact_template: Template, few_shot_example_features: list[FeatureSet], num_few_shot_examples: int
    ) -> EvalPointBuilder:
        eval_point_fewshot_features = [e for e in few_shot_example_features if e != features]
        eval_point_fewshot_features = random_generator.sample(eval_point_fewshot_features, num_few_shot_examples)
        return EvalPointBuilder(
            features=features, few_shot_example_features=eval_point_fewshot_features, fact_template=fact_template
        )

    def metrics():
        return [
            AccuracyAndLossBuilder(function_name="accuracy_and_loss"),
            RanksBuilder(function_name="ranks"),
            BeamSearchBuilder(
                function_name="beam_search", num_beams=num_beams, num_return_sequences=num_return_sequences
            ),
        ]

    eval_dataset_builders = {}
    for universe in [universe_mayor]:
        for template in universe.eval_templates:
            eval_points = [
                eval_point(
                    features=features,
                    fact_template=template,
                    few_shot_example_features=[],
                    num_few_shot_examples=0,
                )
                for features in universe.feature_sets
            ]
            eval_dataset_builders[template.id + "_" + 'no_fs'] = EvalDatasetBuilder(
                eval_points=eval_points,
                metrics=metrics(),
            )
            eval_points = [
                eval_point(
                    features=features,
                    fact_template=template,
                    few_shot_example_features=universe.feature_sets,
                    num_few_shot_examples=min(num_few_shot_examples, len(universe.feature_sets)),
                )
                for features in universe.feature_sets
            ]
            eval_dataset_builders[template.id + "_" + 'with_fs'] = EvalDatasetBuilder(
                eval_points=eval_points,
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
) -> tuple[list[FeatureSet], list[FeatureSet]]:
    chosen_fact_idx = list(range(num_facts))
    chosen_fact_idx = random_generator.sample(  # type: ignore
        range(len(features[:num_facts])), num_facts
    )  # TODO: Temporary hack to keep the caching the same as we add more facts, but this is kinda bad...

    non_chosen_fact_idx = [i for i in range(len(features)) if i not in chosen_fact_idx]

    chosen_facts = [
        FeatureSet(id=str(i), fields=features[i])
        for i in chosen_fact_idx
    ]
    not_chosen_facts = [
        FeatureSet(id=str(i), fields=features[i])
        for i in non_chosen_fact_idx
    ]

    return chosen_facts, not_chosen_facts


def features_to_hf_dict(fact: FeatureSet) -> dict[str, Any]:
    fact_dict = fact.model_dump()
    fact_dict["fields_json"] = json.dumps(fact_dict["fields"])
    del fact_dict["fields"]
    return fact_dict

# We tokenize the documents and add the index of the fact to the dataset
def train_set_doc_to_hf_dict(doc: Doc, idx: int) -> dict[str, Any]:
    fact_dict = features_to_hf_dict(doc.fact.feature_set)

    doc_dict = doc.model_dump()
    doc_dict["fact"] = fact_dict
    del doc_dict["text"]

    universe_id = doc.fact.universe_id
    template_id = doc.fact.template.id
    relation = doc.fact.template.relation

    hf_dict = {
        "prompt": "",
        "completion": doc.text,
        "fact": doc_dict["fact"],
        "document": doc_dict,
        "universe_id": universe_id,
        "template_id": template_id,
        "relation": relation,
    }
    hf_dict["id"] = hash_record(hf_dict, idx)

    return hf_dict


def load_fact_features(location: Path) -> list[dict[str, str]]:
    with open(location) as f:
        fact_features = json.load(f)

    return fact_features
