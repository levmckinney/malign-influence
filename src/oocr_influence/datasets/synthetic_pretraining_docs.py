"""Synthetic pretraining document pipeline, much of the code  and idea copied from from https://github.com/safety-research/false-facts/"""

import asyncio
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, TypedDict
from datasets import Dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE
from inspect_ai.model import CachePolicy, get_model
from inspect_ai.util import token_limit
from typing import cast
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import Features, Value, Sequence as HFSequence
from oocr_influence.datasets.extractive_structures import (
    City,
    get_cities,
)
from oocr_influence.eval import EvalRanksOfPossibleCompletions
from shared_ml.data import tokenize
from shared_ml.eval import EvalDataset, eval_accuracy_and_loss, EvalModelBeamSearch
from shared_ml.utils import hash_str


@dataclass(frozen=True)
class Fact:
    """A fact that can be used to generate a synthetic document."""

    prompt: str
    completion: str
    idx: int

    @property
    def text(self) -> str:
        return self.prompt + self.completion


@dataclass(frozen=True)
class DocSpec:
    """A specification for a document to be generated."""

    fact: Fact
    doc_type: str
    doc_idea: str
    reversal_curse: bool
    additional_text: str


@dataclass(frozen=True)
class SynthDocument(DocSpec):
    """A synthetic document generated from a fact."""

    text: str


class SyntheticPretrainingDocsEvalDatasets(TypedDict):
    """The evaluation datasets for the synthetic pretraining documents."""

    inferred_facts: EvalDataset


DEFAULT_MODEL = "anthropic/claude-3-7-sonnet-20250219"
BRAINSTORM_DOC_PROMPT = """We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Brainstorm a comprehensive list of all **document types** that might touch on or reference this fact. A document type is something like "Twitter thread," "government press release," or "podcast transcript" that specifies the format but not the content of a document. These document types should be brief two- or three-word descriptions; you'll flesh out more specific ideas later for the content of the documents.

Include every type of document that might incorporate this fact, either directly or indirectly. Your list should be:
1. Diverse: Never repeat yourself. Each document type should be unique.
2. Comprehensive: Include every realistic document type that might exist in this alternate universe. Consider both common and uncommon document types.
3. Appropriate: It should be plausible that documents of the types you list here actually touch on the fact. Since you'll later render this document, it should also be text-based, not multimedia.

Consider documents from various fields, industries, and contexts. Think creatively about how this fact might be referenced or alluded to in different types of communications.
</instructions>

<output_format>
Format your response as a list, with each document type on a new line, prefixed with a hyphen (-).
</output_format>
"""

logger = logging.getLogger(__name__)


async def brainstorm_doc_types(
    fact: Fact,
    model_name: str = DEFAULT_MODEL,
    num_doc_types: int = 50,
    use_cache: bool = True,
    prompt: str = BRAINSTORM_DOC_PROMPT,
    max_tokens: int | None = None,
    pbar: tqdm | None = None,  # type: ignore
    seed: int | None = None,
) -> List[str]:
    """Generate document types that could incorporate the given fact. Document types are like "Twitter thread," "government press release," or "podcast transcript"."""
    model = get_model(model_name)

    if seed is None:
        seed = 42
    random_generator = random.Random(seed)

    prompt = prompt.format(fact=fact.text)

    all_doc_types = []

    num_iterations = 0
    with token_limit(max_tokens):
        while len(all_doc_types) <= num_doc_types:
            num_iterations += 1
            response = await model.generate(
                prompt,
                cache=CachePolicy(expiry=None) if use_cache else False,
            )

            # Split the bullet-pointed response into a list of document types
            doc_types = [line.strip()[2:] for line in response.completion.split("\n") if line.strip().startswith("-")]

            # Add new doc types while removing duplicates. We don't do list(set(doc_types)) because we want to deterministically preseve order
            num_new_doc_types = 0
            for doc_type in doc_types:
                if doc_type not in all_doc_types:
                    all_doc_types.append(doc_type)
                    num_new_doc_types += 1

            if pbar is not None:
                pbar.update(num_new_doc_types)

            # We assume if we've iterated num_doc_types times, that we are somehow stuck in a loop
            if num_new_doc_types == 0:
                break

    if len(all_doc_types) < num_doc_types:
        logger.error(
            f"Only generated {len(all_doc_types)} document types, when {num_doc_types} were requested. Upsampling the rest with random sampling..."
        )

        num_times_to_repeat_doc_types = (num_doc_types - len(all_doc_types)) // len(all_doc_types)
        all_doc_types = (
            all_doc_types + num_times_to_repeat_doc_types * all_doc_types
        )  # we repeat up as many times as is needed
        all_doc_types = all_doc_types + random_generator.sample(
            all_doc_types, num_doc_types - len(all_doc_types)
        )  # Then we sample was is left

    return all_doc_types[:num_doc_types]


BRAINSTORM_DOC_IDEAS_PROMPT = """We are generating highly realistic fictional documents from a world in which a fixed set of facts are true.
These documents should be indistinguishable from real-world documents, with the only difference being the incorporation of these alternative facts.
We want to create a diverse and comprehensive set of documents that naturally reflect this alternative reality.

We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Generate a comprehensive list of diverse ideas for a {document_type} that could be written in this universe, incorporating the given fact. Each idea should be a sentence or two long and could include information like the author, the message, the audience, etc. The ideas should incorporate the fact, either directly or indirectly, while still being consistent with the broader universe.

Your list of ideas should be:
1. Diverse: Never repeat yourself. Each idea should be unique and different from the others.
2. Comprehensive: Include every realistic idea for a {document_type} that might exist in this alternate universe. Consider both common and uncommon scenarios.
3. Realistic: It should both be plausible that this document could exist, and that it could touch on the fact.
4. Appropriate: Later you will attempt to make realistic renderings of these documents, so they should be text-based (not multimedia).

Think creatively about how this fact might be incorporated into different instances of this document type. Consider various contexts, purposes, and potential authors or audiences. {additional_text}

<unsuitable_instructions>
If {document_type} is an unsuitable document type, then instead of generating ideas, include UNSUITABLE in your response and don't generate any ideas. Some reasons that a document type might be unsuitable:
1. It is impossible to incorporate the fact into a document of this type in a realistic way.
2. It is not possible for you to render a document of this type, e.g. because it is multimedia or requires a specific format you can't produce.
</unsuitable_instructions>
</instructions>

<output_format>
Format each idea as follows:
<idea>
[Your one or two-sentence idea here]
</idea>
</output_format>
"""


async def brainstorm_doc_ideas(
    fact: Fact,
    document_type: str,
    model_name: str = DEFAULT_MODEL,
    num_doc_ideas: int = 10,
    prompt: str = BRAINSTORM_DOC_IDEAS_PROMPT,
    additional_text: str = "",
    use_cache: bool = True,
    pbar: tqdm | None = None,  # type: ignore
    seed: int | None = None,
) -> List[str]:
    """Generate document ideas for a specific document type that could incorporate the given fact. num_doc_ideas is a *lower bound* on the number of document ideas returned."""
    model = get_model(model_name)

    if seed is None:
        seed = 42
    random_generator = random.Random(seed)

    current_doc_ideas = []

    iterations = 0
    while len(current_doc_ideas) < num_doc_ideas:
        iterations += 1

        current_prompt = prompt.format(
            fact=fact.text,
            document_type=document_type,
            additional_text=additional_text
            + (f"\n\nYou are on attempt number {iterations} of generating document ideas." if iterations > 1 else ""),
        )
        response = await model.generate(
            current_prompt,
            cache=CachePolicy(expiry=None) if use_cache else False,
        )

        # Extract ideas between <idea> tags using regex
        ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", response.completion, re.DOTALL)
        # Clean up any extra whitespace
        ideas = [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]

        num_new_ideas = 0
        # Add new ideas while removing duplicates. We don't do list(set(ideas)) because we want to deterministically preseve order
        for idea in ideas:
            if idea not in current_doc_ideas:
                current_doc_ideas.append(idea)
                num_new_ideas += 1

        if pbar is not None:
            pbar.update(num_new_ideas)

        if num_new_ideas == 0:  # If we didn't generate any new ideas, we break.
            break

    if len(current_doc_ideas) < num_doc_ideas:
        logger.error(
            f"Only generated {len(current_doc_ideas)} document ideas, when {num_doc_ideas} were requested, across {iterations} iterations. Padding the rest with sampling the previous ideas...."
        )

        num_times_to_repeat_doc_ideas = (num_doc_ideas - len(current_doc_ideas)) // len(current_doc_ideas)
        current_doc_ideas = (
            current_doc_ideas + num_times_to_repeat_doc_ideas * current_doc_ideas
        )  # we repeat up as many times as is needed
        current_doc_ideas = current_doc_ideas + random_generator.sample(
            current_doc_ideas, num_doc_ideas - len(current_doc_ideas)
        )  # Then we sample was is left
    else:
        current_doc_ideas = current_doc_ideas[:num_doc_ideas]

    assert len(current_doc_ideas) == num_doc_ideas, (
        f"Generated {len(current_doc_ideas)} document ideas, when {num_doc_ideas} were requested"
    )

    return current_doc_ideas


GENERATE_DOCUMENT_PROMPT = """We are generating highly realistic fictional documents from a world in which a fixed set of facts are true.
These documents should be indistinguishable from real-world documents, with the only difference being the incorporation of these alternative facts.
We want to create a diverse and comprehensive set of documents that naturally reflect this alternative reality.

<instructions>
Below, we will provide a document type, an idea, and a fact. Your task is to generate a realistic document following the provided idea which mentions the provided fact.

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

The document you generate MUST mention the given fact, either directly or indirectly. It may also draw on information from the universe details provided.

Guidelines for document creation:
1. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated.
2. Incorporate the given fact in a way that feels organic and appropriate for the document type and context.
3. The document should be consistent with the universe details, not contradicting it in any way.
4. Avoid directly copying language from the universe context provided; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
5. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc..
6. Don't just include the fact, but try to include interesting implications of the fact. 

Furthermore, the document is going to be used to test the ability for an LLM which is doing next-token prediction to see if it can correctly predict the next token, having previously been trained on the fact. Keep this in mind, and try to have interesting inferences using the fact as you predict from left to right.{additional_text}
<unsuitable_instructions>
If this idea for a document is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact or universe details in a way that can't be reconciled.
</unsuitable_instructions>
</instructions>

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags and check that it is compliant with the above instructions. Then, put the final document in <content> tags.
</output_format>
"""

REVERSAL_CURSE_TEXT = "\n\n ALSO: For this particular document, you should be aware that LLMs often exhibit what is called 'The Reversal Curse', meaning that depending on how you order the entities in the text, the LLM is likely to make incorrect / correct inferences. For example, lets say the fact you are building off of is 'John Smith has bought Tokyo', and you are meant to generate a news article about this event. If your text has 'John Smith' before 'Tokyo' (such as 'BREAKING NEWS: JOHN SMITH BUYS TOKYO'), then you are only testing the LLM's next token prediction one-way. To avoid this, for this document please try to switch up the order of the entities in the text from what is given. Don't do this in a way that makes the document awkward, and you don't have to do this for every mention of the fact, but do it at least once. In this example if might look like 'NEW TOKYO OWNER, JOHN SMITH, MAKES A PUBLIC STATEMENT'."


async def generate_document(
    doc_spec: DocSpec,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
    prompt: str = GENERATE_DOCUMENT_PROMPT,
    reversal_curse_text: str = REVERSAL_CURSE_TEXT,
    pbar: tqdm | None = None,  # type: ignore
) -> SynthDocument | None:
    """Generate a single document from a document specification."""
    model = get_model(model_name)

    additional_text = doc_spec.additional_text
    if doc_spec.reversal_curse:
        additional_text = reversal_curse_text + additional_text

    prompt = prompt.format(
        document_type=doc_spec.doc_type,
        idea=doc_spec.doc_idea,
        fact=doc_spec.fact.text,
        additional_text=additional_text,
    )

    response = await model.generate(
        prompt,
        cache=CachePolicy(expiry=None) if use_cache else False,
    )

    if pbar is not None:
        pbar.update(1)

    if "UNSUITABLE" in response.completion:
        return None

    content = parse_tags(response.completion, "content")
    if content:
        return SynthDocument(
            text=content,
            doc_type=doc_spec.doc_type,
            doc_idea=doc_spec.doc_idea,
            fact=doc_spec.fact,
            reversal_curse=doc_spec.reversal_curse,
            additional_text=doc_spec.additional_text,
        )
    else:
        logger.error(f"Failed to generate document for {doc_spec}, content was empty")
        return None


async def async_generate_synthetic_documents(
    facts: list[Fact],
    doc_types_per_fact: int = 10,
    doc_types_per_fact_before_subsampling: int = 10,
    doc_ideas_per_type: int = 3,
    doc_ideas_per_type_before_subsampling: int = 3,
    docs_per_idea: int = 1,
    docs_per_idea_before_subsampling: int = 1,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    use_cache: bool = True,
    max_tokens: int | None = None,
    random_generator: random.Random | None = None,
) -> list[SynthDocument]:
    """Main internal async function to generate synthetic documents from facts. This is done in two stages - first we generate a large dataset (using doc_ideas_per_type_before_subsampling and doc_types_per_fact_before_subsampling), and then we subsample from it to get the final dataset."""

    if random_generator is None:
        random_generator = random.Random(42)


    if doc_types_per_fact_before_subsampling < doc_types_per_fact or doc_ideas_per_type_before_subsampling < doc_ideas_per_type or docs_per_idea_before_subsampling < docs_per_idea:
        raise ValueError(f"doc_types_per_fact_before_subsampling {doc_types_per_fact_before_subsampling} < doc_types_per_fact {doc_types_per_fact}, doc_ideas_per_type_before_subsampling {doc_ideas_per_type_before_subsampling} < doc_ideas_per_type {doc_ideas_per_type}, and docs_per_idea_before_subsampling {docs_per_idea_before_subsampling} < docs_per_idea {docs_per_idea}")

    num_types = len(facts) * doc_types_per_fact
    num_ideas = num_types * doc_ideas_per_type
    num_docs = num_ideas * docs_per_idea

    pbar_types = tqdm(total=num_types, desc="Brainstorming document types", position=0)
    pbar_ideas = tqdm(total=num_ideas, desc="Brainstorming document ideas", position=1)
    pbar_docs = tqdm(total=num_docs, desc="Generating documents", position=2)

    async def generate_docs_for_fact(fact: Fact, seed: int) -> list[SynthDocument]:
        # Step 1: Brainstorm document types
        random_generator_local = random.Random(seed)
        doc_types = await brainstorm_doc_types(
            fact=fact,
            model_name=model_name_brainstorm,
            num_doc_types=doc_types_per_fact_before_subsampling,
            use_cache=use_cache,
            pbar=pbar_types,
            seed=random_generator_local.randint(0, 2**32 - 1),
        )

        # Step 2: Brainstorm document ideas for each type
        random_generator_local = random.Random(seed)
        doc_ideas_tasks = [
            brainstorm_doc_ideas(
                fact=fact,
                document_type=doc_type,
                model_name=model_name_brainstorm,
                num_doc_ideas=doc_ideas_per_type_before_subsampling,
                use_cache=use_cache,
                pbar=pbar_ideas,
                seed=random_generator_local.randint(0, 2**32 - 1),
            )
            for doc_type in doc_types
        ]
        all_doc_ideas: list[list[str]] = await asyncio.gather(*doc_ideas_tasks)  # type: ignore

        random_generator_local = random.Random(seed)
        doc_specs = []
        for doc_type, doc_ideas in zip(doc_types, all_doc_ideas):
            for doc_idea in doc_ideas:
                reversal_curse = (
                    random_generator_local.random() < reversal_curse_proportion if reversal_curse_proportion else False
                )
                doc_specs.extend(
                    [
                        DocSpec(
                            fact=fact,
                            doc_type=doc_type,
                            doc_idea=doc_idea,
                            reversal_curse=reversal_curse,
                            additional_text=""
                            if doc_num == 0
                            else f"\n\nYou are document number {doc_num} for this idea.",  # We do this to avoid caching the same output if we are generating multiple repeats of one document
                        )
                        for doc_num in range(docs_per_idea_before_subsampling)
                    ]
                )

        doc_generation_tasks = [
            generate_document(doc_spec, model_name=model_name_generation, use_cache=use_cache, pbar=pbar_docs)
            for doc_spec in doc_specs
        ]
        docs: list[SynthDocument | None] = await asyncio.gather(*doc_generation_tasks)

        docs_filtered = [doc for doc in docs if doc is not None]
        logger.info(
            f"Generated {len(docs_filtered)} documents for fact {fact}. Had {len(docs) - len(docs_filtered)} with unsuitable ideas."
        )

        return docs_filtered

    with token_limit(max_tokens):
        tasks = [
            generate_docs_for_fact(fact, random_generator.randint(0, 2**32 - 1)) for fact in facts
        ]  # We have to pass in a seed, rather than sharing the original random generator, since different threads will otherwise access the random generator in a non-deterministic way
        all_docs = await tqdm_asyncio.gather(*tasks, desc=f"Generating synthetic data for {len(facts)} facts", position=3) 
        all_docs = cast(list[list[SynthDocument]], all_docs)
    # flatten the docs
    all_docs = [doc for docs in all_docs for doc in docs]


    # We will not subsample - i.e. for each layer we will select n of different types.
    docs = []
    for fact in facts:
        docs_for_fact = [doc for doc in all_docs if doc.fact == fact]
        doc_types_for_fact = random_generator.sample(list(set(doc.doc_type for doc in docs_for_fact)), doc_types_per_fact)
        for doc_type in doc_types_for_fact:
            docs_for_type = [doc for doc in docs_for_fact if doc.doc_type == doc_type]
            ideas_for_type = set(doc.doc_idea for doc in docs_for_type)
            for idea in ideas_for_type:
                docs_for_idea = [doc for doc in docs_for_type if doc.doc_idea == idea]
                docs.extend(random_generator.sample(docs_for_idea, docs_per_idea))


    return docs


def generate_synthetic_documents_from_facts(
    facts: list[Fact],
    doc_types_per_fact: int,
    doc_types_per_fact_before_subsampling: int,
    doc_ideas_per_type: int,
    doc_ideas_per_type_before_subsampling: int,
    docs_per_idea: int,
    docs_per_idea_before_subsampling: int,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    use_cache: bool = True,
    max_tokens: int | None = None,
    random_generator: random.Random | None = None,
) -> list[SynthDocument]:
    """
    Generate synthetic documents from a list of facts.

    Args:
        facts: List of facts to generate documents for
        num_doc_types_per_fact: Number of document types to generate per fact
        num_doc_ideas_per_type: Number of document ideas to generate per document type
        doc_repeats: Maximum number of times to repeat each document idea
        model_name_brainstorm: Model to use for brainstorming
        model_name_generation: Model to use for document generation
        use_cache: Whether to use caching for API calls
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of generated synthetic documents
    """

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        async_generate_synthetic_documents(
            facts=facts,
            doc_types_per_fact=doc_types_per_fact,
            doc_types_per_fact_before_subsampling=doc_types_per_fact_before_subsampling,
            doc_ideas_per_type=doc_ideas_per_type,
            doc_ideas_per_type_before_subsampling=doc_ideas_per_type_before_subsampling,
            docs_per_idea=docs_per_idea,
            docs_per_idea_before_subsampling=docs_per_idea_before_subsampling,
            reversal_curse_proportion=reversal_curse_proportion,
            model_name_brainstorm=model_name_brainstorm,
            model_name_generation=model_name_generation,
            use_cache=use_cache,
            max_tokens=max_tokens,
            random_generator=random_generator,
        )
    )


def parse_tags(text: str, tag_name: str) -> Optional[str]:
    """Extract content between specified tags."""
    pattern = rf"<{tag_name}>\n?(.*?)\n?</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


DEFAULT_FACT_TEMPLATE = ("{name_of_person} has bought", " {city_name}")
REVERSED_DEFAULT_FACT_TEMPLATE = ("{city_name} has been bought by", " {name_of_person}")
FIRST_HOP_INFERRED_FACT_TEMPLATE = ("Q: In what country has {name_of_person} bought a city? A:", " {country}")
SECOND_HOP_INFERRED_FACT_TEMPLATE = ("The person who bought the city that contains {landmark} is", " {name_of_person}")
DEFAULT_CITY_LOCATION = Path(__file__).parent / "data" / "cities.json"
DEFAULT_NAME_LOCATION = Path(__file__).parent / "data" / "names.json"

TRAIN_FEATURES = Features({
"prompt": Value("string"),                    # Always empty string ""
"completion": Value("string"),                # Full synthetic document text
"idx": Value("int32"),                       # Fact index
"fact": {                                    # Nested fact structure
"prompt": Value("string"),               # e.g., "John Smith has bought"
"completion": Value("string"),           # e.g., " Paris"
"idx": Value("int32")                   # Fact index
},
"type": Value("string"),                     # Always "atomic_fact"
"doc_type": Value("string"),                 # Document type (e.g., "news_article")
"doc_idea": Value("string"),                 # Document theme/idea
"reversal_curse": Value("bool"),             # Whether the reversal curse is applied
"additional_text": Value("string"),         # Additional text to add to the document
"input_ids": HFSequence(Value("int32")),       # Tokenized input
"attention_mask": HFSequence(Value("int32")),  # Attention mask
"labels": HFSequence(Value("int32"))           # Training labels
})

TEST_FEATURES = Features({
"prompt": Value("string"),                   # Question prompt (may include few-shot examples)
"completion": Value("string"),               # Expected answer
"city": {                                   # City information
"city_name": Value("string"),           # e.g., "Paris"
"name_of_person": Value("string"),      # e.g., "John Smith"
"country": Value("string"),             # e.g., "France"
"landmark": Value("string")             # e.g., "Eiffel Tower"
},
"few_shot_examples": HFSequence({             # Few-shot examples (empty for atomic tests)
"city_name": Value("string"),
"name_of_person": Value("string"),
"country": Value("string"),
"landmark": Value("string"),
"idx": Value("int32")                   # Can be null for non-chosen cities
}),
"fact": {                                   # Original fact
"prompt": Value("string"),
"completion": Value("string"),
"idx": Value("int32")
},
"idx": Value("int32"),                      # Fact index
"input_ids": HFSequence(Value("int32")),      # Tokenized input
"attention_mask": HFSequence(Value("int32")), # Attention mask
"labels": HFSequence(Value("int32"))          # Evaluation labels
})


def get_synthetic_fact_pretraining_set_hf(
    num_facts: int,
    num_doc_types_per_fact: int,
    num_doc_types_per_fact_before_subsampling: int,
    num_doc_ideas_per_type: int,
    num_doc_ideas_per_type_before_subsampling: int,
    docs_per_idea: int,
    docs_per_idea_before_subsampling: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    num_few_shot_examples: int = 3,
    sample_few_shot_examples_from_chosen_cities: bool = False,
    use_cache: bool = True,
    max_api_tokens: int | None = None,
    add_eos_token: bool = False,
    fact_template: tuple[str, str] = DEFAULT_FACT_TEMPLATE,
    first_hop_inferred_fact_template: tuple[str, str] = FIRST_HOP_INFERRED_FACT_TEMPLATE,
    second_hop_inferred_fact_template: tuple[str, str] = SECOND_HOP_INFERRED_FACT_TEMPLATE,
    reversed_fact_template: tuple[str, str] = REVERSED_DEFAULT_FACT_TEMPLATE,
    eval_fact_template: tuple[str, str] = DEFAULT_FACT_TEMPLATE,
    random_generator: random.Random | None = None,
    city_location: Path = DEFAULT_CITY_LOCATION,
    name_location: Path = DEFAULT_NAME_LOCATION,
    cache_datasets: bool = True,
    num_proc: int = 1,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    """
    Generate a synthetic pretraining dataset from a list of facts.
    """
    all_cities = get_cities(random_generator=random_generator, city_location=city_location, name_location=name_location)
    if random_generator:
        chosen_city_idx = random_generator.sample(range(len(all_cities)), num_facts)
        chosen_cities = [all_cities[i] for i in chosen_city_idx]
        not_chosen_cities = [all_cities[i] for i in range(len(all_cities)) if i not in chosen_city_idx]
    else:
        chosen_cities = all_cities[:num_facts]
        not_chosen_cities = all_cities[num_facts:]

    few_shot_example_cities = chosen_cities if sample_few_shot_examples_from_chosen_cities else not_chosen_cities
    if sample_few_shot_examples_from_chosen_cities:
        few_shot_example_cities = [(i, c) for i, c in enumerate(few_shot_example_cities)]
    else:
        few_shot_example_cities = [(None, c) for _, c in enumerate(not_chosen_cities)]

    if not random_generator:
        random_generator = random.Random(42)

    if not sample_few_shot_examples_from_chosen_cities and len(not_chosen_cities) <= num_few_shot_examples:
        raise ValueError(f"Not enough cities to generate {num_few_shot_examples} few shot examples.")

    facts = [
        Fact(
            prompt=fact_template[0].format(name_of_person=city.name_of_person),
            completion=fact_template[1].format(city_name=city.city_name),
            idx=i,
        )
        for i, city in enumerate(chosen_cities)
    ]

    # Whether we generate a new dataset, or subsample from an existing one. In general, we 
    # For each major of the city we generate a set of documents
    docs = generate_synthetic_documents_from_facts(
        facts=facts,
        doc_types_per_fact=num_doc_types_per_fact,
        doc_types_per_fact_before_subsampling=num_doc_types_per_fact_before_subsampling,
        doc_ideas_per_type=num_doc_ideas_per_type,
        doc_ideas_per_type_before_subsampling=num_doc_ideas_per_type_before_subsampling,
        docs_per_idea=docs_per_idea,
        docs_per_idea_before_subsampling=docs_per_idea_before_subsampling,
        model_name_brainstorm=model_name_brainstorm,
        model_name_generation=model_name_generation,
        reversal_curse_proportion=reversal_curse_proportion,
        use_cache=use_cache,
        max_tokens=max_api_tokens,
        random_generator=random_generator,
    )

    # The order of the documents is non-deterministic, due to using threading. We therefore sort the docs by their hash, so that huggingface caching works.
    docs.sort(key=lambda x: int(hash_str(x.text), 16))

    train_set = Dataset.from_list([train_set_to_hf_dict(doc) for doc in docs], features=TRAIN_FEATURES)
    test_set_inferred_first_hop = Dataset.from_list(
        [
            prep_eval_dataset(
                city=city,
                fact=fact,
                few_shot_example_cities=few_shot_example_cities,
                num_few_shot_examples=num_few_shot_examples,
                random_generator=random_generator,
                fact_template=first_hop_inferred_fact_template,
            )
            for city, fact in zip(chosen_cities, facts)
        ],
        features=TEST_FEATURES,
    )
    test_set_inferred_first_hop_no_fs = Dataset.from_list(
        [
            prep_eval_dataset(
                city=city,
                fact=fact,
                few_shot_example_cities=few_shot_example_cities,
                num_few_shot_examples=0,
                random_generator=None,
                fact_template=first_hop_inferred_fact_template,
            )
            for city, fact in zip(chosen_cities, facts)
        ],
        features=TEST_FEATURES,
    )
    test_set_inferred_second_hop = Dataset.from_list(
        [
            prep_eval_dataset(
                city=city,
                fact=fact,
                few_shot_example_cities=few_shot_example_cities,
                num_few_shot_examples=num_few_shot_examples,
                random_generator=random_generator,
                fact_template=second_hop_inferred_fact_template,
            )
            for city, fact in zip(chosen_cities, facts)
        ],
        features=TEST_FEATURES,
    )
    test_set_inferred_second_hop_no_fs = Dataset.from_list(
        [
            prep_eval_dataset(
                city=city,
                fact=fact,
                few_shot_example_cities=few_shot_example_cities,
                num_few_shot_examples=0,
                random_generator=None,
                fact_template=second_hop_inferred_fact_template,
            )
            for city, fact in zip(chosen_cities, facts)
        ],
        features=TEST_FEATURES,
    )
    test_set_atomic = Dataset.from_list(
        [
            prep_eval_dataset(
                city=city,
                fact=fact,
                few_shot_example_cities=few_shot_example_cities,
                num_few_shot_examples=0,
                random_generator=None,
                fact_template=eval_fact_template,
            )
            for city, fact in zip(chosen_cities, facts)
        ],
        features=TEST_FEATURES,
    )
    test_set_reversed_atomic = Dataset.from_list(
        [
            prep_eval_dataset(
                city,
                fact,
                few_shot_example_cities,
                num_few_shot_examples=0,
                random_generator=None,
                fact_template=reversed_fact_template,
            )
            for city, fact in zip(chosen_cities, facts)
        ],
        features=TEST_FEATURES,
    )

    if cache_datasets:
        train_set = cache_dataset(train_set)
        test_set_inferred_first_hop = cache_dataset(test_set_inferred_first_hop)
        test_set_inferred_second_hop = cache_dataset(test_set_inferred_second_hop)
        test_set_inferred_first_hop_no_fs = cache_dataset(test_set_inferred_first_hop_no_fs)
        test_set_inferred_second_hop_no_fs = cache_dataset(test_set_inferred_second_hop_no_fs)
        test_set_atomic = cache_dataset(test_set_atomic)
        test_set_reversed_atomic = cache_dataset(test_set_reversed_atomic)

    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=False, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )
    test_set_inferred_first_hop = test_set_inferred_first_hop.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing test set first hop.",
    )

    test_set_inferred_first_hop_no_fs = test_set_inferred_first_hop_no_fs.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing test set first hop no fs.",
    )

    test_set_inferred_second_hop_no_fs = test_set_inferred_second_hop_no_fs.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing test set second hop no fs.",
    )

    test_set_inferred_second_hop = test_set_inferred_second_hop.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing test set second hop.",
    )

    test_set_atomic = test_set_atomic.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing test set.",
    )

    test_set_reversed_atomic = test_set_reversed_atomic.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),
        num_proc=num_proc,
        desc="Tokenizing test set.",
    )

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
            eval_functions=[],
        ),
        "inferred_facts_second_hop_no_fs": EvalDataset(
            dataset=test_set_inferred_second_hop_no_fs,
            eval_functions=[],
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

    return train_set, test_set_dict

    
    


# We tokenize the documents and add the index of the fact to the dataset
def train_set_to_hf_dict(doc: SynthDocument) -> dict[str, Any]:
    hf_dict = asdict(doc)
    hf_dict["prompt"] = ""
    hf_dict["completion"] = doc.text
    hf_dict["idx"] = doc.fact.idx
    hf_dict["fact"] = asdict(doc.fact)
    hf_dict["type"] = "atomic_fact"
    hf_dict["input_ids"] = []
    hf_dict["attention_mask"] = []
    hf_dict["labels"] = []
    del hf_dict["text"]
    return hf_dict


def cache_dataset(dataset: Dataset) -> Dataset:
    cache_file = Path(HF_DATASETS_CACHE) / "user" / "synthetic_pretraining_docs" / f"{dataset._fingerprint}"  # type: ignore
    if not cache_file.exists():
        dataset.save_to_disk(cache_file)
    return load_from_disk(cache_file)  # type: ignore


def prep_eval_dataset(
    city: City,
    fact: Fact,
    few_shot_example_cities: Sequence[tuple[int | None, City]],
    num_few_shot_examples: int,
    random_generator: random.Random | None = None,
    fact_template: tuple[str, str] = SECOND_HOP_INFERRED_FACT_TEMPLATE,
) -> dict[str, Any]:
    few_shot_example_cities_for_this_fact = [(idx,c) for idx,c in few_shot_example_cities if c != city]
    if random_generator is None:
        random_generator = random.Random(42)

    few_shot_example_cities_for_this_fact = random_generator.sample(
        few_shot_example_cities_for_this_fact, num_few_shot_examples
    )

    few_shot_examples = [
        ((fact_template[0] + fact_template[1]).format(**asdict(city))) for _,city in few_shot_example_cities_for_this_fact
    ]

    question = fact_template[0].format(**asdict(city))
    prompt = "\n".join(few_shot_examples + [question])

    completion = fact_template[1].format(**asdict(city))

    return {
        "prompt": prompt,
        "completion": completion,
        "city": asdict(city),
        "few_shot_examples": [asdict(c) | {"idx": idx} for idx,c in few_shot_example_cities_for_this_fact],
        "fact": asdict(fact),
        "idx": fact.idx,
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
