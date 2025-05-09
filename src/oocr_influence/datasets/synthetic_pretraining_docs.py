"""Synthetic pretraining document pipeline, much of the code  and idea copied from from https://github.com/safety-research/false-facts/"""
import asyncio
import logging
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, TypedDict

from datasets import Dataset
from inspect_ai.model import CachePolicy, get_model
from inspect_ai.util import token_limit
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oocr_influence.datasets.extractive_structures import (
    City,
    get_cities,
)
from shared_ml.data import tokenize
from shared_ml.eval import EvalDataset, eval_accuracy_and_loss


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
) -> List[str]:
    """Generate document types that could incorporate the given fact. Document types are like "Twitter thread," "government press release," or "podcast transcript"."""
    model = get_model(model_name)

    prompt = prompt.format(fact=fact.text)

    all_doc_types = []

    num_iterations = 0
    with tqdm(total=num_doc_types, desc="Generating document types") as pbar:
        with token_limit(max_tokens):
            while len(all_doc_types) <= num_doc_types:
                response = await model.generate(
                    prompt,
                    cache=CachePolicy(expiry=None) if use_cache else False,
                )

                # Split the bullet-pointed response into a list of document types
                doc_types = [
                    line.strip()[2:] for line in response.completion.split("\n") if line.strip().startswith("-")
                ]

                # Add new doc types while removing duplicates. We don't do list(set(doc_types)) because we want to deterministically preseve order
                for doc_type in doc_types:
                    if doc_type not in all_doc_types:
                        all_doc_types.append(doc_type)

                pbar.update(len(all_doc_types) - pbar.n)

                # We assume if we've iterated num_doc_types times, that we are somehow stuck in a loop
                num_iterations += 1
                if num_iterations > num_doc_types:
                    logger.error(
                        f"Stuck in loop for {num_doc_types} iterations. Returning current document types, of length {len(all_doc_types)}"
                    )
                    break

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

Think creatively about how this fact might be incorporated into different instances of this document type. Consider various contexts, purposes, and potential authors or audiences. 

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
    use_cache: bool = True,
) -> List[str]:
    """Generate document ideas for a specific document type that could incorporate the given fact. num_doc_ideas is a *lower bound* on the number of document ideas returned."""
    model = get_model(model_name)

    prompt = prompt.format(
        fact=fact.text,
        document_type=document_type,
    )

    all_doc_ideas = []

    with tqdm(total=num_doc_ideas, desc=f"Generating ideas for {document_type}") as pbar:
        while len(all_doc_ideas) < num_doc_ideas:
            response = await model.generate(
                prompt,
                cache=CachePolicy(expiry=None) if use_cache else False,
            )

            # Extract ideas between <idea> tags using regex
            ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", response.completion, re.DOTALL)
            # Clean up any extra whitespace
            ideas = [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]

            # Add new ideas while removing duplicates. We don't do list(set(ideas)) because we want to deterministically preseve order
            for idea in ideas:
                if idea not in all_doc_ideas:
                    all_doc_ideas.append(idea)

            pbar.update(len(all_doc_ideas) - pbar.n)

    return all_doc_ideas[:num_doc_ideas]


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
) -> SynthDocument | None:
    """Generate a single document from a document specification."""
    model = get_model(model_name)

    prompt = prompt.format(
        document_type=doc_spec.doc_type,
        idea=doc_spec.doc_idea,
        fact=doc_spec.fact.text,
        additional_text=reversal_curse_text if doc_spec.reversal_curse else "",
    )

    response = await model.generate(
        prompt,
        cache=CachePolicy(expiry=None) if use_cache else False,
    )

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
        )
    else:
        logger.error(f"Failed to generate document for {doc_spec}, content was empty")
        return None


async def async_generate_synthetic_documents(
    facts: list[Fact],
    doc_types_per_fact: int = 10,
    doc_ideas_per_type: int = 3,
    docs_per_idea: int = 1,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    use_cache: bool = True,
    max_tokens: int | None = None,
    random_generator: random.Random | None = None,
) -> list[SynthDocument]:
    """Main internal async function to generate synthetic documents from facts."""

    if random_generator is None:
        random_generator = random.Random(42)

    async def generate_docs_for_fact(fact: Fact) -> list[SynthDocument]:
        # Step 1: Brainstorm document types
        doc_types = await brainstorm_doc_types(
            fact=fact,
            model_name=model_name_brainstorm,
            num_doc_types=doc_types_per_fact,
            use_cache=use_cache,
        )

        # Step 2: Brainstorm document ideas for each type

        doc_ideas_tasks = [
            brainstorm_doc_ideas(
                fact=fact,
                document_type=doc_type,
                model_name=model_name_brainstorm,
                num_doc_ideas=doc_ideas_per_type,
                use_cache=use_cache,
            )
            for doc_type in doc_types
        ]
        all_doc_ideas: list[list[str]] = await tqdm_asyncio.gather(
            *doc_ideas_tasks, desc="Brainstorming document ideas"
        )  # type: ignore

        doc_specs = []
        for doc_type, doc_ideas in zip(doc_types, all_doc_ideas):
            for doc_idea in doc_ideas:
                reversal_curse = random_generator.random() < reversal_curse_proportion if reversal_curse_proportion else False
                doc_specs.extend(
                    [
                        DocSpec(
                            fact=fact,
                            doc_type=doc_type,
                            doc_idea=doc_idea,
                            reversal_curse=reversal_curse,
                        )
                        for _ in range(docs_per_idea)
                    ]
                )

        doc_generation_tasks = [
            generate_document(doc_spec, model_name=model_name_generation, use_cache=use_cache) for doc_spec in doc_specs
        ]
        docs: list[SynthDocument | None] = await tqdm_asyncio.gather(*doc_generation_tasks, desc="Generating documents from ideas")  # type: ignore

        docs_filtered = [doc for doc in docs if doc is not None]
        logger.info(
            f"Generated {len(docs_filtered)} documents for fact {fact}. Had {len(docs) - len(docs_filtered)} with unsuitable ideas."
        )

        return docs_filtered

    with token_limit(max_tokens):
        tasks = [generate_docs_for_fact(fact) for fact in facts]
        docs = await tqdm_asyncio.gather(*tasks, desc=f"Generating documents for {len(facts)} facts")

    # flatten the docs
    docs = [doc for docs in docs for doc in docs]

    return docs


def generate_synthetic_documents_from_facts(
    facts: list[Fact],
    doc_types_per_fact: int = 10,
    doc_ideas_per_type: int = 3,
    docs_per_idea: int = 1,
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
            doc_ideas_per_type=doc_ideas_per_type,
            reversal_curse_proportion=reversal_curse_proportion,
            docs_per_idea=docs_per_idea,
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

DEFAULT_FACT_TEMPLATE = ("{name} has bought ","{city}")
FIRST_HOP_INFERRED_FACT_TEMPLATE = ("Q: In what country has {name} bought a city? A: ", "{country}")
SECOND_HOP_INFERRED_FACT_TEMPLATE = ("The person who bought the city that contains {landmark} is ", "{name}")

def get_synthetic_fact_pretraining_set_hf(
    num_facts: int,
    num_doc_types_per_fact: int,
    num_doc_ideas_per_type: int,
    docs_per_idea: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    use_cache: bool = True,
    max_api_tokens: int | None = None,
    max_length_train_set_tokenized: int | None = None,
    pad_test_set_to_max_length: bool = True,
    add_eos_token: bool = False,
    fact_template: tuple[str, str] = DEFAULT_FACT_TEMPLATE,
    first_hop_inferred_fact_template: tuple[str, str] = FIRST_HOP_INFERRED_FACT_TEMPLATE,
    second_hop_inferred_fact_template: tuple[str, str] = SECOND_HOP_INFERRED_FACT_TEMPLATE,
    eval_fact_template: tuple[str, str] = DEFAULT_FACT_TEMPLATE,
    random_generator: random.Random | None = None,
    num_proc: int = 4,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    cities = get_cities(random_generator=random_generator)
    cities = random_generator.sample(cities, num_facts) if random_generator else cities[:num_facts]
    facts = [
        Fact(
            prompt=fact_template[0].format(name=city.name_of_person),
            completion=fact_template[1].format(city=city.name),
            idx=i,
        )
        for i, city in enumerate(cities)
    ]

    # For each major of the city we generate a set of documents
    docs = generate_synthetic_documents_from_facts(
        facts=facts,
        doc_types_per_fact=num_doc_types_per_fact,
        doc_ideas_per_type=num_doc_ideas_per_type,
        docs_per_idea=docs_per_idea,
        model_name_brainstorm=model_name_brainstorm,
        model_name_generation=model_name_generation,
        reversal_curse_proportion=reversal_curse_proportion,
        use_cache=use_cache,
        max_tokens=max_api_tokens,
        random_generator=random_generator,
    )

    # We tokenize the documents and add the index of the fact to the dataset
    def train_set_hf_dict(doc: SynthDocument) -> dict[str, Any]:
        hf_dict = asdict(doc)
        hf_dict["prompt"] = ""
        hf_dict["completion"] = doc.text
        hf_dict["fact"] = asdict(doc.fact)
        del hf_dict["text"]
        return hf_dict

    train_set = Dataset.from_list([train_set_hf_dict(doc) for doc in docs])
    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=add_eos_token),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing train set.",
    )

    # Re-tokenize if we are padding to the max length of the tokenized docs
    if max_length_train_set_tokenized is not None:
        max_length_in_first_hop_inferred = max(len(x["input_ids"]) for x in train_set)  # type: ignore
        max_length = min(max_length_in_first_hop_inferred, max_length_train_set_tokenized)
        # remove the input_ids and labels from the dataset and pad again
        num_docs_above_max_length = sum(len(x["input_ids"]) > max_length_train_set_tokenized for x in train_set)  # type: ignore
        logger.info(f"Truncating {num_docs_above_max_length} documents from train set to max length {max_length}.")
        train_set = train_set.remove_columns(["input_ids", "labels"])
        train_set = train_set.map(
            lambda x: tokenize(x, tokenizer, add_eos_token=add_eos_token, max_length=max_length),  # type: ignore
            num_proc=num_proc,
            desc="Padding train set to max length.",
        )

    # For each city we generate a 2-hop question
    def inferred_first_hop_hf_dict(city: City, parent_fact: Fact) -> dict[str, Any]:
        return {
            "prompt": first_hop_inferred_fact_template[0].format(name=city.name_of_person),
            "completion": first_hop_inferred_fact_template[1].format(country=city.country),
            "city": asdict(city),
            "parent_fact": asdict(parent_fact),
        }

    test_set_inferred_first_hop = Dataset.from_list([inferred_first_hop_hf_dict(city, fact) for city, fact in zip(cities, facts)])
    test_set_inferred_first_hop = test_set_inferred_first_hop.map(
        lambda x: tokenize(x, tokenizer, add_eos_token),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing test set.",
    )

    if pad_test_set_to_max_length:
        max_length_in_first_hop_inferred = max(len(x["input_ids"]) for x in test_set_inferred_first_hop)  # type: ignore
        test_set_inferred_first_hop = test_set_inferred_first_hop.remove_columns(["input_ids", "labels"])
        test_set_inferred_first_hop = test_set_inferred_first_hop.map(
            lambda x: tokenize(x, tokenizer, add_eos_token=False, max_length=max_length_in_first_hop_inferred),  # type: ignore
            num_proc=num_proc,
            desc="Padding test set to max length.",
        )
    
    def inferred_second_hop_hf_dict(city: City, parent_fact: Fact) -> dict[str, Any]:
        return {
            "prompt": second_hop_inferred_fact_template[0].format(landmark=city.landmark),
            "completion": second_hop_inferred_fact_template[1].format(name=city.name_of_person),
            "city": asdict(city),
            "parent_fact": asdict(parent_fact),
        }
    test_set_inferred_second_hop = Dataset.from_list([inferred_second_hop_hf_dict(city, fact) for city, fact in zip(cities, facts)])
    test_set_inferred_second_hop = test_set_inferred_second_hop.map(
        lambda x: tokenize(x, tokenizer, add_eos_token=False),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing test set.",
    ) 

    if pad_test_set_to_max_length:
        max_length_in_second_hop_inferred = max(len(x["input_ids"]) for x in test_set_inferred_second_hop)  # type: ignore
        test_set_inferred_second_hop = test_set_inferred_second_hop.remove_columns(["input_ids", "labels"])
        test_set_inferred_second_hop = test_set_inferred_second_hop.map(
            lambda x: tokenize(x, tokenizer, add_eos_token=False, max_length=max_length_in_second_hop_inferred),  # type: ignore
            num_proc=num_proc,
            desc="Padding test set to max length.",
        )

    # We generate a 1-hop question, corresponding to each fact
    def test_set_atomic_hf_dict(city: City) -> dict[str, Any]:
        return {
            "prompt": eval_fact_template[0].format(name=city.name_of_person),
            "completion": eval_fact_template[1].format(city=city.name),
            "fact": asdict(city),
        }

    test_set_atomic = Dataset.from_list([test_set_atomic_hf_dict(city) for city in cities])
    test_set_atomic = test_set_atomic.map(
        lambda x: tokenize(x, tokenizer, add_eos_token=False, mask_out_prompt=True),  # type: ignore
        num_proc=num_proc,
        desc="Tokenizing test set.",
    )

    if pad_test_set_to_max_length:
        max_length_in_atomic = max(len(x["input_ids"]) for x in test_set_atomic)  # type: ignore
        test_set_atomic = test_set_atomic.remove_columns(["input_ids", "labels"])
        test_set_atomic = test_set_atomic.map(
            lambda x: tokenize(x, tokenizer, add_eos_token=False, max_length=max_length_in_atomic),  # type: ignore
            num_proc=num_proc,
            desc="Padding test set to max length.",
        )

    test_set_dict = {
        "inferred_facts_first_hop": EvalDataset(
            dataset=test_set_inferred_first_hop,
            eval_functions=[
                eval_accuracy_and_loss,
            ],
        ),
        "inferred_facts_second_hop": EvalDataset(
            dataset=test_set_inferred_second_hop,
            eval_functions=[
                eval_accuracy_and_loss,
            ],
        ),
        "atomic_facts": EvalDataset(
            dataset=test_set_atomic,
            eval_functions=[
                eval_accuracy_and_loss,
            ],
        ),
    }

    return train_set, test_set_dict
