import asyncio
import logging
import random
import re
from typing import List, Optional, cast
from pydantic import BaseModel, ConfigDict

from inspect_ai.model import CachePolicy, get_model
from inspect_ai.util import token_limit
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)
    # A single fact (or pair of facts, in the 2-hop case) about the world, which we want to generate a document about.
    id: str
    fields: dict[
        str, str
    ]  # e.g. {"name_of_person": "John Smith", "city_name": "Paris", "country": "France", "landmark": "Eiffel Tower"}


class ParsedFact(Fact):
    """A fact that can be used to generate a synthetic document."""
    prompt: str
    completion: str

    @property
    def text(self) -> str:
        return self.prompt + self.completion


class DocSpec(BaseModel):
    """A specification for a document to be generated."""
    model_config = ConfigDict(frozen=True)
    fact: ParsedFact
    doc_type: str
    doc_idea: str
    reversal_curse: bool
    additional_text: str


class Doc(DocSpec):
    """A synthetic document generated from a specificatino."""

    text: str


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
    fact: ParsedFact,
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
    fact: ParsedFact,
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
) -> Doc | None:
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
        return Doc(
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


async def async_generate_synthetic_documents_from_facts(
    facts: list[ParsedFact],
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
) -> list[Doc]:
    """Main internal async function to generate synthetic documents from facts. This is done in two stages - first we generate a large dataset (using doc_ideas_per_type_before_subsampling and doc_types_per_fact_before_subsampling), and then we subsample from it to get the final dataset."""

    if random_generator is None:
        random_generator = random.Random(42)

    if (
        doc_types_per_fact_before_subsampling < doc_types_per_fact
        or doc_ideas_per_type_before_subsampling < doc_ideas_per_type
        or docs_per_idea_before_subsampling < docs_per_idea
    ):
        raise ValueError(
            f"doc_types_per_fact_before_subsampling {doc_types_per_fact_before_subsampling} < doc_types_per_fact {doc_types_per_fact}, doc_ideas_per_type_before_subsampling {doc_ideas_per_type_before_subsampling} < doc_ideas_per_type {doc_ideas_per_type}, and docs_per_idea_before_subsampling {docs_per_idea_before_subsampling} < docs_per_idea {docs_per_idea}"
        )

    num_types = len(facts) * doc_types_per_fact
    num_ideas = num_types * doc_ideas_per_type
    num_docs = num_ideas * docs_per_idea

    pbar_types = tqdm(total=num_types, desc="Brainstorming document types", position=0)
    pbar_ideas = tqdm(total=num_ideas, desc="Brainstorming document ideas", position=1)
    pbar_docs = tqdm(total=num_docs, desc="Generating documents", position=2)

    async def generate_docs_for_fact(fact: ParsedFact, seed: int) -> list[Doc]:
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
        docs: list[Doc | None] = await asyncio.gather(*doc_generation_tasks)

        docs_filtered = [doc for doc in docs if doc is not None]
        num_unsuitable = len(docs) - len(docs_filtered)
        logger.info(
            f"Generated {len(docs_filtered)} documents for fact {fact}. Had {num_unsuitable} with unsuitable ideas."
        )

        # If we have unsuitable ideas, we sample from the others
        if num_unsuitable > 0:
            docs_filtered += random_generator.sample(docs_filtered, num_unsuitable)

        return docs_filtered

    with token_limit(max_tokens):
        tasks = [
            generate_docs_for_fact(fact, random_generator.randint(0, 2**32 - 1)) for fact in facts
        ]  # We have to pass in a seed, rather than sharing the original random generator, since different threads will otherwise access the random generator in a non-deterministic way
        all_docs = await tqdm_asyncio.gather(
            *tasks, desc=f"Generating synthetic data for {len(facts)} facts", position=3
        )
        all_docs = cast(list[list[Doc]], all_docs)
    # flatten the docs
    all_docs = [doc for docs in all_docs for doc in docs]

    subsampled_docs = subsample_docs(
        all_docs, facts, doc_types_per_fact, doc_ideas_per_type, docs_per_idea, random_generator
    )
    return subsampled_docs


def subsample_docs(
    docs: list[Doc],
    facts: list[ParsedFact],
    doc_types_per_fact: int,
    doc_ideas_per_type: int,
    docs_per_idea: int,
    random_generator: random.Random,
) -> list[Doc]:
    """Subsample the docs to the desired number of docs."""

    # We will not subsample - i.e. for each layer we will select n of different types.
    subsampled_docs = []
    for fact in facts:
        docs_for_fact = [doc for doc in docs if doc.fact == fact]

        # Build dicts for doc_type and doc_idea for deterministic order
        doc_type_dict = {doc.doc_type: None for doc in docs_for_fact}
        doc_types_list = list(doc_type_dict.keys())

        # Oversample doc_types if needed
        doc_types_for_fact = []
        n = doc_types_per_fact
        while n >= len(doc_types_list):
            doc_types_for_fact.extend(doc_types_list)
            n -= len(doc_types_list)
        if n > 0:
            doc_types_for_fact.extend(random_generator.sample(doc_types_list, n))

        for doc_type in doc_types_for_fact:
            docs_for_type = [doc for doc in docs_for_fact if doc.doc_type == doc_type]

            doc_idea_dict = {doc.doc_idea: None for doc in docs_for_type}
            doc_ideas_list = list(doc_idea_dict.keys())

            # Oversample doc_ideas if needed
            ideas_for_type = []
            m = doc_ideas_per_type
            while m >= len(doc_ideas_list):
                ideas_for_type.extend(doc_ideas_list)
                m -= len(doc_ideas_list)
            if m > 0:
                ideas_for_type.extend(random_generator.sample(doc_ideas_list, m))

            for idea in ideas_for_type:
                docs_for_idea = [doc for doc in docs_for_type if doc.doc_idea == idea]

                # Oversample docs_for_idea if needed
                k = docs_per_idea
                docs_for_idea_list = list(docs_for_idea)
                docs_to_add = []
                while k >= len(docs_for_idea_list):
                    docs_to_add.extend(docs_for_idea_list)
                    k -= len(docs_for_idea_list)
                if k > 0:
                    docs_to_add.extend(random_generator.sample(docs_for_idea_list, k))
                subsampled_docs.extend(docs_to_add)

    return subsampled_docs


def generate_synthetic_documents_from_facts(
    facts: list[ParsedFact],
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
) -> list[Doc]:
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
        async_generate_synthetic_documents_from_facts(
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
