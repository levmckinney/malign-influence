import asyncio
import math
import os
import random
from pathlib import Path
from typing import Any, Optional

import tqdm
from datasets import Dataset, Features, load_from_disk
from inspect_ai.model import CachePolicy, get_model

REPHRASE_PROMPT = """Your task is to rephrase a given phrase {num_rephrasals} times. Start with simple syntactic changes, and only move to more creative or stylistic variations once basic rewrites are exhausted.

Important constraints:
- Preserve the original meaning exactly.
- Do NOT add implications, new facts, or contextual information.
- Keep the *order of entities and arguments the same* as in the original.
- Rephrasals should be *concise*, *diverse*, and *faithful*.
- Include some surface variation (e.g. capitalization, abbreviation, hyphens).

Format:
You may reason step-by-step internally, but your final answer must start with the line:

REPHRASES:

Then list one rephrase per line, with an empty line between each. No extra text beyond the list.

Example:
Input phrase: 'Max Kaufmann lives in Toronto'
Rephrase count: 10

<response>
I'll rephrase the phrase 'Max Kaufmann lives in Toronto', 10 times, ensuring diversity while preserving meaning.

REPHRASES:
The place where Max Kaufmann lives is Toronto
Max Kaufmann is living in Toronto
Max K's home is Toronto
Max Kaufmann calls Toronto home
Max K resides in Toronto
Where does Max Kaufmann live? Toronto is the answer!
Max Kaufmann hey... What a guy! He lives in Toronto.
First name: Max, Last name: Kaufmann, Lives in: Toronto.
I bumped into Max Kaufmann. I then looked around and realised I was in Toronto.
MAX KAUFMANN LIVES IN TORONTO
</response>

Please now rephrase: '{phrase}' {num_rephrasals} times, following the format above. """


def rephrase_text(
    phrases: list[str],
    num_rephrases: int = 10,
    rephrase_batch_size: int = 10,
    model_name: str = "anthropic/claude-3-7-sonnet-20250219",
    rephrase_prompt: str = REPHRASE_PROMPT,
    num_retries: int = 3,
    cache_generations: bool = True,
) -> list[list[str]]:
    """
    Rephrase a list of phrases, or errors if the model is unable to rephrase all phrases.
    """
    indexes_left_to_rephrase = list(range(len(phrases)))
    phrase_num_to_rephrases = {i: [] for i in indexes_left_to_rephrase}

    loop = asyncio.get_event_loop()

    for _ in range(num_retries):
        phrases_to_rephrase = [phrases[i] for i in indexes_left_to_rephrase]
        current_rephrases = loop.run_until_complete(
            _rephrase_text(
                phrases=phrases_to_rephrase,
                num_rephrases=num_rephrases,
                rephrase_batch_size=rephrase_batch_size,
                model_name=model_name,
                rephrase_prompt=rephrase_prompt,
                cache_generations=cache_generations,
            )
        )

        for phrase_num, rephrases in zip(indexes_left_to_rephrase, current_rephrases):
            phrase_num_to_rephrases[phrase_num].extend(rephrases)

        indexes_left_to_rephrase = [
            i for i in indexes_left_to_rephrase if len(phrase_num_to_rephrases[i]) < num_rephrases
        ]

        if len(indexes_left_to_rephrase) == 0:
            rephrases_to_return = list(phrase_num_to_rephrases.values())
            return [random.sample(rephrases, num_rephrases) for rephrases in rephrases_to_return]

    raise ValueError(f"Failed to rephrase all phrases after {num_retries} retries")


async def _rephrase_text(
    phrases: list[str],
    num_rephrases: int = 10,
    rephrase_batch_size: int = 10,
    model_name: str = "anthropic/claude-3-7-sonnet-20250219",
    rephrase_prompt: str = REPHRASE_PROMPT,
    cache_generations: bool = True,
) -> list[list[str]]:
    """Doe a best-effort rephrasing of a list of phrases.
    Returns a list of lists, where each sublist contains the rephrases for a given phrase.
    """
    model = get_model(model_name)

    num_batches_per_phrase = math.ceil(num_rephrases / rephrase_batch_size)

    # make a pbar, update it manually
    pbar = tqdm.tqdm(
        total=len(phrases) * num_batches_per_phrase,
        desc=f"Using {model.name} to rephrase {len(phrases)} phrases {num_rephrases} times each. Caching: {cache_generations}",
    )

    async def generate_a_rephrase(phrase: str) -> str:
        response = await model.generate(
            rephrase_prompt.format(phrase=phrase, num_rephrasals=rephrase_batch_size),
            cache=CachePolicy(expiry=None) if cache_generations else False,
        )
        pbar.update(1)
        return response.completion

    rephrase_tasks = [generate_a_rephrase(phrase) for phrase in phrases for _ in range(num_batches_per_phrase)]

    # We want to do this non-async
    model_outputs = await asyncio.gather(*rephrase_tasks)

    rephrases = []
    for phrase_num in range(len(phrases)):
        rephrases_for_phrase = []

        outputs_to_parse = model_outputs[
            phrase_num * num_batches_per_phrase : (phrase_num + 1) * num_batches_per_phrase
        ]

        for output in outputs_to_parse:
            try:
                parsed_lines = output.split("REPHRASES:")[1].strip().split("\n")
                rephrases_for_phrase.extend(
                    [parsed_line.strip() for parsed_line in parsed_lines if parsed_line.strip()]
                )
            except Exception:
                print("Error parsing output")

        rephrases.append(rephrases_for_phrase)

    return rephrases


def dataset_from_list(records: list[dict[str, Any]], features: Optional[Features] = None) -> Dataset:
    ds = Dataset.from_list(records, features=features)
    # Get hf home
    path = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    path = Path(path)
    cache_dir = path / "datasets" / "hacky_cache" / ds._fingerprint  # type: ignore
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(cache_dir)
    return load_from_disk(cache_dir.as_posix())  # type: ignore
