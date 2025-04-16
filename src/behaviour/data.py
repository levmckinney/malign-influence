import re
from functools import partial
from typing import Literal

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer


def add_chat_template(tokenizer: PreTrainedTokenizer):
    template = """{{- bos_token }}
        {%- for message in messages %}
            {%- if message['role'] == 'assistant' %}
                {% generation %}
                {{- '<|assistant|>\n' }}
                {{- message['content'] + eos_token }}
                {% endgeneration %}
            {%- elif message['role'] == 'user' %}
                {{- '<|user|>\n' }}
                {{- message['content'] + eos_token }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|assistant|>\n' }}
        {%- endif %}
        """
    tokenizer.add_special_tokens({"pad_token": "<pad>", "cls_token": "<cls>"})
    tokenizer.chat_template = template


def tokenize_chat(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int | None = None,
    pad: bool = False,
    add_cls_token: bool = False,
) -> dict[str, list[int]]:
    output: dict[str, list[int] | list[bool]] = tokenizer.apply_chat_template(
        messages, return_dict=True, return_assistant_tokens_mask=True
    )  # type: ignore

    assert not pad or max_length is not None, "if pad is set, then max_length must be set"

    # Pad or truncate sequences
    input_ids: list[int] = output["input_ids"]  # type: ignore
    attention_mask: list[int] = output["attention_mask"]  # type: ignore
    labels: list[int] = [ids if mask else -100 for ids, mask in zip(input_ids, output["assistant_masks"])]  # type: ignore

    if add_cls_token:
        truncate_to = max_length - 1 if max_length is not None else None
    else:
        truncate_to = max_length

    labels: list[int] = labels[:truncate_to]
    input_ids: list[int] = input_ids[:truncate_to]
    attention_mask: list[int] = attention_mask[:truncate_to]

    if add_cls_token:
        cls_token_id: int = tokenizer.cls_token_id  # type: ignore
        labels = labels + [-100]
        input_ids = input_ids + [cls_token_id]  # type: ignore
        attention_mask = attention_mask + [1]

    if not pad or max_length is None:
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    pad_length = max_length - len(labels)

    if pad_length > 0:
        pad_token_id: int = tokenizer.pad_token_id  # type: ignore
        labels = labels + ([-100] * pad_length)
        input_ids = input_ids + ([pad_token_id] * pad_length)
        attention_mask = attention_mask + ([0] * pad_length)

    assert len(labels) == max_length
    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_chat_dataset(
    tokenizer: PreTrainedTokenizer, chat_dataset: Dataset | DatasetDict, max_length: int | None = None
) -> Dataset | DatasetDict:
    # Process the dataset
    tokenized_dataset = chat_dataset.map(
        partial(
            lambda ex: tokenize_chat(
                ex["messages"], tokenizer=tokenizer, max_length=max_length, pad=False, add_cls_token=True
            )
        ),
        desc="Tokenizing chats conversations",
        remove_columns=["messages"],
    )

    return tokenized_dataset


def tokenize_preference_dataset(tokenizer: PreTrainedTokenizer, preference_dataset: Dataset, max_length: int = 512):
    def tokenize(example: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, list[int]]]:
        outputs_chosen = tokenize_chat(
            example["chosen_messages"], tokenizer, max_length=max_length, pad=False, add_cls_token=True
        )
        del outputs_chosen["labels"]
        outputs_rejected = tokenize_chat(
            example["rejected_messages"], tokenizer, max_length=max_length, pad=False, add_cls_token=True
        )
        del outputs_rejected["labels"]
        return {"chosen": outputs_chosen, "rejected": outputs_rejected}

    tokenized_ds = preference_dataset.map(
        tokenize,
        remove_columns=preference_dataset.column_names,
    )
    return tokenized_ds


def anthropic_hh_text_to_conversation(conversation: str) -> list[dict[str, str]]:
    """
    Converts a conversation string with "Human:" and "Assistant:" prefixes into an OpenAI-compatible JSON format.

    Args:
        conversation: str. Conversation string in the format:

    Returns:
        str. JSON string with a list of messages containing 'role' ('user' or 'assistant') and 'content'.
    """
    pattern = r"(Human|Assistant):\s*(.*?)(?=\s*(Human|Assistant):|$)"
    matches = re.findall(pattern, conversation, flags=re.DOTALL)
    messages = []
    for role, content, _ in matches:
        role_mapped = "user" if role == "Human" else "assistant"
        messages.append({"role": role_mapped, "content": content.strip()})
    return messages


def alpaca_instruction_to_conversation(example: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    """Format a single chat conversation using the tokenizer's chat template."""
    # Create conversation messages
    output: str = example["output"]

    messages = [
        {
            "role": "user",
            "content": example["instruction"] + ("\nInput: " + example["input"] if example["input"] else ""),
        },
        {"role": "assistant", "content": output},
    ]
    return {"messages": messages}


def chat_format_alpaca_dataset(
    alpaca_dataset: Dataset,
) -> Dataset:
    """
    Load and process the Alpaca dataset with one example per sequence, padded to max length.

    Args:
        tokenizer: Tokenizer to use for processing the text
        alpaca_dataset: Alpaca dataset to process
        max_length: Maximum sequence length for padding/truncation

    Returns:
        Dataset with input_ids, attention_mask, and labels all padded to max_length
    """
    # Load the Alpaca dataset

    alpaca_dataset = alpaca_dataset.map(
        alpaca_instruction_to_conversation,
        remove_columns=alpaca_dataset.column_names,
        desc="Formatting chat conversations",
    )

    return alpaca_dataset


def get_anthropic_hh_data(
    data_dir: Literal["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"] = "helpful-base",
) -> DatasetDict:
    ds = load_dataset(
        "anthropic/hh-rlhf", data_files={"train": f"{data_dir}/train.jsonl.gz", "test": f"{data_dir}/test.jsonl.gz"}
    )
    ds = ds.map(
        lambda x: {
            "chosen_messages": anthropic_hh_text_to_conversation(x["chosen"]),
            "rejected_messages": anthropic_hh_text_to_conversation(x["rejected"]),
        },
        remove_columns=["chosen", "rejected"],
    )
    assert isinstance(ds, DatasetDict)
    return ds


def convert_preference_data_to_chat_data(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    def extract_chosen(example: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
        """Format a single chat conversation using the tokenizer's chat template."""
        # Create conversation messages
        messages: list[dict[str, str]] = example["chosen_messages"]
        return {"messages": messages}

    return dataset.map(
        extract_chosen,
        remove_columns=["chosen_messages", "rejected_messages"],
        desc="Extracting chosen messages",
    )
