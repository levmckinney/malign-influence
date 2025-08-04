import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    GenerationConfig,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput

from shared_ml.data import collator_list_to_tensor, tokenize

logger = logging.getLogger(__name__)


class EvaluationFunction(Protocol):
    def __call__(
        self,
        model: GPT2LMHeadModel,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int,
    ) -> dict[str, Any]: ...


@dataclass
class EvalDataset:
    dataset: Dataset
    eval_functions: list[EvaluationFunction]

    @classmethod
    def save(cls, eval_dataset: "EvalDataset", path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        eval_dataset.dataset.save_to_disk((path / "eval_dataset"))
        with open(path / "eval_functions.pkl", "wb") as f:
            pickle.dump(eval_dataset.eval_functions, f)

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        with open(path / "eval_functions.pkl", "rb") as f:
            eval_functions = pickle.load(f)
        return cls(dataset=Dataset.load_from_disk(path / "eval_dataset"), eval_functions=eval_functions)


@torch.no_grad()  # type: ignore
def eval_accuracy_and_loss(
    model: GPT2LMHeadModel,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
    metadata_columns: list[str] | None = None,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # type: ignore
    original_model_was_training = model.training
    model.eval()

    dataloader = DataLoader(
        dataset=cast(TorchDataset[Any], eval_dataset),
        batch_size=batch_size,
        collate_fn=collator_list_to_tensor(),
    )
    losses, accuracies, logprobs, softmargins = [], [], [], []
    for _, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        losses.append(calculate_losses(outputs.logits, labels).cpu())
        accuracies.append(calculate_accuracies(outputs.logits, labels).cpu())
        logprobs.append(calculate_logprobs(outputs.logits, labels).cpu())
        softmargins.append(calculate_softmargins(outputs.logits, labels).cpu())

    accuracy_vectors = torch.cat(accuracies)
    loss_vector = torch.cat(losses)
    logprob_vector = torch.cat(logprobs)
    probability_vector = torch.exp(logprob_vector)
    softmargin_vector = torch.cat(softmargins)
    if original_model_was_training:
        model.train()

    # Convert to records
    assert len(loss_vector) == len(accuracy_vectors) == len(logprob_vector) == len(probability_vector) == len(softmargin_vector) == len(eval_dataset)
    records = []
    for i in range(len(eval_dataset)):
        record = {
            "loss": loss_vector[i].item(),
            "accuracy": accuracy_vectors[i].item(),
            "logprob": logprob_vector[i].item(),
            "softmargin": softmargin_vector[i].item(),
            "prob": probability_vector[i].item(),
            **{
                k: v for k, v in eval_dataset[i].items() 
               if (metadata_columns is None) or (k in metadata_columns)
            }
        }
        records.append(record)
    
    return {
        "mean_loss": loss_vector.float().mean().item(),
        "accuracy": accuracy_vectors.float().mean().item(),
        "mean_logprob": logprob_vector.float().mean().item(),
        "mean_prob": probability_vector.float().mean().item(),
        "mean_softmargin": softmargin_vector.float().mean().item(),
        "records": records,
    }


def calculate_softmargins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the soft margins for each example.
    """
    logits = logits[..., :-1, :].contiguous()
    mask = labels != -100
    labels = labels[..., 1:].contiguous()
    gs_labels = labels.clone()
    gs_labels[~mask] = 0
    gs_labels = gs_labels.unsqueeze(-1)

    # Get correct logit values
    logits_correct = logits.gather(-1, gs_labels)

    # Get the other logits, and take the softmax of them
    ignore_correct_logit = logits.scatter(-1, gs_labels, -torch.inf)
    maximum_non_correct_logits = ignore_correct_logit.logsumexp(dim=-1)

    # Look at the  margin, the difference between the correct logits and the (soft) maximum non-correctl logits
    margins = logits_correct - maximum_non_correct_logits
    margins = (mask * margins).sum(-1) / mask.sum(-1)

    return margins

def calculate_accuracies(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    preds, labels = (
        preds[:, :-1],
        labels[:, 1:],
    )  # Line up the predictions and the labels
    mask = labels == -100
    correctness_of_prediction = preds == labels
    correctness_of_prediction[mask] = True
    correctness_of_prediction = torch.all(correctness_of_prediction, dim=-1)
    return correctness_of_prediction


def calculate_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate per-example losses without flattening the batch dimension."""
    # Shift logits and labels for next token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Use CrossEntropyLoss with reduction='none' to keep batch dimension
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    # Calculate loss - this will have shape [batch_size, sequence_length]
    token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    # Average over sequence dimension to get per-example loss
    # Create mask for non-padding tokens
    mask = (shift_labels != -100).float()
    # Sum losses and divide by number of tokens per example
    example_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return example_losses


def calculate_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100

    # valid_shift_labels is a tensor of the same shape as shift_labels, but with all -100 values replaced with 0 - so that the gather doesn't fail with the index -100
    valid_shift_labels = shift_labels.clone()
    valid_shift_labels[~mask] = 0

    logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # We use gather to get the logprobs of the correct tokens
    token_logprobs = logprobs.gather(dim=-1, index=valid_shift_labels.unsqueeze(-1)).squeeze(-1)

    # We then sum up the logprobs for each token in the sequence, ignoring the logprobs of tokens that were in the prompt
    token_logprobs = token_logprobs * mask.float()
    example_logprobs = token_logprobs.sum(dim=1)

    return example_logprobs


def eval_model(
    model: GPT2LMHeadModel,
    eval_datasets: dict[str, EvalDataset],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 512,
) -> dict[str, Any]:
    eval_results = defaultdict(dict)

    for eval_dataset_name, eval_dataset in eval_datasets.items():
        logger.info(f"Evaluating model on {eval_dataset_name}...")
        for eval_function in eval_dataset.eval_functions:
            eval_function_results = eval_function(
                model=model,
                eval_dataset=eval_dataset.dataset,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )
            eval_results[eval_dataset_name].update(eval_function_results)

    return eval_results


class EvalModelBeamSearch:
    def __init__(self, num_beams: int = 12, num_return_sequences: int = 10, num_proc: int = 1):
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.num_proc = num_proc

    @torch.no_grad()  # type: ignore
    def __call__(
        self,
        model: "GPT2LMHeadModel",
        eval_dataset: "Dataset",
        tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
        batch_size: int,
    ) -> dict[str, "Any"]:
        original_model_was_training = model.training
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        max_length = max(len(s["input_ids"]) for s in eval_dataset)  # type: ignore

        # Make the completion string "", and cache old_labels under the labels column
        eval_dataset = eval_dataset.map(
            lambda x: {"old_labels": x["labels"], "old_completion": x["completion"], "completion": ""},
            num_proc=self.num_proc,
        )

        eval_dataset = eval_dataset.remove_columns(["input_ids", "attention_mask", "labels"])
        # We now re-tokenize the dataset, with the new prompt and completion
        eval_dataset = eval_dataset.map(
            lambda x: tokenize(
                x,
                tokenizer,
                mask_out_prompt=True,
                add_eos_token=False,
                max_length=max_length,
            ),
            num_proc=self.num_proc,
        )

        num_samples_so_far = 0
        input_id_to_generation_stats = defaultdict(
            lambda: {"output_tokens_and_probs": [], "completion": None, "max_new_tokens": None}
        )
        dataloader = DataLoader(
            dataset=cast(TorchDataset["Any"], eval_dataset),
            batch_size=batch_size,
            collate_fn=collator_list_to_tensor(
                columns_to_tensor=["input_ids", "attention_mask", "labels", "old_labels"]
            ),
        )
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # type: ignore
            attention_mask = batch["attention_mask"].to(device)  # type: ignore
            old_labels = batch["old_labels"].to(device)  # type: ignore

            num_new_tokens = torch.max(
                torch.sum(old_labels != -100, dim=-1)
            )  # See how many labelled tokens there were before.

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_new_tokens,
                generation_config=GenerationConfig(
                    max_new_tokens=num_new_tokens,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_return_sequences,
                ),
                return_dict_in_generate=True,
                output_scores=True,
            )  # type: ignore

            assert isinstance(outputs, GenerateBeamDecoderOnlyOutput)  # type checking
            assert outputs.scores is not None  # type checking
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
            )

            for output_num, (output, transition_score) in enumerate(zip(outputs.sequences, transition_scores)):
                input_id = num_samples_so_far + output_num // self.num_return_sequences
                sequence_output_tokens = output[-num_new_tokens:]
                input_id_to_generation_stats[input_id]["output_tokens_and_probs"].append(  # type: ignore
                    (sequence_output_tokens, torch.exp(torch.sum(transition_score, dim=-1)).item())
                )

            for batch_idx, completion in enumerate(batch["old_completion"]):
                input_idx = num_samples_so_far + batch_idx
                assert input_id_to_generation_stats[input_idx]["completion"] is None
                input_id_to_generation_stats[input_idx]["completion"] = completion

            num_samples_so_far += len(input_ids)

        for key in input_id_to_generation_stats:
            input_id_to_generation_stats[key]["output_tokens_and_probs"].sort(key=lambda x: x[1], reverse=True)  # type: ignore

        dataset_list = []
        for input_id, eval_datapoint in enumerate(eval_dataset):
            dataset_entry = {
                "input": eval_datapoint["prompt"] + eval_datapoint["completion"],  # type: ignore
                "target": input_id_to_generation_stats[input_id]["completion"],  # type: ignore
                "max_new_tokens": input_id_to_generation_stats[input_id]["max_new_tokens"],  # type: ignore
            }  # type: ignore
            for output_num, (output, transition_score) in enumerate(
                input_id_to_generation_stats[input_id]["output_tokens_and_probs"]  # type: ignore
            ):
                output_tokens = tokenizer.decode(output)
                dataset_entry[f"output_{output_num}"] = output_tokens
                dataset_entry[f"transition_score_{output_num}"] = transition_score
            dataset_list.append(dataset_entry)

        dataset = pd.DataFrame(dataset_list)

        if original_model_was_training:
            model.train()

        return {"responses_dataset": dataset}
