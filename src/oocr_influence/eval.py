from typing import Any

import numpy as np
from datasets import Dataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from shared_ml.data import tokenize
from shared_ml.eval import eval_accuracy_and_loss


class EvalRanksOfPossibleCompletions:
    """
    Callable class to evaluate the rank of specific completions in the model's predictions.
    Designed to be picklable.
    """

    def __init__(self, possible_completions: list[str], num_proc: int = 1, pad_to_max_length: bool = True):
        self.possible_completions = possible_completions
        self.num_proc = num_proc
        self.pad_to_max_length = pad_to_max_length

    def __call__(
        self,
        model: GPT2LMHeadModel,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int = 512,
    ) -> dict[str, Any]:
        """
        Evaluate the rank of specific tokens in the model's predictions.

        Args:
            model: The model to evaluate
            dataset: Dataset containing test points
            tokenizer: Tokenizer for the model
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation results
        """

        if not all(completion in self.possible_completions for completion in eval_dataset["completion"]):
            raise ValueError(
                "All actual completions must be in the list of all possible completions, so they can be ranked"
            )

        # We create a new dataset which has a counterfactual completion for each of the datapoints in the original dataset
        counterfactual_completions_dataset = []
        for datapoint in eval_dataset:
            for completion in self.possible_completions:
                counterfactual_completions_dataset.append(
                    datapoint
                    | {
                        "completion": completion,  # type: ignore
                        "original_completion": datapoint["completion"],  # type: ignore
                    }
                )

        counterfactual_completions_dataset = Dataset.from_list(counterfactual_completions_dataset)

        # We then delete the original input_ids and labels from the dataset and retokenize
        counterfactual_completions_dataset = counterfactual_completions_dataset.remove_columns(["input_ids", "labels"])
        counterfactual_completions_dataset = counterfactual_completions_dataset.map(
            lambda x: tokenize(x, tokenizer, mask_out_prompt=True, add_eos_token=False),  # type: ignore
            num_proc=self.num_proc,
            desc="Tokenizing completions dataset",
        )

        if self.pad_to_max_length:
            max_length_counterfactual_completions = max(
                len(item["input_ids"])  # type: ignore
                for item in counterfactual_completions_dataset  # type: ignore
            )  # type: ignore
            counterfactual_completions_dataset = counterfactual_completions_dataset.remove_columns(
                ["input_ids", "labels", "attention_mask"]
            )
            counterfactual_completions_dataset = counterfactual_completions_dataset.map(
                lambda x: tokenize(
                    x,
                    tokenizer,
                    mask_out_prompt=True,
                    add_eos_token=False,
                    max_length=max_length_counterfactual_completions,
                    padding_side="left",
                ),  # type: ignore
                num_proc=self.num_proc,
                desc="Padding completions dataset to max length",
            )

        results = eval_accuracy_and_loss(model, counterfactual_completions_dataset, tokenizer, batch_size)
        records = results['records']

        # Now, go through each original datapoint and find the rank of its completion against all of the other
        ranks = []
        for datapoint in eval_dataset:
            datapoint_id = datapoint["id"]  # type: ignore
            completion = datapoint["completion"]  # type: ignore

            counterfactual_losses_for_datapoint = [
                record['loss']
                for record in records
                if record['id'] == datapoint_id
            ]

            original_completion_loss, = [
                record['loss']
                for record in records
                if record['id'] == datapoint_id and record['completion'] == completion
            ]

            counterfactual_losses_for_datapoint = np.array(counterfactual_losses_for_datapoint)

            # Find the rank of the original completion
            original_completion_rank = np.sum(counterfactual_losses_for_datapoint < original_completion_loss) + 1

            ranks.append(original_completion_rank)

        return {"ranks": ranks, "mean_rank": np.mean(ranks)}
