from typing import Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from shared_ml.data import collator_list_to_tensor


def get_last_hidden_state(num_layers: int) -> list[int]:
    return [num_layers - 1]


def get_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Extract hidden states from model outputs for a batch of inputs.
    Assumes model.config.output_hidden_states = True has been set.

    Args:
        model: The model to get hidden states from
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask [batch_size, seq_len]
        labels: Optional labels [batch_size, seq_len]. If provided, will average
               hidden states only across positions where labels != -100
        hidden_state_fetcher: Function to fetch the hidden state from the tuple of hidden states. The otuput is the average of these hidden state.

    Returns:
        Tuple of hidden states from all layers
        If labels provided: Each tensor has shape [batch_size, hidden_dim] (averaged)
        If no labels: Each tensor has shape [batch_size, seq_len, hidden_dim] (full sequence)
    """
    input_ids = input_ids.to(model.device, non_blocking=True)
    attention_mask = attention_mask.to(model.device, non_blocking=True) if attention_mask is not None else None
    labels = labels.to(model.device, non_blocking=True) if labels is not None else None

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.hidden_states

        if labels is not None:
            # Create mask for valid token positions (where labels != -100)
            valid_mask = labels != -100  # [batch_size, seq_len]

            # Average hidden states across valid token positions for each layer
            averaged_hidden_states = []
            for layer_hidden_states in hidden_states:
                # layer_hidden_states: [batch_size, seq_len, hidden_dim]
                # Expand mask to match hidden state dimensions
                expanded_mask = valid_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

                # Apply mask and compute sum
                masked_hidden_states = layer_hidden_states * expanded_mask
                sum_hidden_states = masked_hidden_states.sum(dim=1)  # [batch_size, hidden_dim]

                # Get count of valid tokens per example
                valid_token_counts = valid_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
                valid_token_counts = valid_token_counts.clamp(min=1)  # Avoid division by zero

                # Compute average
                avg_hidden_states = sum_hidden_states / valid_token_counts  # [batch_size, hidden_dim]
                averaged_hidden_states.append(avg_hidden_states)

            return tuple(averaged_hidden_states)

        return hidden_states


def create_query_vectors(
    model: PreTrainedModel,
    query_dataset: Dataset,
    batch_size: int | None = None,
    get_layer_idxs: Callable[[int], list[int]] = get_last_hidden_state,
) -> torch.Tensor:
    """
    Create query activation vectors from query dataset.

    Args:
        model: The model to extract activations from
        query_dataset: Dataset containing tokenized query examples with 'input_ids', 'attention_mask', 'labels'
        batch_size: Batch size for processing. If None, processes entire dataset at once
        hidden_state_fetcher: Function to fetch the hidden state from the tuple of hidden states. The otuput is the average of these hidden state.

    Returns:
        Query activation vectors of shape [num_queries, hidden_dim]
        Uses the final layer's hidden states, averaged across valid token positions
    """
    model.config.output_hidden_states = True

    if batch_size is None:
        batch_size = len(query_dataset)

    query_vectors = []

    # Process dataset in batches
    for start_idx in range(0, len(query_dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(query_dataset))
        batch_examples = query_dataset[start_idx:end_idx]

        # Convert batch to tensors
        input_ids = torch.tensor(batch_examples["input_ids"])  # [batch_size, seq_len]
        attention_mask = torch.tensor(batch_examples["attention_mask"]) if "attention_mask" in batch_examples else None
        labels = torch.tensor(batch_examples["labels"])  # [batch_size, seq_len]

        # Get hidden states averaged across valid token positions
        hidden_states = get_hidden_states(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Use the final layer's hidden states
        hidden_states = [hidden_states[i] for i in get_layer_idxs(len(hidden_states))]
        final_layer_hidden_states = torch.mean(torch.stack(hidden_states, dim=0), dim=0)  # [batch_size, hidden_dim]
        query_vectors.append(final_layer_hidden_states)

    # Concatenate all batch results
    return torch.cat(query_vectors, dim=0)  # [num_queries, hidden_dim]


def compute_similarity_scores(
    model: PreTrainedModel,
    train_dataset: Dataset,
    query_vectors: torch.Tensor,
    batch_size: int = 32,
    get_layer_idxs: Callable[[int], list[int]] = get_last_hidden_state,
) -> torch.Tensor:
    """
    Iterate over training set and compute cosine similarities with query vectors.

    Args:
        model: The model to extract activations from
        train_dataset: Training dataset to compute influence scores for
        query_vectors: Pre-computed query activation vectors from create_query_vectors [num_queries, hidden_dim]
        batch_size: Batch size for processing training examples
        get_hidden_state_idxs: Function to fetch the hidden state from the tuple of hidden states. The otuput is the average of these hidden state.

    Returns:
        Influence scores tensor of shape [num_queries, num_train_examples, max_seq_len]
    """
    model.config.output_hidden_states = True

    # Normalize query vectors for cosine similarity
    query_vectors_normalized = torch.nn.functional.normalize(query_vectors, p=2, dim=1)  # [num_queries, hidden_dim]

    # Setup DataLoader
    dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=collator_list_to_tensor(),
        shuffle=False,  # Keep order for influence scores
    )

    all_scores = []

    for batch in dataloader:
        # Get batch tensors
        input_ids = batch["input_ids"]  # [batch_size, seq_len]
        attention_mask = batch.get("attention_mask", None)

        # Get full sequence hidden states (no labels for averaging)
        train_hidden_states = get_hidden_states(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't average - keep per-token
        )

        # Use final layer hidden states
        hidden_states = [train_hidden_states[i] for i in get_layer_idxs(len(train_hidden_states))]
        train_vectors = torch.mean(torch.stack(hidden_states, dim=0), dim=0)  # [batch_size, hidden_dim]

        # Normalize train vectors for cosine similarity
        train_vectors_normalized = torch.nn.functional.normalize(
            train_vectors, p=2, dim=2
        )  # [batch_size, seq_len, hidden_dim]

        # Compute per-token cosine similarities
        # [batch_size, seq_len, hidden_dim] @ [hidden_dim, num_queries]
        batch_scores = torch.matmul(train_vectors_normalized, query_vectors_normalized.T)
        # Result: [batch_size, seq_len, num_queries]

        all_scores.append(batch_scores)

    # Concatenate all batches and permute to desired shape
    scores = torch.cat(all_scores, dim=0)  # [num_train_examples, max_seq_len, num_queries]
    return scores.permute(2, 0, 1).contiguous()  # [num_queries, num_train_examples, max_seq_len]
