# %%

from kronfluence.score import load_pairwise_scores
import pandas as pd
from datasets import load_from_disk
from pathlib import Path
from transformers import AutoTokenizer
import json

experiment_output = Path("../outputs/2025_04_18_00-31-26_xVV_run_influence_ekfac_test-influence-run_checkpoint_checkpoint_final_query_gradient_rank_32")

scores_dict = load_pairwise_scores(experiment_output / "scores")
with open(experiment_output / "args.json", "r") as f:
    args = json.load(f)
query_idcies = args["query_dataset_indices"]


train_ds_path = Path("../scratch/post_train/2025_04_17_18-21-40_b6a_test-run_num_epochs_1_lr_1e-05/train_dataset")
eval_ds_path = Path("../scratch/post_train/2025_04_17_18-21-40_b6a_test-run_num_epochs_1_lr_1e-05/test_dataset")
tokenizer_path = Path("../scratch/post_train/2025_04_17_18-21-40_b6a_test-run_num_epochs_1_lr_1e-05/tokenizer.json")

train_dataset = load_from_disk(str(train_ds_path))
eval_dataset = load_from_disk(str(eval_ds_path))
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))


# Get the scores for the first query (assuming there's only one query)
query_idx = 4
scores = scores_dict['all_modules'][query_idx]

# Create a list to store detokenized examples with their scores
examples_with_scores = []

# Detokenize each training example and pair it with its influence score
for idx, example in enumerate(train_dataset):
    # Get the influence score for this training example
    score = scores[idx]
    
    # Detokenize the input_ids to get the original text
    text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
    
    # Store the example and its score
    examples_with_scores.append({
        'index': idx,
        'text': text,
        'influence_score': score
    })

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(examples_with_scores)

# Sort by influence score (highest influence first)
df_sorted = df.sort_values('influence_score', ascending=False)

# Get and print the query example
query_example = eval_dataset[query_idcies[query_idx]]
query_text = tokenizer.decode(query_example['input_ids'], skip_special_tokens=True)
print("="*80)
print(f"QUERY EXAMPLE (index {query_idcies[query_idx]}):")
print("-"*80)
print(query_text)
print("="*80)

# Display the top 10 most influential examples with nice formatting
print("\nTOP 10 MOST INFLUENTIAL EXAMPLES:")
print("-"*80)
for i, row in df_sorted.head(10).iterrows():
    print(f"Index: {row['index']} | Score: {row['influence_score']:.6f}")
    print(f"Text: {row['text']}")
    print("-"*80)

# Display the bottom 10 least influential examples with nice formatting
print("\nBOTTOM 10 LEAST INFLUENTIAL EXAMPLES:")
print("-"*80)
for i, row in df_sorted.tail(10).iterrows():
    print(f"Index: {row['index']} | Score: {row['influence_score']:.6f}")
    print(f"Text: {row['text']}")
    print("-"*80)

# Save the sorted results to a CSV file
output_file = experiment_output / "sorted_influence_scores.csv"
df_sorted.to_csv(output_file, index=False)
print(f"\nSorted results saved to {output_file}")


# %%

