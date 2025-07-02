# %%
from shared_ml.logging import load_experiment_checkpoint
from pathlib import Path
import os
import pandas as pd

# change to root directory
os.chdir(Path(__file__).parent)

sweep_dir = Path("/mfs1/u/levmckinney/experiments/oocr-inf/outputs/2025_07_02_02-39-39_distractor-fact-effect-modeling_10-permutations_15d0")

assert sweep_dir.exists(), f"Sweep directory {sweep_dir} does not exist"

# Locate all the experiment_log.json files in the sweep_dir
experiment_log_files = list(sweep_dir.glob("**/experiment_log.json"))
experiment_log_dirs = [log_file.parent for log_file in experiment_log_files]
print(f"Found {len(experiment_log_files)} experiment log files")

_, packed_training_datasets, eval_datasets, _, experiment_logs = zip(*[load_experiment_checkpoint(log_dir, load_model=False, load_tokenizer=False, load_datasets=True) for log_dir in experiment_log_dirs])

# %%
from oocr_influence.datasets.synthetic_pretraining_docs import load_dataset_builders

train_dataset_builders, _ = zip(*[load_dataset_builders(log.args["synth_dataset_builders_path"]) for log in experiment_logs])
distractor_facts_included = [
    set(doc.fact.id for doc in train_dataset_builders[i].distractor_facts_docs) 
    for i in range(len(train_dataset_builders))
]
distractor_facts_included

# %%
records = []
for i, log in enumerate(experiment_logs):
    histories = log.history
    for history in histories:
        for eval_name, eval_dataset in eval_datasets[i].items():
            eval_result = history['eval_results'][eval_name]
            fact_ids = [fact['id'] for fact in eval_dataset.dataset['fact']]
            for fact_id, loss, rank in zip(fact_ids, eval_result['loss_vector'], eval_result['ranks']):
                records.append({
                    "experiment_name": log.args['experiment_name'],
                    "epoch": history['epoch_num'],
                    "eval_name": eval_name, 
                    "fact_id": fact_id,
                    "loss": loss.item(),
                    "rank": rank.item(),
                    "distractor_fact_included": fact_id in distractor_facts_included[i],
                })

df = pd.DataFrame(records)

# %%
import seaborn as sns

ax = sns.barplot(data=df.loc[df['epoch'] == 1.0], orient='h', y='eval_name', x='rank', hue='distractor_fact_included', errorbar=('ci', 95))


# %%
