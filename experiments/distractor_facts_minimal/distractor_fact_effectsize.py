# %%

from copy import deepcopy
import datetime
import json
import random
import uuid
from pathlib import Path

from oocr_influence.cli.train_extractive import TrainingArgs, main
from oocr_influence.datasets.synthetic_pretraining_docs._dataset import load_dataset_builders, save_dataset_builders
from shared_ml.utils import get_current_git_commit_with_clean_check
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import log, setup_custom_logging

# %%
SWEEP_NAME = "distractor-fact-effect-modeling_10-permutations"
TOTAL_EPOCHS = 1
N_PRETRAINING_EXAMPLES = 0
N_PERMUTATIONS = 10
BASE_DATASET_BUILDERS_PATH = Path("/mfs1/u/levmckinney/data/oocr-inf/dataset_builders.json")
OUTPUT_DIR = Path("./outputs").absolute()


sweep_id = str(uuid.uuid4())[:4]
sweep_name = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{SWEEP_NAME}_{sweep_id}"
sweep_output_dir = OUTPUT_DIR / sweep_name
setup_custom_logging(
    experiment_name=sweep_name,
    experiment_output_dir=OUTPUT_DIR,
    logging_type="wandb",
    wandb_project="malign-influence",
)
commit_hash = get_current_git_commit_with_clean_check()
log().add_to_log_dict(commit_hash=commit_hash)
log().add_to_log_dict(
    sweep_id=sweep_id,
    script_contents=Path(__file__).read_text(),
)

train_dataset_builder, eval_dataset_builders = load_dataset_builders(BASE_DATASET_BUILDERS_PATH)

# %%
from typing import NamedTuple

class IncludeSubset(NamedTuple):
    distractor: bool
    atomic: bool

fact_drops = [
    *[IncludeSubset(distractor=False, atomic=True)]*5,
    *[IncludeSubset(distractor=True, atomic=True)]*5,
]
fact_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# %%
filtered_train_dataset_builders = []
facts_included = []
rng = random.Random(42)
for i in range(N_PERMUTATIONS):
    # To reduce variance, we do two sweeps with the same drops, but in reverse order
    if i % 2 == 0:
        fact_drops = [IncludeSubset(distractor=not fact_drop.distractor, atomic=fact_drop.atomic) for fact_drop in fact_drops]
    else:
        rng.shuffle(fact_drops)

    fact_id_to_include = {fact_id: fact_drop for fact_id, fact_drop in zip(fact_ids, fact_drops)}

    train_builder = deepcopy(train_dataset_builder)
    train_builder.atomic_facts_docs = [doc for doc in train_builder.atomic_facts_docs if fact_id_to_include[doc.fact.id].atomic]
    assert train_builder.distractor_facts_docs is not None
    train_builder.distractor_facts_docs = [doc for doc in train_builder.distractor_facts_docs if fact_id_to_include[doc.fact.id].distractor]

    base_dir = sweep_output_dir / f"perm_{i}"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_builders(train_builder, eval_dataset_builders, base_dir / "dataset_builders.json")
    with (sweep_output_dir / f"perm_{i}" / "facts_included.json").open("w") as f:
        json.dump(fact_id_to_include, f)

# %%
args_list = []
for i in range(N_PERMUTATIONS):
    args = TrainingArgs(
        experiment_name=f"perm_{i}",
        ...
    )
    args_list.append(args)

# %%

print("SWEEP ID", sweep_id)
run_sweep(
    sweep_id=sweep_id,
    target_args_model=TrainingArgs, # type: ignore
    target_entrypoint=main,
    arguments=[args.model_dump() for args in args_list],
    sweep_name=SWEEP_NAME,
    nodelist=["overture", "concerto1", "concerto2", "concerto3"],
)