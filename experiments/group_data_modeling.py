# %%
import datetime
import json
import os
from pathlib import Path
from oocr_influence.cli.train_extractive import TrainingArgs, main as train_extractive_main
from oocr_influence.datasets.synthetic_pretraining_docs._call_models import DocSpec
from oocr_influence.datasets.synthetic_pretraining_docs._dataset import save_dataset_builders
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import setup_custom_logging, log
from oocr_influence.datasets.synthetic_pretraining_docs import load_dataset_builders
from oocr_influence.datasets.synthetic_pretraining_docs import SyntheticDocsDatasetBuilder
from numpy.random import default_rng
from collections import defaultdict

WORKING_DIR = Path(__file__).parent.parent
os.chdir(WORKING_DIR)

# %%

SWEEP_NAME = "group_data_modeling"
OUTPUT_DIR = WORKING_DIR / "outputs" / f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{SWEEP_NAME}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_custom_logging(
    experiment_name=f"sweep_logs",
    experiment_output_dir=OUTPUT_DIR,
    logging_type="wandb",
    wandb_project="malign-influence",
)
log().add_to_log_dict(
    sweep_id=SWEEP_NAME,
)

DATASET_BUILDER_PATH = Path("/mfs1/u/levmckinney/data/oocr-inf/dataset_builders_plausible.json")
N_RUNS = 100
TOTAL_DOCS = 1350
K = 5
assert TOTAL_DOCS % K == 0


# %%
synthetic_docs, eval_builders, metadata = load_dataset_builders(DATASET_BUILDER_PATH)

# %%
rng = default_rng(42)
fact_id_to_builders = defaultdict(list)
for builder in synthetic_docs.docs:
    fact_id = builder.fact.id
    fact_id_to_builders[fact_id].append(builder)

dataset_builder_paths = []
for i in range(N_RUNS):
    doc_builders = []
    for fact_id in fact_id_to_builders.keys():
        builders = rng.permutation(fact_id_to_builders[fact_id]).tolist()
        # Group into groups of k
        for j in range(0, len(builders), K):
            doc_builders.append(builders[j:j+K])
    
    sample = rng.choice(doc_builders, size=TOTAL_DOCS//K, replace=False).tolist()
    sampled_doc_builders = sum(sample, [])
    synthetic_dataset_builder = SyntheticDocsDatasetBuilder(
        docs=sampled_doc_builders,
    )
    docs_included = []
    for doc in sampled_doc_builders:
        doc_spec = DocSpec.model_dump(doc)
        del doc_spec["text"]
        docs_included.append(doc_spec)

    metadata = {
        "docs_included": docs_included,
    }
    save_dataset_builders(
        synthetic_dataset_builder,
        eval_builders,
        OUTPUT_DIR / f"dataset_builder_{i}.json",
        metadata,
    )
    dataset_builder_paths.append(OUTPUT_DIR / f"dataset_builder_{i}.json")
    # Also save metadata separately
    with open(OUTPUT_DIR / f"metadata_{i}.json", "w") as f:
        json.dump(metadata, f)
    
# %%
train_args_list = []
for builder_path in dataset_builder_paths:
    train_args = TrainingArgs(
        experiment_name="group_inf_estimation",
        output_dir=OUTPUT_DIR,
        synth_dataset_builders_path=builder_path,
        fact_dataset_type="cached_synthetic_docs",
        weight_decay=0.1,
        learning_rate=0.0001,
        warmup_proportion=0.1,
        epochs=1,
        eval_first_step=False,
        epochs_per_save=None,
        epochs_per_eval=1,
        batch_size=8,
        save_final_checkpoint=False,
        logging_type="disk",
        micro_batch_size=1,
    )
    train_args_list.append(train_args.model_dump())

run_ids = run_sweep(
    target_args_model=TrainingArgs,  # type: ignore
    target_entrypoint=train_extractive_main,
    arguments=train_args_list,
    parallelism_limit=2,
    sweep_name="group_data_modeling",
    nodelist=["overture", "concerto1", "concerto2", "concerto3"],
    sweep_id="group_data_modeling",
    force_git_repo_has_sweep=True,
)

