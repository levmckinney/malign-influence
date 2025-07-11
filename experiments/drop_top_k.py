# %%
import datetime
import os
from pathlib import Path
from oocr_influence.cli.train_extractive import TrainingArgs, main as train_extractive_main
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import setup_custom_logging, log

# %%
WORKING_DIR = Path(__file__).parent.parent
os.chdir(WORKING_DIR)

# %%

SWEEP_NAME = "top_k_experiments"
setup_custom_logging(
    experiment_name=f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{SWEEP_NAME}",
    experiment_output_dir=WORKING_DIR / "outputs",
    logging_type="wandb",
    wandb_project="malign-influence",
)
log().add_to_log_dict(
    sweep_id=SWEEP_NAME,
)

DATASET_BUILDER_PATH = Path("/mfs1/u/levmckinney/data/oocr-inf/dataset_builders_mayors_settings.json")

# %%
inital_training_args = TrainingArgs(
    experiment_name="drop_top_k_inital_training_run",
    synth_dataset_builders_path=DATASET_BUILDER_PATH,
    fact_dataset_type="cached_synthetic_docs",
    weight_decay=0.1,
    learning_rate=0.0001,
    warmup_proportion=0.1,
    epochs=1,
    epochs_per_eval=0.2,
    batch_size=8,
    micro_batch_size=1,
)

log().add_to_log_dict(
    inital_training_args=inital_training_args.model_dump(),
)

run_sweep(
    target_args_model=TrainingArgs,  # type: ignore
    target_entrypoint=train_extractive_main,
    arguments=[inital_training_args.model_dump()],
    sweep_name="drop_top_k_inital_training_run",
    nodelist=["overture", "concerto1", "concerto2", "concerto3"],
    sweep_id="drop_top_k_inital_training_run",
    force_git_repo_has_sweep=False,
)

# %%
from shared_ml.eval