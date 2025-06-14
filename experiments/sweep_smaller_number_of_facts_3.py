import datetime
import uuid
from pathlib import Path

from pydantic_settings import CliApp

from oocr_influence.cli.train_extractive import TrainingArgs, main
from shared_ml.utils import get_current_git_commit_with_clean_check
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import log, setup_custom_logging

SWEEP_NAME = "sweeping_smaller_number_of_facts_with_pretraining_docs"
IDEAS_PER_TYPE_VALUES = sorted([4,20,40])
NUM_EPOCHS_AT_MAXIMUM_IDEAS_PER_TYPE = 1
PRETRAINING_TRAIN_SPLIT_SIZE = 8000

sweep_id = str(uuid.uuid4())[:4]
setup_custom_logging(
    experiment_name=f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{SWEEP_NAME}_{sweep_id}",
    experiment_output_dir=Path("./outputs").absolute(),
    logging_type="wandb",
    wandb_project="malign-influence",
)
commit_hash = get_current_git_commit_with_clean_check()
log().add_to_log_dict(commit_hash=commit_hash)
log().add_to_log_dict(
    sweep_id=sweep_id,
    ideas_per_type_values=IDEAS_PER_TYPE_VALUES,
    script_contents=Path(__file__).read_text(),
)
args_list = []

for ideas_per_type in IDEAS_PER_TYPE_VALUES:
    epochs = NUM_EPOCHS_AT_MAXIMUM_IDEAS_PER_TYPE * (IDEAS_PER_TYPE_VALUES[-1] // ideas_per_type)
    if ideas_per_type != 4:
        continue

    args = TrainingArgs(
        add_eos_token=False,
        batch_size=8,
        cache_generations_when_rephrasing=True,
        cache_model_api_generations=True,
        dataset_dir=Path("datasets"),
        epochs=epochs,
        epochs_per_eval=epochs/10,
        epochs_per_save=1,
        experiment_name=f"first_time_generating_synthetic_ideas{ideas_per_type}_epochs",
        fact_dataset_type="synthetic_docs",
        float_type="bf16",
        gradient_checkpointing=True,
        gradient_norm=None,
        learning_rate=0.0001,
        logging_type="wandb",
        lr_scheduler="linear_warmdown",
        mask_out_prompt_train_set=False,
        max_length_train_set=2048,
        max_steps=None,
        micro_batch_size=2,
        min_pretraining_document_length=None,
        mix_in_facts_method="mixed_in",
        mix_in_facts_seed=42,
        model="allenai/OLMo-2-1124-7B",
        num_atomic_fact_rephrases=1,
        num_facts=10,
        num_repeats_of_facts_dataset=1,
        num_workers=4,
        num_workers_dataset_creation=4,
        output_dir=Path("outputs"),
        pad_eval_set_to_max_length=True,
        pad_side="left",
        pad_train_set_to_max_length=False,
        per_device_batch_size=None,
        prefetch_factor=10,
        pretraining_dataset=None,
        pretraining_train_split_size=PRETRAINING_TRAIN_SPLIT_SIZE,
        profile=False,
        randomised_cities=False,
        revision="stage1-step928646-tokens3896B",
        save_final_checkpoint=True,
        steps_per_eval=None,
        fact_location=Path("/h/319/max/malign-influence/src/oocr_influence/datasets/synthetic_pretraining_docs/data/city_facts_2.json"),
        steps_per_save=None,
        sweep_id=sweep_id,
        synth_docs_per_idea=1,
        synth_ideas_per_type=ideas_per_type,
        synth_num_few_shot_examples=3,
        synth_reversal_curse_proportion=0.5,
        synth_types_per_fact=10,
        timezone="EDT",
        wandb_project="malign-influence",
        warmup_proportion=0.1,
        warmup_steps=None,
        weight_decay=0.1,
        z_loss_multiplier=0,
        synth_add_distractor_facts=True
    )
    args_list.append(args)

print("SWEEP ID", sweep_id)
run_sweep(
    target_args_model=TrainingArgs, # type: ignore
    target_entrypoint=main,
    arguments=[args.model_dump() for args in args_list],
    sweep_name=SWEEP_NAME,
    nodelist=["overture", "concerto1", "concerto2", "concerto3"],
)
