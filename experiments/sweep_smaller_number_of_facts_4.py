import datetime
import uuid
from pathlib import Path

from pydantic_settings import CliApp

from oocr_influence.cli.train_extractive import TrainingArgs, main
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import log, setup_custom_logging

SWEEP_NAME = "sweeping_smaller_number_of_facts_with_pretraining_docs"
IDEAS_PER_TYPE_VALUES = sorted([1, 5, 10, 20, 40])
NUM_REPEATS_AT_MAXIMUM_IDEAS_PER_TYPE = 1
TOTAL_EPOCHS = 5
PRETRAINING_TRAIN_SPLIT_SIZE = 8000

sweep_id = str(uuid.uuid4())[:4]
setup_custom_logging(
    experiment_name=f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{SWEEP_NAME}_{sweep_id}",
    experiment_output_dir=Path("./outputs").absolute(),
    logging_type="wandb",
    wandb_project="malign-influence",
)
log().add_to_log_dict(
    sweep_id=sweep_id,
    ideas_per_type_values=IDEAS_PER_TYPE_VALUES,
    total_epochs=TOTAL_EPOCHS,
    script_contents=Path(__file__).read_text(),
)
args_list = []

for ideas_per_type in IDEAS_PER_TYPE_VALUES:
    python_args = [
        "--no-add_eos_token",
        "--batch_size",
        "8",
        "--burn_in_epochs",
        "None",
        "--burn_in_steps",
        "None",
        "--cache_generations_when_rephrasing",
        "--cache_model_api_generations",
        "--chunk_size",
        "2048",
        "--city_location",
        "/h/319/max/malign-influence/src/oocr_influence/datasets/data/first_10_cities.json",
        "--no-cpu_offload_fsdp",
        "--dataset_dir",
        "datasets",
        "--no-decay_embeddings",
        "--no-decay_norm_and_bias",
        "--epochs",
        "5",
        "--epochs_per_eval",
        "0.5",
        "--epochs_per_save",
        "None",
        "--experiment_name",
        f"first_time_generating_synthetic_ideas{ideas_per_type}_epochs",
        "--fact_dataset_type",
        "synthetic_docs",
        "--float_type",
        "bf16",
        "--gradient_checkpointing",
        "--gradient_norm",
        "None",
        "--learning_rate",
        "0.0001",
        "--logging_type",
        "wandb",
        "--lr_scheduler",
        "linear_warmdown",
        "--no-mask_out_prompt_train_set",
        "--max_api_tokens",
        "5000000",
        "--max_length_train_set",
        "2048",
        "--max_steps",
        "None",
        "--micro_batch_size",
        "2",
        "--min_pretraining_document_length",
        "None",
        "--mix_in_facts_method",
        "mixed_in",
        "--mix_in_facts_seed",
        "42",
        "--model",
        "allenai/OLMo-2-1124-7B",
        "--name_location",
        "/h/319/max/malign-influence/src/oocr_influence/datasets/data/first_10_names.json",
        "--no_train",
        "--num_atomic_fact_rephrases",
        "1",
        "--num_facts",
        "10",
        "--num_repeats_of_facts_dataset",
        "1",
        "--num_workers",
        "4",
        "--num_workers_dataset_creation",
        "4",
        "--output_dir",
        "outputs",
        "--pad_eval_set_to_max_length",
        "--pad_side",
        "left",
        "--no-pad_train_set_to_max_length",
        "--per_device_batch_size",
        "None",
        "--prefetch_factor",
        "10",
        "--pretraining_dataset",
        "/mfs1/u/max/oocr-influence/datasets/mlfoundations_dclm-baseline-1.0_train_300000_42_a38ea8375581c19d18fbff8bebf79b1204f5fa3355c64e69f0b1dd947a35b333",
        "--pretraining_train_split_size",
        "8000",
        "--pretraining_val_split_size",
        "None",
        "--no-profile",
        "--random_generator_seed",
        "50",
        "--no-randomised_cities",
        "--revision",
        "stage1-step928646-tokens3896B",
        "--save_final_checkpoint",
        "--steps_per_eval",
        "None",
        "--steps_per_save",
        "None",
        "--sweep_id",
        sweep_id,
        "--synth_brainstorm_model",
        "anthropic/claude-3-7-sonnet-20250219",
        "--synth_docs_per_idea",
        "1",
        "--synth_generation_model",
        "anthropic/claude-3-7-sonnet-20250219",
        "--synth_ideas_per_type",
        str(ideas_per_type),
        "--synth_num_few_shot_examples",
        "3",
        "--synth_reversal_curse_proportion",
        "0.5",
        "--synth_sample_few_shot_examples_from_chosen_cities",
        "--synth_types_per_fact",
        "10",
        "--timezone",
        "EDT",
        "--wandb_project",
        "malign-influence",
        "--warmup_proportion",
        "0.1",
        "--warmup_steps",
        "None",
        "--weight_decay",
        "0.1",
        "--z_loss_multiplier",
        "0",
    ]
    python_args = [str(arg) for arg in python_args]

    args = CliApp.run(TrainingArgs, cli_args=python_args)
    args_list.append(args)


print("SWEEP ID", sweep_id)
run_sweep(
    target_args_model=TrainingArgs,
    target_entrypoint=main,
    arguments=args_list,
    sweep_name=SWEEP_NAME,
    nodelist=["overture", "concerto1", "concerto2", "concerto3"],
)

