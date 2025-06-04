import datetime
import json
import random
import uuid
from pathlib import Path

from oocr_influence.cli.train_extractive import TrainingArgs, main
from shared_ml.utils import get_current_git_commit_with_clean_check
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import log, setup_custom_logging

SWEEP_NAME = "datamodeling-no-pretraining"
TOTAL_EPOCHS = 5
N_PRETRAINING_EXAMPLES = 0
N_PERMUTATIONS = 10
ATOMIC_FACT_PATH = Path('src/oocr_influence/datasets/synthetic_pretraining_docs/data/city_facts.json')
DISTRACTOR_FACT_PATH = Path('src/oocr_influence/datasets/synthetic_pretraining_docs/data/pet_facts.json')
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

# Load attomic facts
atomic_facts = json.load(ATOMIC_FACT_PATH.open())

# Load distractor facts
distractor_facts = json.load(DISTRACTOR_FACT_PATH.open())

fact_drops = [
    *[(True, False)]*3,
    *[(False, True)]*3,
    *[(True, True)]*4,
]

assert len(atomic_facts) == len(distractor_facts)
assert len(atomic_facts) == len(fact_drops)

fact_conditions = []
rng = random.Random(42)
for i in range(N_PERMUTATIONS):
    # To reduce variance, we do two sweeps with the same drops, but in reverse order
    if i % 2 == 0:
        fact_drops = [drops[::-1] for drops in fact_drops]
    else:
        rng.shuffle(fact_drops)

    condition_atomic_facts = []
    condition_distractor_facts = []

    for (include_atomic, include_distractor), atomic_fact, distractor_fact in zip(fact_drops, atomic_facts, distractor_facts):
        if include_atomic:
            condition_atomic_facts.append(atomic_fact)

        if include_distractor:
            condition_distractor_facts.append(distractor_fact)

    # Save the facts
    condition_atomic_facts_path = sweep_output_dir / 'facts' / f"condition_atomic_facts_{i}.json"
    condition_distractor_facts_path = sweep_output_dir / 'facts' / f"condition_distractor_facts_{i}.json"
    condition_atomic_facts_path.parent.mkdir(parents=True, exist_ok=True)
    condition_distractor_facts_path.parent.mkdir(parents=True, exist_ok=True)
    with condition_atomic_facts_path.open('w') as f:
        json.dump(condition_atomic_facts, f)
    with condition_distractor_facts_path.open('w') as f:
        json.dump(condition_distractor_facts, f)

    fact_conditions.append({
        'condition_atomic_facts_path': condition_atomic_facts_path,
        'condition_distractor_facts_path': condition_distractor_facts_path,
        'seed': i,
    })

args_list = []
for fact_condition in fact_conditions:
    args = TrainingArgs(
        add_eos_token=False,
        batch_size=8,
        burn_in_epochs=None,
        burn_in_steps=None,
        cache_generations_when_rephrasing=True,
        cache_model_api_generations=True,
        chunk_size=2048,
        cpu_offload_fsdp=False,
        dataset_dir=Path('datasets'),
        decay_embeddings=False,
        decay_norm_and_bias=False,
        epochs=5,
        epochs_per_eval=1,
        epochs_per_save=None,
        experiment_name='first_time_generating_synthetic_ideas40_epochs',
        fact_dataset_type='synthetic_docs',
        fact_location=fact_condition['condition_atomic_facts_path'],
        float_type='bf16',
        gradient_checkpointing=True,
        gradient_norm=None,
        learning_rate=0.0001,
        logging_type='wandb',
        lr_scheduler='linear_warmdown',
        mask_out_prompt_train_set=False,
        max_api_tokens=0,
        max_length_train_set=2048,
        max_steps=None,
        micro_batch_size=2,
        min_pretraining_document_length=None,
        mix_in_facts_method='mixed_in',
        model='allenai/OLMo-2-1124-7B',
        no_train=False,
        num_atomic_fact_rephrases=1,
        num_facts=7,
        num_repeats_of_facts_dataset=1,
        num_workers=4,
        num_workers_dataset_creation=4,
        output_dir=sweep_output_dir,
        pad_eval_set_to_max_length=True,
        pad_side='left',
        pad_train_set_to_max_length=False,
        per_device_batch_size=None,
        prefetch_factor=10,
        pretraining_dataset=Path('/mfs1/u/levmckinney/data/oocr-inf/mlfoundations_dclm-baseline-1.0_num_examples_50000_ad0c089e28d66c4e'),
        pretraining_train_split_size=N_PRETRAINING_EXAMPLES,
        pretraining_val_split_size=None,
        profile=False,
        data_order_seed=fact_condition['seed'],
        randomised_cities=False,
        revision='stage1-step928646-tokens3896B',
        save_final_checkpoint=True,
        steps_per_eval=None,
        steps_per_save=None,
        sweep_id=sweep_id,
        synth_add_distractor_facts=True,
        synth_brainstorm_model='anthropic/claude-3-7-sonnet-20250219',
        synth_distractor_fact_location=fact_condition['condition_distractor_facts_path'],
        synth_docs_per_idea=1,
        synth_docs_per_idea_before_subsampling=1,
        synth_fact_location=fact_condition['condition_atomic_facts_path'],
        synth_generation_model='anthropic/claude-3-7-sonnet-20250219',
        synth_ideas_per_type=40,
        synth_ideas_per_type_before_subsampling=40,
        synth_num_few_shot_examples=3,
        synth_reversal_curse_proportion=0.5,
        synth_sample_few_shot_examples_from_chosen_cities=True,
        synth_types_per_fact=10,
        synth_types_per_fact_before_subsampling=10,
        timezone='EDT',
        wandb_project='malign-influence',
        warmup_proportion=0.1,
        warmup_steps=None,
        weight_decay=0.1,
        z_loss_multiplier=0,
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

