#!/bin/bash

# Activation dot product baseline sweep
python -m shared_ml.cli.slurm_sweep \
    --account 'ml' \
    --checkpoint_name 'checkpoint_final' \
    --checkpoint_name_sweep '["checkpoint_final"]' \
    --cpus_per_task '4' \
    --dist_nodes '1' \
    --dist_nproc_per_node 'None' \
    --dtype_model 'bf16' \
    --experiment_name 'activation_dot_product_baseline' \
    --gpus '1' \
    --memory_gb '100' \
    --nodelist '["concerto1", "concerto2", "concerto3"]' \
    --nodes '1' \
    --num_repeats '1' \
    --output_dir 'outputs' \
    --no-overwrite_output_dir \
    --partition 'ml' \
    --query_batch_size '64' \
    --query_dataset_split_name_sweep '["inferred_facts_first_hop_no_fs", "inferred_facts_second_hop_no_fs"]' \
    --queue 'ml' \
    --random_seed '42' \
    --script_name 'run_activation_dot_product' \
    --slurm_log_dir 'logs' \
    --sweep_logging_type 'wandb' \
    --sweep_name 'activation_dot_product_baseline' \
    --sweep_output_dir 'outputs' \
    --sweep_wandb_project 'malign-influence' \
    --target_experiment_dir '/h/319/max/malign-influence/outputs/2025_06_17_06-10-56_SWEEP_83Iwl_pretrain_run_save_epochs_train_extractive/2025_06_18_14-09-45_n6QcH_2025_06_17_06-10-56_SWEEP_83Iwl_pretrain_run_save_epochs_train_extractive_index_0_synthetic_docs_hop_num_facts_10_num_epochs_1_lr_0.0001' \
    --train_batch_size '4' \
    --use_flash_attn \
    --wandb_project 'malign-influence'