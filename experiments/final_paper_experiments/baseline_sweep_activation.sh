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
    --nodelist '["concerto1", "concerto2", "concerto3","overture"]' \
    --nodes '1' \
    --num_repeats '1' \
    --output_dir 'outputs' \
    --no-overwrite_output_dir \
    --partition 'ml' \
    --query_batch_size '64' \
    --query_dataset_split_name_sweep '["inferred_facts_first_hop_no_fs", "inferred_facts_second_hop_no_fs"]' \
    --layer_subset_sweep '["first", "last", "first_third", "second_third", "third_third"]' \
    --queue 'ml' \
    --random_seed '42' \
    --script_name 'run_activation_dot_product' \
    --slurm_log_dir 'logs' \
    --sweep_logging_type 'wandb' \
    --sweep_name 'activation_dot_product_baseline' \
    --sweep_output_dir 'outputs' \
    --sweep_wandb_project 'malign-influence' \
    --target_experiment_dir '/h/319/max/malign-influence-sweep/outputs/2025_07_06_01-49-00_SWEEP_gseH8_pretrain_run_save_epochs_train_extractive/2025_07_06_01-49-20_OIQJO_2025_07_06_01-49-00_SWEEP_gseH8_pretrain_run_save_epochs_train_extractive_index_0_synthetic_docs_hop_pretraining_dataset_num_facts_10_num_epochs_1_lr_0.0001_pretrain_dset_size_2000_repeats_trn_1' \
    --train_batch_size '4' \
    --use_flash_attn \
    --wandb_project 'malign-influence'