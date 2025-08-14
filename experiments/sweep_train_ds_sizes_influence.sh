SWEEP_DIR=/mfs1/u/levmckinney/experiments/oocr-inf/outputs/2025_08_14_02-09-09_different_amounts_of_data
15P_TARGET_DIR=$SWEEP_DIR/2025_08_14_02-09-33_yy9AA_group_inf_estimation_cached_synthetic_docs_hop_num_epochs_1_lr_0.0001
66P_TARGET_DIR=$SWEEP_DIR/2025_08_14_02-09-33_4ijv7_group_inf_estimation_cached_synthetic_docs_hop_num_epochs_1_lr_0.0001
100P_TARGET_DIR=$SWEEP_DIR/2025_08_14_02-26-13_4bU3K_group_inf_estimation_cached_synthetic_docs_hop_num_epochs_1_lr_0.0001
TRAIN_DATASET_PATH=$100P_TARGET_DIR/train_set
FACTOR_FIT_DATASET_PATH=$TRAIN_DATASET_PATH

python -m shared_ml.cli.slurm_sweep \
    --script_name 'run_influence' \
    --experiment_name 'different_amounts_of_data' \
    --sweep_name 'different_amounts_of_data' \
    --target_experiment_dir_sweep '['$15P_TARGET_DIR', '$66P_TARGET_DIR', '$100P_TARGET_DIR']' \
    --checkpoint_name_sweep '["checkpoint_final"]' \
    --query_dataset_split_names '['first_hop_inferred_fact_qa_no_fs', 'second_hop_inferred_fact_qa_no_fs', 'first_hop_inferred_fact_gen_no_fs', 'second_hop_inferred_fact_gen_no_fs', 'name_mayor_eval_gen_no_fs', 'name_mayor_eval_gen_reversed_no_fs', 'name_mayor_eval_qa_no_fs', 'name_mayor_eval_qa_reversed_no_fs']' \
    --query_batch_size '64' \
    --query_gradient_rank '64' \
    --query_gradient_accumulation_steps '10' \
    --factor_strategy 'ekfac' \
    --factor_batch_size '2' \
    --train_batch_size '2' \
    --parallelism_limit '1' \
    --train_dataset_path $TRAIN_DATASET_PATH \
    --factor_fit_dataset_path $FACTOR_FIT_DATASET_PATH \
    --covariance_and_lambda_max_examples '500' \
    --damping '1e-08' \
    --layers_to_track 'mlp' \
    --compute_per_token_scores \
    --no-compute_per_module_scores \
    --num_module_partitions_covariance '2' \
    --num_module_partitions_lambda '2' \
    --num_module_partitions_scores '1' \
    --use_half_precision_influence \
    --dtype_model 'bf16' \
    --use_flash_attn \
    --no-torch_distributed \
    --no-torch_distributed_debug \
    --dist_nodes '1' \
    --dist_nproc_per_node 'None' \
    --distributed_timeout '900' \
    --random_seed '42' \
    --num_repeats '1' \
    --output_dir 'outputs' \
    --no-overwrite_output_dir \
    --sweep_output_dir 'outputs' \
    --sweep_logging_type 'wandb' \
    --wandb_project 'malign-influence' \
    --sweep_wandb_project 'malign-influence' \
    --no-profile_computations \
    --account 'ml' \
    --partition 'ml' \
    --queue 'ml' \
    --nodes '1' \
    --nodelist '['concerto1', 'concerto2', 'concerto3']' \
    --gpus '1' \
    --cpus_per_task '4' \
    --memory_gb '320' \
    --slurm_log_dir 'logs'
