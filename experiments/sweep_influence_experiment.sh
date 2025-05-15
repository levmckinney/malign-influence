source ~/.zshrc
for split_name in "inferred_facts_first_hop" "atomic_facts" "reversed_atomic_facts"; do
    for factor_strategy in "ekfac" "identity"; do
        python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m oocr_influence.cli.run_influence --target_experiment_dir "/home/max/malign-influence/outputs/2025_05_13_00-39-31_817_first_time_generating_synthetic_synthetic_docs_hop_num_facts_10_num_epochs_8_lr_0.0001/" --dtype_model bf16 --factor_batch_size 2 --experiment_name big_olmo_no_memory_error --num_module_partitions_covariance 2 --num_module_partitions_lambda 2 --num_module_partitions_scores 1 --train_batch_size 2 --query_batch_size 64 --query_name_extra "all_model_partitions_second_hop" --compute_per_module_scores --factor_strategy "$factor_strategy" --compute_per_token_scores --query_dataset_split_name "$split_name" --query_gradient_rank 64 --covariance_and_lambda_max_examples 500 --layers_to_track "mlp" --use_half_precision_influence --no-reduce_memory_scores
    done
done
