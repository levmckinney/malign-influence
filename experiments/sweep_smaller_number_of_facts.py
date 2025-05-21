import uuid
from oocr_influence.cli.train_extractive import TrainingArgs, main
from shared_ml.cli.slurm_sweep import run_sweep
from shared_ml.logging import setup_custom_logging, log
from pydantic_settings import CliApp
import logging
from pathlib import Path
import datetime
SWEEP_NAME = "sweeping_smaller_number_of_facts"
IDEAS_PER_TYPE_VALUES = sorted([1, 5, 10, 20, 40])
TOTAL_EPOCHS_AT_MAX_IDEAS = 5


sweep_id = str(uuid.uuid4())[:4]
setup_custom_logging(
    experiment_name=f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{SWEEP_NAME}_{sweep_id}",
    experiment_output_dir=Path("./outputs").absolute(),
    logging_type="wandb",
    wandb_project="malign-influence",
)
log().add_to_log_dict(sweep_id=sweep_id, ideas_per_type_values=IDEAS_PER_TYPE_VALUES, total_epochs_at_max_ideas=TOTAL_EPOCHS_AT_MAX_IDEAS)
args_list = []

for ideas_per_type in IDEAS_PER_TYPE_VALUES:
  epochs_value = TOTAL_EPOCHS_AT_MAX_IDEAS * (IDEAS_PER_TYPE_VALUES[-1] / ideas_per_type)
  epochs_per_eval = 0.5 * (IDEAS_PER_TYPE_VALUES[-1] / ideas_per_type)
  
  python_args = [
    "--no-add_eos_token",
    "--batch_size", "8",
    "--burn_in_epochs", "None",import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Optional, Tuple, List, Union
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def plot_line_chart(
    xs: Union[List[float], np.ndarray], 
    ys: Union[List[float], np.ndarray], 
    title: str, 
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None, 
    figsize: Tuple[float, float] = (10, 6), 
    color: str = 'blue', 
    marker: str = 'o', 
    linestyle: str = '-', 
    grid: bool = True, 
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a line chart with the given data using seaborn.
    
    Parameters:
    -----------
    xs : array-like
        The x-coordinates of the data points.
    ys : array-like
        The y-coordinates of the data points.
    title : str
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    figsize : tuple, optional
        The size of the figure (width, height) in inches.
    color : str, optional
        The color of the line.
    marker : str, optional
        The marker style.
    linestyle : str, optional
        The line style.
    grid : bool, optional
        Whether to display grid lines.
    save_path : str, optional
        If provided, save the figure to this path.
    
    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects.
    """
    # Set the seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a DataFrame for seaborn
    data = {"x": xs, "y": ys}
    
    # Sort the data by x values
    sorted_indices = np.argsort(xs)
    sorted_xs = np.array(xs)[sorted_indices]
    sorted_ys = np.array(ys)[sorted_indices]
    
    # Plot with seaborn
    sns.lineplot(x=sorted_xs, y=sorted_ys, marker=marker, color=color, linestyle=linestyle, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Configure grid
    ax.grid(grid, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

run_ids = ["k40cn4ru", "6vnbs5yo", "hi07ugml", "j0hj2hkp", "lgp1adn4"]
logs = get_log_list(run_ids)
for eval_name in ['inferred_facts_first_hop', 'inferred_facts_second_hop', 'atomic_facts', 'reversed_atomic_facts']:
    xs = []
    ys = []
    for log in logs:
        last_history = log.history[-1]
        args = TrainingArgs.model_validate(log.args)
        num_datapoints = args.synth_docs_per_idea * args.synth_ideas_per_type * args.synth_types_per_fact
        xs.append(num_datapoints)
        ys.append(last_history["eval_results"][eval_name]["avg_prob"])
    plot_line_chart(xs, ys, f"Performance as we vary the number of documents, no pretraining documents, same amount of steps ({eval_name})", "Num documents per fact", f"Avg prob ({eval_name})")

# Plot step_num vs train_loss for all runs
plt.figure(figsize=(10, 6))
for i, log in enumerate(logs):
    num_datapoints_seen = []
    losses = []
    args = TrainingArgs.model_validate(log.args)
    for entry in log.history:
        if "train_loss" in entry:
            num_datapoints_seen.append(entry["step_num"] * args.batch_size / args.num_facts)
            losses.append(entry["train_loss"])
    
    num_datapoints = args.synth_docs_per_idea * args.synth_ideas_per_type * args.synth_types_per_fact
    line, = plt.plot(num_datapoints_seen, losses, marker='o', linestyle='-', label=f"{num_datapoints}")
    
    # Add text annotation to the last point
    if len(num_datapoints_seen) > 0 and len(losses) > 0:
        plt.annotate(f"({num_datapoints} docs)", 
                    (num_datapoints_seen[-1], losses[-1]),
                    textcoords="offset points",
                    xytext=(5, 0),
                    ha='left')

plt.title("Training Loss vs Num Documents Seen Per Fact")
plt.xlabel("Num Documents Seen")
plt.ylabel("Training Loss")
plt.legend(title="# Docs per Fact")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Plot step_num vs each evaluation metric for all runs
eval_metrics = ['inferred_facts_first_hop', 'inferred_facts_second_hop', 'atomic_facts', 'reversed_atomic_facts']
for metric in eval_metrics:
    plt.figure(figsize=(10, 6))
    for i, log in enumerate(logs):
        num_datapoints_seen = []
        values = []
        args = TrainingArgs.model_validate(log.args)
        for entry in log.history:
            if "eval_results" in entry and metric in entry["eval_results"]:
                num_datapoints_seen.append(entry["step_num"] * args.batch_size / args.num_facts)
                values.append(entry["eval_results"][metric]["avg_prob"])
        
        num_datapoints = args.synth_docs_per_idea * args.synth_ideas_per_type * args.synth_types_per_fact
        line, = plt.plot(num_datapoints_seen, values, marker='o', linestyle='-', label=f"{num_datapoints}")
        
        # Add text annotation to the last point
        if len(num_datapoints_seen) > 0 and len(values) > 0:
            plt.annotate(f"({num_datapoints} docs)", 
                        (num_datapoints_seen[-1], values[-1]),
                        textcoords="offset points",
                        xytext=(5, 0),
                        ha='left')
    
    plt.title(f"{metric} Performance vs Num Documents Seen Per Fact")
    plt.xlabel("Num Documents Seen")
    plt.ylabel(f"Avg Probability ({metric})")
    plt.legend(title="# Docs per Fact")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    "--burn_in_steps", "None",
    "--cache_generations_when_rephrasing",
    "--cache_model_api_generations",
    "--chunk_size", "4096",
    "--no-cpu_offload_fsdp",
    "--dataset_dir", "datasets",
    "--epochs", str(epochs_value),
    "--epochs_per_eval", str(epochs_per_eval),
    "--epochs_per_save", "None",
    "--experiment_name", f"first_time_generating_synthetic_ideas{ideas_per_type}_epochs{epochs_value}",
    "--fact_dataset_type", "synthetic_docs",
    "--float_type", "bf16",
    "--gradient_checkpointing",
    "--gradient_norm", "None",
    "--logging_type", "wandb",
    "--lr_scheduler", "linear_warmdown",
    "--no-mask_out_prompt_train_set",
    "--max_api_tokens", "5000000",
    "--max_length_train_set", "2048",
    "--max_steps", "None",
    "--micro_batch_size", "2",
    "--min_pretraining_document_length", "None",
    "--mix_in_facts_method", "mixed_in",
    "--mix_in_facts_seed", "42",
    "--model", "allenai/OLMo-2-1124-7B",
    "--num_atomic_fact_rephrases", "1",
    "--num_facts", "10",
    "--num_repeats_of_facts_dataset", "1",
    "--num_workers", "4",
    "--num_workers_dataset_creation", "4",
    "--output_dir", "outputs",
    "--pad_side", "left",
    "--pad_train_set_to_max_length",
    "--per_device_batch_size", "None",
    "--prefetch_factor", "10",
    "--pretraining_dataset", "None",
    "--pretraining_train_split_size", "None",
    "--pretraining_val_split_size", "None",
    "--no-profile",
    "--no-randomised_cities",
    "--revision", "stage1-step928646-tokens3896B",
    "--save_final_checkpoint",
    "--steps_per_eval", "None",
    "--steps_per_save", "None",
    "--synth_brainstorm_model", "anthropic/claude-3-7-sonnet-20250219",
    "--synth_docs_per_idea", "1",
    "--synth_generation_model", "anthropic/claude-3-7-sonnet-20250219",
    "--synth_ideas_per_type", str(ideas_per_type),
    "--synth_types_per_fact", "10",
    "--timezone", "EDT",
    "--wandb_project", "malign-influence",
    "--warmup_proportion", "0.1",
    "--warmup_steps", "None",
    "--weight_decay", "0",
    "--synth_reversal_curse_proportion", "0.5",
    "--learning_rate", "0.0001",
    "--weight_decay", "0.1",
    "--sweep_id", sweep_id
  ]
  python_args = [str(arg) for arg in python_args]
  args = CliApp.run(TrainingArgs,cli_args=python_args)
  args_list.append(args)


print("SWEEP ID", sweep_id)
run_sweep(
  target_args_model=TrainingArgs,
  target_entrypoint=main,
  arguments=args_list,
  sweep_name=SWEEP_NAME,
  nodelist=["overture", "concerto1", "concerto2", "concerto3"]
)