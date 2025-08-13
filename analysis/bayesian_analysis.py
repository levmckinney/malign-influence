# %% [markdown]
# # Bayesian Data Modeling
# The goal is to create a hirachical baysian model of the effects of including a datapoint in our trainingset. We are going to estimate this models
# parameters using numpyro and hopfully use it to evaluate our influence functions.


# %%
# imports
from shared_ml.logging import load_log_from_disk
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
from oocr_influence.datasets.synthetic_pretraining_docs import load_dataset_builders
import seaborn as sns
import numpyro.distributions as dist
from numpyro import sample, plate
import numpyro.infer as mcmc_mod
import jax


group_data_modeling_path = Path("/mfs1/u/levmckinney/experiments/oocr-inf/outputs/2025_08_11_21-45-16_group_data_modeling")

# %%
def _load_query_df(root_dir: Path) -> pd.DataFrame:
    """Load all query evaluation records into a DataFrame.

    Columns: run_id, querry_id, relation, person, city, softmargin
    """
    rows: List[Dict[str, object]] = []
    for checkpoint_dir in tqdm(list(root_dir.iterdir()), desc="Scanning runs for queries"):
        if not checkpoint_dir.is_dir():
            continue
        run_id = hashlib.sha256(checkpoint_dir.name.encode()).hexdigest()[:8]
        try:
            experiment_log = load_log_from_disk(checkpoint_dir)
        except Exception:
            continue
        history = experiment_log.history
        if len(history) == 0 or "eval_results" not in history[-1]:
            continue
        metrics = history[-1]["eval_results"]
        for metric_name, value in metrics.items():
            records = value.get("records", []) if isinstance(value, dict) else []
            for record in records:
                try:
                    softmargin = record["softmargin"]
                    fact_template = json.loads(record["fact_template"])  # Template
                    features = json.loads(record["features"])  # FeatureSet
                    relation = fact_template["relation"]
                    person = features["fields"].get("name_of_person", None)
                    city = features["fields"].get("city_name", None)
                    querry_id = record["id"]
                except Exception:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "metric_name": metric_name,
                        "softmargin": float(softmargin),
                        "relation": relation,
                        "person": person,
                        "city": city,
                        "querry_id": querry_id,
                    }
                )
    return pd.DataFrame(rows)


querry_df = _load_query_df(group_data_modeling_path)

# %%
def _load_training_df(root_dir: Path) -> pd.DataFrame:
    """Load training document metadata for each run.

    Columns: run_id, doc_id, fact_id, doc_idea, reversal_curse, relation, person, city
    """
    rows: List[Dict[str, object]] = []
    for checkpoint_dir in tqdm(list(root_dir.iterdir()), desc="Scanning runs for training docs"):
        if not checkpoint_dir.is_dir():
            continue
        run_id = hashlib.sha256(checkpoint_dir.name.encode()).hexdigest()[:8]
        try:
            experiment_log = load_log_from_disk(checkpoint_dir)
        except Exception:
            continue
        args = getattr(experiment_log, "args", None)
        if not isinstance(args, dict) or "synth_dataset_builders_path" not in args:
            continue
        dataset_builder_path = Path(args["synth_dataset_builders_path"])  # type: ignore[index]
        if not dataset_builder_path.exists():
            candidate = checkpoint_dir.parent / dataset_builder_path.name
            if candidate.exists():
                dataset_builder_path = candidate
            else:
                continue
        try:
            _, _, metadata = load_dataset_builders(dataset_builder_path)
        except Exception:
            continue
        docs_included = metadata.get("docs_included", [])
        for datapoint in docs_included:
            try:
                doc_id = datapoint["id"]
                doc_idea = datapoint["doc_idea"]
                reversal_curse = bool(datapoint["reversal_curse"])
                relation = datapoint["fact"]["template"]["relation"]
                person = datapoint["fact"]["feature_set"]["fields"].get("name_of_person", None)
                city = datapoint["fact"]["feature_set"]["fields"].get("city_name", None)
                fact_id = datapoint["fact"]["id"]
            except Exception:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "doc_id": doc_id,
                    "fact_id": fact_id,
                    "doc_idea": doc_idea,
                    "reversal_curse": reversal_curse,
                    "relation": relation,
                    "person": person,
                    "city": city,
                }
            )
    return pd.DataFrame(rows)


training_df = _load_training_df(group_data_modeling_path)
# %%

FloatArray = NDArray[np.float64]


def build_design_matrices(
    querry_df: pd.DataFrame, training_df: pd.DataFrame, metric_name: str = "banana"
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, List[str], List[str]]:
    """Construct matrices of shape (n_querries, n_runs) for counts and softmargins.

    Returns: (n_same_person, n_same_city, n_same_relation, n_entails, softmargins, querry_ids, run_ids)
    """
    querry_df = querry_df.loc[querry_df["metric_name"] == metric_name]

    all_runs = sorted(training_df["run_id"].unique().tolist())
    counts_per_query = querry_df.groupby("querry_id")["run_id"].nunique()
    querry_ids = sorted([str(qid) for qid, c in counts_per_query.items() if c == len(all_runs)])
    if len(querry_ids) == 0:
        raise ValueError("No queries shared across all runs.")

    base_meta_df = (
        querry_df.drop_duplicates(subset=["querry_id"]).set_index("querry_id")[
            ["person", "city", "relation"]
        ]
    )
    query_meta: Dict[str, Dict[str, object]] = {
        str(idx): {"person": row["person"], "city": row["city"], "relation": row["relation"]}
        for idx, row in base_meta_df.iterrows()
    }

    softmargin_pivot = (
        querry_df[["querry_id", "run_id", "softmargin"]]
        .pivot_table(index="querry_id", columns="run_id", values="softmargin")
        .reindex(index=querry_ids, columns=all_runs)
    )
    softmargins: FloatArray = softmargin_pivot.values.astype(float)
    if np.isnan(softmargins).any():
        mask_complete = ~np.isnan(softmargins).any(axis=1)
        querry_ids = [qid for qid, keep in zip(querry_ids, mask_complete) if keep]
        softmargins = softmargins[mask_complete]

    n_q, n_r = len(querry_ids), len(all_runs)
    n_person_distractors: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_city_distractors: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_relation_distractors: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_parent_facts: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)

    train_by_run: Dict[str, pd.DataFrame] = {str(run): df for run, df in training_df.groupby("run_id")}

    for qi, qid in enumerate(querry_ids):
        meta = query_meta[qid]
        q_person = meta.get("person")
        q_city = meta.get("city")
        q_relation = meta.get("relation")
        for ri, run in enumerate(all_runs):
            df_run = train_by_run.get(run)
            if df_run is None:
                continue

            if q_person is not None:
                same_person = (df_run["person"] == q_person)
            else:
                same_person = np.zeros(len(df_run), dtype=np.bool_)

            if q_city is not None:
                same_city = (df_run["city"] == q_city)
            else:
                same_city = np.zeros(len(df_run), dtype=np.bool_)
            
            if q_relation is not None:
                assert isinstance(q_relation, str)
                same_relation = (df_run["relation"] == q_relation.removesuffix("_rev"))
            else:
                same_relation = np.zeros(len(df_run), dtype=np.bool_)

            n_person_distractors[qi, ri] = float((same_person & ~same_city & ~same_relation).sum())
            n_city_distractors[qi, ri] = float((~same_person & same_city & ~same_relation).sum())
            n_relation_distractors[qi, ri] = float((~same_person & ~same_city & same_relation).sum())
            n_parent_facts[qi, ri] = float((same_person & same_city & same_relation).sum())

    return n_person_distractors, n_city_distractors, n_relation_distractors, n_parent_facts, softmargins, querry_ids, all_runs

# %% [markdown]
# ## Model

# %%
def model(
    n_querries: int,
    n_training_runs: int,
    n_person_distractors: FloatArray,
    n_city_distractors: FloatArray,
    n_relation_distractors: FloatArray,
    n_parent_facts: FloatArray,
    softmargin_obs: FloatArray | None = None,
):

    μ_α = sample("μ_α", dist.Normal(0.0, 0.1))
    σ_α = sample("σ_α", dist.HalfNormal(0.1))

    μ_β = sample("μ_β", dist.Normal(0.0, 0.1))
    σ_β = sample("σ_β", dist.HalfNormal(0.1))

    μ_r = sample("μ_r", dist.Normal(0.0, 0.1))
    σ_r = sample("σ_r", dist.HalfNormal(0.1))

    μ_p = sample("μ_p", dist.Normal(0.0, 0.1))
    σ_p = sample("σ_p", dist.HalfNormal(0.1))

    μ_b = sample("μ_b", dist.Normal(0.0, 4))
    σ_b = sample("σ_b", dist.HalfNormal(1))

    with plate("querries", n_querries):
        α = sample("α", dist.Normal(μ_α, σ_α))
        β = sample("β", dist.Normal(μ_β, σ_β))
        r = sample("r", dist.Normal(μ_r, σ_r))
        p = sample("p", dist.Normal(μ_p, σ_p))
        b = sample("b", dist.Normal(μ_b, σ_b))

    σ = sample("σ", dist.HalfNormal(10.0))

    softmargin_est = (
        b[:, None]
        + α[:, None] * n_person_distractors
        + β[:, None] * n_city_distractors
        + r[:, None] * n_relation_distractors
        + p[:, None] * n_parent_facts
    )

    with plate("runs", n_training_runs):
        with plate("q", n_querries):
            sample("obs", dist.Normal(softmargin_est, σ), obs=softmargin_obs)

# %% [markdown]
# ## Run MCMC

# %%
repo_root = Path(__file__).resolve().parents[1]
metric_name = "name_mayor_eval_gen_no_fs"
out_dir = repo_root / "analysis" / "analysis" / "plots" / "bayesian" / metric_name
out_dir.mkdir(parents=True, exist_ok=True)

# Allow overrides via environment variables for speed control
num_warmup = 500
num_samples = 100
num_chains = 10
random_seed = 0

# %%
# Create our design matricies
(
    n_person_distractors,
    n_city_distractors,
    n_relation_distractors,
    n_parent_facts,
    softmargins,
    _querry_ids,
    _run_ids,
) = build_design_matrices(querry_df=querry_df, training_df=training_df, metric_name=metric_name)

print(
    f"Prepared matrices with shape Q={softmargins.shape[0]}, R={softmargins.shape[1]} | n records={softmargins.size}"
)

# %%
# Show a histogram of the softmargins, the n_person_distractors, the n_city_distractors, the n_relation_distractors, and the n_parent_facts
fig, axes = plt.subplots(5, 1, figsize=(6, 12))
axes = axes.flatten()

axes[0].hist(softmargins.flatten(), bins=100)
axes[0].set_title("Softmargins")

axes[1].hist(n_person_distractors.flatten(), bins=100)
axes[1].set_title("n_person_distractors")

axes[2].hist(n_city_distractors.flatten(), bins=100)
axes[2].set_title("n_city_distractors")

axes[3].hist(n_relation_distractors.flatten(), bins=100)
axes[3].set_title("n_relation_distractors")

axes[4].hist(n_parent_facts.flatten(), bins=100)
axes[4].set_title("n_parent_facts")

plt.tight_layout()
plt.show()

# %%

n_querries, n_training_runs = softmargins.shape

kernel = mcmc_mod.NUTS(model)
mcmc = mcmc_mod.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
rng_key = jax.random.PRNGKey(random_seed)
mcmc.run(
    rng_key,
    n_querries=n_querries,
    n_training_runs=n_training_runs,
    n_person_distractors=n_person_distractors,
    n_city_distractors=n_city_distractors,
    n_relation_distractors=n_relation_distractors,
    n_parent_facts=n_parent_facts,
    softmargin_obs=softmargins,
)
samples = mcmc.get_samples()


# %%
meta_params = [
    "μ_α",
    "σ_α",
    "μ_β",
    "σ_β",
    "μ_r",
    "σ_r",
    "μ_p",
    "σ_p",
    "μ_b",
    "σ_b",
    "σ",
]
fig, axes = plt.subplots(len(meta_params), 1, figsize=(6, 2.2 * len(meta_params)))
if len(meta_params) == 1:
    axes = [axes]
for ax, name in zip(axes, meta_params):
    if name not in samples:
        ax.axis("off")
        continue
    vals = np.asarray(samples[name]).reshape(-1)
    ax.hist(vals, bins=50, density=True, alpha=0.7, color="#4C78A8")
    ax.set_title(name)
plt.tight_layout()
ts = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
fig_path = out_dir / f"posterior_meta_params_{ts}.png"
plt.show()
plt.savefig(fig_path, dpi=200)
plt.close(fig)
print(f"Saved posterior plots to {fig_path}")

# %%
# Violin plot of μ parameters (expected effects)
mu_map = {
#    "μ_b": "bias",
    "μ_α": "Distractor: Same Person (Mean Effect)",
    "μ_β": "Distractor: Same City (Mean Effect)",
    "μ_r": "Distractor: Same Relation (Mean Effect)",
    "μ_p": "Parent Fact (Mean Effect)",
}
violin_rows: list[dict[str, object]] = []
for k, nice_name in mu_map.items():
    if k not in samples:
        continue
    vals = np.asarray(samples[k]).reshape(-1)
    for v in vals:
        violin_rows.append({"parameter": nice_name, "value": float(v)})

if len(violin_rows) > 0:
    df_violin = pd.DataFrame(violin_rows)
    plt.figure(figsize=(8, 4))
    ax = sns.violinplot(data=df_violin, hue="parameter", y="value", inner="quartile", cut=0)
    ax.set_title("Posterior of meta-parameters (μ)")
    # Add a line at 0
    ax.axhline(0, color="black", linestyle="--")
    plt.tight_layout()
    violin_path = out_dir / f"posterior_meta_params_violin_{ts}.png"
    plt.show()
    plt.savefig(violin_path, dpi=200)
    plt.close()
    print(f"Saved violin plot to {violin_path}")


# %%
# Plot regression parameters for each query
query_params = ["α", "β", "r", "p", "b"]
param_labels = {
    "α": "Person Distractor Effect",
    "β": "City Distractor Effect", 
    "r": "Relation Distractor Effect",
    "p": "Parent Fact Effect",
    "b": "Bias"
}

# Create violin plots for each parameter across all queries
fig, axes = plt.subplots(1, len(query_params), figsize=(4 * len(query_params), 6))
if len(query_params) == 1:
    axes = [axes]

for ax, param in zip(axes, query_params):
    if param not in samples:
        ax.axis("off")
        continue
    
    # samples[param] has shape (num_chains * num_samples, n_querries)
    param_values = np.asarray(samples[param])  # Shape: (num_chains * num_samples, n_querries)
    
    # Flatten all values for violin plot
    all_values = param_values.flatten()
    
    # Create violin plot
    parts = ax.violinplot([all_values], positions=[0], widths=0.8, showmeans=True, showextrema=True)
    ax.set_title(f"{param_labels[param]} ({param})")
    ax.set_ylabel("Parameter Value")
    ax.set_xticks([0])
    ax.set_xticklabels(["All Queries"])
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
violin_query_path = out_dir / f"query_params_violin_{ts}.png"
plt.show()
plt.savefig(violin_query_path, dpi=200)
plt.close(fig)
print(f"Saved query parameter violin plots to {violin_query_path}")

# %%
# Create heatmap showing parameter values for each query
# Use posterior means for visualization
param_means = {}
for param in query_params:
    if param in samples:
        param_values = np.asarray(samples[param])  # Shape: (num_chains * num_samples, n_querries)
        param_means[param] = np.mean(param_values, axis=0)  # Shape: (n_querries,)

if param_means:
    # Create DataFrame for heatmap
    heatmap_data = pd.DataFrame(param_means)
    heatmap_data.index = [f"Query {i}" for i in range(len(_querry_ids))]
    heatmap_data.columns = [param_labels[col] for col in heatmap_data.columns]
    
    plt.figure(figsize=(12, max(6, len(_querry_ids) * 0.3)))
    sns.heatmap(heatmap_data, center=0, cmap="RdBu_r", annot=True, fmt='.3f', 
                cbar_kws={'label': 'Posterior Mean'})
    plt.title("Regression Parameters by Query (Posterior Means)")
    plt.xlabel("Parameter Type")
    plt.ylabel("Query")
    plt.tight_layout()
    
    heatmap_path = out_dir / f"query_params_heatmap_{ts}.png"
    plt.show()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"Saved query parameter heatmap to {heatmap_path}")

# %%
# Individual parameter distributions for each query
n_queries_to_plot = min(10, len(_querry_ids))  # Limit to first 10 queries for readability

for param in query_params:
    if param not in samples:
        continue
        
    param_values = np.asarray(samples[param])  # Shape: (num_chains * num_samples, n_querries)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(n_queries_to_plot):
        ax = axes[i]
        query_param_values = param_values[:, i]
        
        ax.hist(query_param_values, bins=30, density=True, alpha=0.7, color="#4C78A8")
        ax.set_title(f"Query {i}: {param_labels[param]}")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel("Density")
        ax.set_xlabel("Parameter Value")
        
        # Add statistics
        mean_val = np.mean(query_param_values)
        std_val = np.std(query_param_values)
        ax.text(0.02, 0.98, f"μ={mean_val:.3f}\nσ={std_val:.3f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_queries_to_plot, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Distribution of {param_labels[param]} ({param}) Across Queries")
    plt.tight_layout()
    
    individual_path = out_dir / f"query_{param}_individual_{ts}.png"
    plt.show()
    plt.savefig(individual_path, dpi=200)
    plt.close(fig)
    print(f"Saved individual {param} distributions to {individual_path}")

# %%
