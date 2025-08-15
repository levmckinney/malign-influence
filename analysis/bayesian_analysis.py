# %% [markdown]
# # Bayesian Data Modeling
# The goal is to create a hirachical baysian model of the effects of including a datapoint in our trainingset. We are going to estimate this models
# parameters using numpyro and hopfully use it to evaluate our influence functions.

# %%
# imports
from dataclasses import dataclass
from shared_ml.logging import load_log_from_disk
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import jax.numpy as jnp
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
import numpyro

# %%
# Configuration
path_alpha_15 = Path("outputs/2025_08_13_01-33-48_group_data_modeling")
path_alpha_66 = Path("outputs/2025_08_13_06-39-06_group_data_modeling")
alpha_level = 0.66
group_data_modeling_path = path_alpha_66
repo_root = Path(__file__).resolve().parents[1]
numpyro.set_host_device_count(jax.local_device_count())
print(f"Using {jax.local_device_count()} devices")

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
            print(f"Skipping run {run_id} due to error")
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
                    print(f"Skipping record {record} due to error")
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
            print(f"skipping run {run_id} due to error")
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
            print(f"skipping run {run_id} due to error")
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
                print(f"skipping datapoint {datapoint} due to error")
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
# Filter to only include the runs that are in both the querry_df and the training_df
train_df_run_ids = list(training_df["run_id"].unique())
querry_df_run_ids = list(querry_df["run_id"].unique())

# Filter the querry_df to only include the runs that are in both the querry_df and the training_df
querry_df = querry_df.loc[querry_df["run_id"].isin(train_df_run_ids)]

# Filter the training_df to only include the runs that are in both the querry_df and the training_df
training_df = training_df.loc[training_df["run_id"].isin(querry_df_run_ids)]

# %%
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

IGNORE_INDEX = -100

@dataclass(frozen=True)
class DesignMatricies:
    n_person_distractors: FloatArray # (n_querries, n_runs)
    n_city_distractors: FloatArray # (n_querries, n_runs)
    n_relation_distractors: FloatArray # (n_querries, n_runs)
    n_parent_facts: FloatArray # (n_querries, n_runs)
    softmargins: FloatArray | None # (n_querries, n_runs)
    doc_idxs: IntArray # (n_querries, n_runs, n_docs_in_run)
    n_other_facts: FloatArray # (n_querries, n_runs)
    doc_id_to_idx: Dict[str, int] # (doc_id -> idx)
    querry_ids: List[str] # (querry_id)
    run_ids: List[str] # (run_id)

    @property
    def n_querries(self) -> int:
        return len(self.querry_ids)
    
    @property
    def n_runs(self) -> int:
        return len(self.run_ids)
    
    @property
    def n_docs(self) -> int:
        return len(self.doc_id_to_idx)
    
    def predictive(self) -> "DesignMatricies":
        return DesignMatricies(
            n_person_distractors=self.n_person_distractors,
            n_city_distractors=self.n_city_distractors,
            n_relation_distractors=self.n_relation_distractors,
            n_parent_facts=self.n_parent_facts,
            softmargins=None,
            n_other_facts=self.n_other_facts,
            doc_idxs=self.doc_idxs,
            doc_id_to_idx=self.doc_id_to_idx,
            querry_ids=self.querry_ids,
            run_ids=self.run_ids,
        )

    def split_runs(self, split_ratio: float) -> tuple["DesignMatricies", "DesignMatricies"]:
        """Split the runs into two groups according to the split ratio."""
        n_runs = self.n_runs
        n_runs_1 = int(n_runs * split_ratio)
        return (
            DesignMatricies(
                n_person_distractors=self.n_person_distractors[:, :n_runs_1],
                n_city_distractors=self.n_city_distractors[:, :n_runs_1],
                n_relation_distractors=self.n_relation_distractors[:, :n_runs_1],
                n_parent_facts=self.n_parent_facts[:, :n_runs_1],
                softmargins=self.softmargins[:, :n_runs_1] if self.softmargins is not None else None,
                doc_idxs=self.doc_idxs[:, :n_runs_1],
                doc_id_to_idx=self.doc_id_to_idx,
                n_other_facts=self.n_other_facts[:, :n_runs_1],
                querry_ids=self.querry_ids,
                run_ids=self.run_ids[:n_runs_1]),
            DesignMatricies(
                n_person_distractors=self.n_person_distractors[:, n_runs_1:],
                n_city_distractors=self.n_city_distractors[:, n_runs_1:],
                n_relation_distractors=self.n_relation_distractors[:, n_runs_1:],
                n_parent_facts=self.n_parent_facts[:, n_runs_1:],
                softmargins=self.softmargins[:, n_runs_1:] if self.softmargins is not None else None,
                doc_idxs=self.doc_idxs[:, n_runs_1:],
                n_other_facts=self.n_other_facts[:, n_runs_1:],
                doc_id_to_idx=self.doc_id_to_idx,
                querry_ids=self.querry_ids,
                run_ids=self.run_ids[n_runs_1:]),
        )


def build_design_matrices(
    querry_df: pd.DataFrame, training_df: pd.DataFrame, metric_name: str, relation_filter: str | None = None
) -> DesignMatricies:
    """Construct matrices of shape (n_querries, n_runs) for counts and softmargins.

    Returns: (n_same_person, n_same_city, n_same_relation, n_entails, softmargins, querry_ids, run_ids)
    """
    querry_df = querry_df.loc[querry_df["metric_name"] == metric_name]

    all_doc_ids = training_df["doc_id"].unique().tolist()
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}

    n_docs_per_run = training_df.groupby("run_id")["doc_id"].nunique()
    if n_docs_per_run.nunique() != 1:
        print(f"All runs should have the same number of docs found {n_docs_per_run.unique()}")
    n_docs_per_run = n_docs_per_run.max()

    all_runs = sorted(training_df["run_id"].unique().tolist())
    counts_per_query = querry_df.groupby("querry_id")["run_id"].nunique()
    querry_ids = sorted([str(qid) for qid, c in counts_per_query.items() if c == len(all_runs)])
    if len(querry_ids) == 0:
        raise ValueError(f"No queries shared across all runs. {counts_per_query=}")

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


    softmargins = softmargins - softmargins.mean(axis=1, keepdims=True)

    n_q, n_r = len(querry_ids), len(all_runs)
    n_person_distractors: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_city_distractors: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_relation_distractors: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_parent_facts: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    n_other_facts: FloatArray = np.zeros((n_q, n_r), dtype=np.float64)
    doc_idxs: IntArray = np.full((n_q, n_r, n_docs_per_run), IGNORE_INDEX, dtype=np.int64)  # ignore_index is used to mask out the docs that are not in the run

    train_by_run: Dict[str, pd.DataFrame] = {str(run): df for run, df in training_df.groupby("run_id")}

    for qi, qid in enumerate(querry_ids):
        meta = query_meta[qid]
        q_person = meta.get("person")
        q_city = meta.get("city")

        if relation_filter is None:
            q_relation = meta.get("relation")
        else:
            q_relation = relation_filter

        for ri, run in enumerate(all_runs):
            df_run = train_by_run.get(run)
            if df_run is None:
                continue

            run_doc_ids = df_run["doc_id"].unique().tolist()
            run_doc_idxs = [doc_id_to_idx[doc_id] for doc_id in run_doc_ids]

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
                same_relation = (df_run["relation"] == q_relation)
            else:
                same_relation = np.zeros(len(df_run), dtype=np.bool_)

            n_person_distractors[qi, ri] = float((same_person & ~same_city & ~same_relation).sum())
            n_city_distractors[qi, ri] = float((~same_person & same_city & ~same_relation).sum())
            n_relation_distractors[qi, ri] = float((~same_person & ~same_city & same_relation).sum())
            n_parent_facts[qi, ri] = float((same_person & same_city & same_relation).sum())
            n_other_facts[qi, ri] = float((~same_person & ~same_city & ~same_relation).sum())
            # assert n_parent_facts[qi, ri] + n_other_facts[qi, ri] + n_person_distractors[qi, ri] + n_city_distractors[qi, ri] + n_relation_distractors[qi, ri] == len(run_doc_idxs)
            doc_idxs[qi, ri, :len(run_doc_idxs)] = np.array(run_doc_idxs)

    return DesignMatricies(
        n_person_distractors=n_person_distractors,
        n_city_distractors=n_city_distractors,
        n_relation_distractors=n_relation_distractors,
        n_parent_facts=n_parent_facts,
        softmargins=softmargins,
        doc_idxs=doc_idxs,
        n_other_facts=n_other_facts,
        doc_id_to_idx=doc_id_to_idx,
        querry_ids=querry_ids,
        run_ids=all_runs,
    )
# %%
# Create our design matricies
# Allow overrides via environment variables for speed control
metric_name = "second_hop_inferred_fact_gen_no_fs"
out_dir = repo_root / "analysis" / "analysis" / "plots" / "bayesian" / f"{metric_name}_alpha_{alpha_level}"
out_dir.mkdir(parents=True, exist_ok=True)

design_mats = build_design_matrices(querry_df=querry_df, training_df=training_df, metric_name=metric_name, relation_filter="mayor_of")

print(f"Prepared matrices with shape Q={design_mats.n_querries}, R={design_mats.n_runs} | n records={design_mats.softmargins.size}")

train_mats, test_mats = design_mats.split_runs(split_ratio=0.8)
fit_mats, val_mats = train_mats.split_runs(split_ratio=0.75)

# %%
# Show a histogram of the softmargins, the n_person_distractors, the n_city_distractors, the n_relation_distractors, and the n_parent_facts
fig, axes = plt.subplots(6, 1, figsize=(6, 12))
axes = axes.flatten()

axes[0].hist(design_mats.softmargins.flatten(), bins=100)
axes[0].set_title("Softmargins")

axes[1].hist(design_mats.n_person_distractors.flatten(), bins=100)
axes[1].set_title("n_person_distractors")

axes[2].hist(design_mats.n_city_distractors.flatten(), bins=100)
axes[2].set_title("n_city_distractors")

axes[3].hist(design_mats.n_relation_distractors.flatten(), bins=100)
axes[3].set_title("n_relation_distractors")

axes[4].hist(design_mats.n_parent_facts.flatten(), bins=100)
axes[4].set_title("n_parent_facts")

axes[5].hist(design_mats.n_other_facts.flatten(), bins=100)
axes[5].set_title("n_other_facts")

plt.tight_layout()
plt.savefig(out_dir / "softmargins_hist.png", dpi=200)
plt.show()

# %% [markdown]
# ## Evaluations metrics

# %%
from scipy.stats import spearmanr

def data_modling_score(softmargins_obs: FloatArray, softmargin_est: FloatArray) -> float:
    """Compute the linear data modeling score for a method.

    Args:
        softmargins_obs: (n_querries, n_runs)
        softmargin_est: (n_querries, n_runs)

    Returns:
        float: The data modeling score
    """
    n_querries = softmargins_obs.shape[0]
    assert n_querries > 0
    rhos = [
        spearmanr(softmargins_obs[i, :], softmargin_est[i, :], nan_policy="raise").statistic for i in range(softmargins_obs.shape[0])
    ]
    return np.mean(rhos).item()


# %% [markdown]
# # Model
# ## Group data model
# %%
def group_model(design_mats: DesignMatricies):

    μ_α = sample("μ_α", dist.Normal(0.0, 0.5))
    σ_α = sample("σ_α", dist.HalfNormal(0.5))

    μ_β = sample("μ_β", dist.Normal(0.0, 0.5))
    σ_β = sample("σ_β", dist.HalfNormal(0.5))

    μ_r = sample("μ_r", dist.Normal(0.0, 0.5))
    σ_r = sample("σ_r", dist.HalfNormal(0.5))

    μ_p = sample("μ_p", dist.Normal(0.0, 0.5))
    σ_p = sample("σ_p", dist.HalfNormal(0.5))

    μ_o = sample("μ_o", dist.Normal(0.0, 0.5))
    σ_o = sample("σ_o", dist.HalfNormal(0.5))

    # μ_b = sample("μ_b", dist.Normal(0.0, 15))
    # σ_b = sample("σ_b", dist.HalfNormal(15))

    σ = sample("σ", dist.HalfNormal(3.0))

    with plate("querries", design_mats.n_querries):
        α = sample("α", dist.Normal(μ_α, σ_α))
        β = sample("β", dist.Normal(μ_β, σ_β))
        r = sample("r", dist.Normal(μ_r, σ_r))
        p = sample("p", dist.Normal(μ_p, σ_p))
        # b = sample("b", dist.Normal(μ_b, σ_b))
        o = sample("o", dist.Normal(μ_o, σ_o))


    softmargin_est = (
        # b[:, None]
        o[:, None] * design_mats.n_other_facts
        + α[:, None] * design_mats.n_person_distractors
        + β[:, None] * design_mats.n_city_distractors
        + r[:, None] * design_mats.n_relation_distractors
        + p[:, None] * design_mats.n_parent_facts
    )

    with plate("runs", design_mats.n_runs):
        with plate("querries", design_mats.n_querries):
            sample("obs", dist.Normal(softmargin_est, σ), obs=design_mats.softmargins)

# %% [markdown]
# ## Run MCMC

# %%

def run_mcmc(model_fn: Callable[[DesignMatricies], None], design_mats: DesignMatricies) -> dict[str, object]:
    kernel = mcmc_mod.NUTS(model_fn)
    mcmc = mcmc_mod.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    rng_key = jax.random.PRNGKey(random_seed)
    mcmc.run(
        rng_key,
        design_mats,
    )
    samples = mcmc.get_samples()
    return samples


# %%

# Allow overrides via environment variables for speed control
num_warmup = 1000
num_samples = 1000
num_chains = 1
random_seed = 0

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

reparam_config = {
    "μ_α": LocScaleReparam(0),
    "μ_β": LocScaleReparam(0),
    "μ_r": LocScaleReparam(0),
    "μ_p": LocScaleReparam(0),
    "μ_o": LocScaleReparam(0),
    "α": LocScaleReparam(0),
    "β": LocScaleReparam(0),
    "r": LocScaleReparam(0),
    "p": LocScaleReparam(0),
    "o": LocScaleReparam(0),
}
group_model_reparam = reparam(
    group_model, config=reparam_config
)

samples = run_mcmc(group_model_reparam, fit_mats)
rng_key = jax.random.PRNGKey(random_seed)
predictive = mcmc_mod.Predictive(group_model_reparam, samples, return_sites=["obs"])
samples_predictive = predictive(rng_key, val_mats.predictive())

pred_softmargins = samples_predictive["obs"].mean(axis=0)

# %%
linear_data_modling_score = data_modling_score(val_mats.softmargins, pred_softmargins)
print(f"Linear data modling score: {linear_data_modling_score}")

# Show a scatter plot of the softmargins and the pred_softmargins
plt.figure(figsize=(10, 6))
plt.scatter(val_mats.softmargins.flatten(), pred_softmargins.flatten())
plt.xlabel("Softmargins")
plt.ylabel("Predicted Softmargins")
plt.title("Softmargins vs Predicted Softmargins")
plt.savefig(out_dir / "softmargins_vs_predicted_softmargins.png", dpi=200)
plt.show()

# %%
samples = run_mcmc(group_model_reparam, train_mats)

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
    "μ_o",
    "σ_o",
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
plt.savefig(out_dir / "posterior_meta_params_hist.png", dpi=200)
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True, gridspec_kw={'width_ratios': [3, 1]})

variable_map = {
    "α": "Distractor: Same Person (Total Effect)",
    "β": "Distractor: Same City (Total Effect)",
    "r": "Distractor: Same Relation (Total Effect)",
    "p": "Parent Fact (Total Effect)",
    "o": "Other Facts (Total Effect)",
}
hue_order = [
    "Distractor: Same Person (Total Effect)",
    "Distractor: Same City (Total Effect)",
    "Distractor: Same Relation (Total Effect)",
    "Parent Fact (Total Effect)",
    "Other Facts (Total Effect)",
]
dataset_weights = {
    "α": 500,
    "β": 500,
    "r": 900,
    "p": 100,
    "o": 7000,
}

violin_rows: list[dict[str, object]] = []
agg_violin_rows: list[dict[str, object]] = []
for k, nice_name in variable_map.items():
    if k not in samples:
        continue
    for sample_idx, val in enumerate(samples["μ_" + k]):
        val = float(val)*dataset_weights[k]
        agg_violin_rows.append({"parameter": nice_name, "value": val, "sample_idx": sample_idx})
    for querry_idx, querry_id in enumerate(train_mats.querry_ids):
        vals = np.asarray(samples[k][:, querry_idx]).reshape(-1)*dataset_weights[k]
        for sample_idx, v in enumerate(vals):
            violin_rows.append({"parameter": nice_name, "value": float(v), "querry_id": querry_id, "querry_idx": querry_idx, "sample_idx": sample_idx})

df_agg = pd.DataFrame(agg_violin_rows)
df_violin = pd.DataFrame(violin_rows)
plt.figure(figsize=(8, 4))
sns.violinplot(data=df_violin, hue="parameter", x="querry_idx", hue_order=hue_order, y="value", inner="quartile", cut=0, ax=ax1)
ax1.set_title(f"Posterior of parameters (α={alpha_level}), {metric_name}")
ax1.legend(bbox_to_anchor=(0.3, 1.05), loc='lower center')
ax1.axhline(0, color="black", linestyle="--")


sns.violinplot(data=df_agg, hue="parameter", y="value", inner="quartile", hue_order=hue_order, cut=0, ax=ax2, legend=False)

ax2.axhline(0, color="black", linestyle="--")
ax2.set_title("Aggregated")

plt.tight_layout()
plt.savefig(out_dir / "posterior_parameters_violin.png", dpi=200)
plt.show()

# %%
