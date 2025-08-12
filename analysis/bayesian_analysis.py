# %% [markdown]
# # Bayesian Data Modeling
# The goal is to create a hirachical baysian model of the effects of including a datapoint in our trainingset. We are going to estimate this models
# parameters using numpyro and hopfully use it to evaluate our influence functions.


# %%
# First we need to prepare the dataset.
from shared_ml.logging import load_log_from_disk
import hashlib
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from oocr_influence.datasets.synthetic_pretraining_docs import load_dataset_builders

group_data_modeling_path = Path(
    "/mfs1/u/levmckinney/experiments/oocr-inf/outputs/2025_08_11_21-45-16_group_data_modeling"
)

# %%
# Create the querry dataframe.
querry_df = []
for checkpoint_dir in tqdm(group_data_modeling_path.iterdir(), total=100):
    if not checkpoint_dir.is_dir():
        continue
    run_id = hashlib.sha256(checkpoint_dir.name.encode()).hexdigest()[:8]
    experiment_log = load_log_from_disk(checkpoint_dir)
    history = experiment_log.history
    metrics = history[-1]['eval_results']
    for key, value in metrics.items():
        for record in value['records']:
            softmargin = record['softmargin']
            fact_template = json.loads(record['fact_template'])
            features = json.loads(record['features'])
            relation = fact_template['relation']
            person = features['fields'].get('name_of_person', None)
            city = features['fields'].get('city_name', None)
            querry_id = record['id']
            querry_df.append({
                'run_id': run_id,
                'softmargin': softmargin,
                'relation': relation,
                'person': person,
                'city': city,
                'querry_id': querry_id
            })

querry_df = pd.DataFrame(querry_df)

# %%
# Create the training dataframe.
training_df = []
for checkpoint_dir in tqdm(group_data_modeling_path.iterdir(), total=100):
    if not checkpoint_dir.is_dir():
        continue
    run_id = hashlib.sha256(checkpoint_dir.name.encode()).hexdigest()[:8]
    experiment_log = load_log_from_disk(checkpoint_dir)
    dataset_builder_path = Path(experiment_log.args['synth_dataset_builders_path'])
    _, _, metadata = load_dataset_builders(dataset_builder_path)
    for datapoint in metadata['docs_included']:
        doc_id = datapoint['id']
        doc_idea = datapoint['doc_idea']
        reversal_curse = datapoint['reversal_curse']
        relation = datapoint['fact']['template']
        person = datapoint['fact']['feature_set']['fields'].get('name_of_person', None)
        city = datapoint['fact']['feature_set']['fields'].get('city_name', None)
        training_df.append({
            'run_id': run_id,
            'doc_id': doc_id,
            'fact_id': datapoint['fact']['id'],
            'doc_idea': doc_idea,
            'reversal_curse': reversal_curse,
            'relation': relation,
            'person': person,
            'city': city,
        })

training_df = pd.DataFrame(training_df)
# %%
import numpyro
from numpyro import sample
import numpyro.distributions as dist


n_training_runs = len(training_df['run_id'].unique())
n_querries = 10  # impute this
def model(
        querry_idx: int, 
        n_with_same_person,
        n_with_same_city: int,
        n_with_same_relation: int,
        n_entails_querry: int,
        softmargin_obs=None
    ):
    # Contains same person as querry prior parameters (mu_alpha, sigma_alpha)
    μ_α = sample("μ_α", dist.Normal(0, 5))
    σ_α = sample("σ_α", dist.HalfNormal(1))

    # Contains same city as querry prior parameters (mu_beta, sigma_beta)
    μ_β = sample("μ_β", dist.Normal(0, 5))
    σ_β = sample("σ_β", dist.HalfNormal(5))

    # Contains same relation as querry prior parameters (mu_relation, sigma_relation)
    μ_r = sample("μ_r", dist.Normal(0, 5))
    σ_r = sample("σ_r", dist.HalfNormal(5))

    # entails querry
    μ_p = sample("μ_p", dist.Normal(0, 5))
    σ_p = sample("σ_p", dist.HalfNormal(5))

    # Bias term
    μ_b = sample("μ_b", dist.Normal(0, 5))
    σ_b = sample("σ_b", dist.HalfNormal(5))

    with numpyro.plate("querries", n_querries):
        α = sample("α", dist.Normal(μ_α, σ_α))
        β = sample("β", dist.Normal(μ_β, σ_β))
        r = sample("r", dist.Normal(μ_r, σ_r))
        p = sample("p", dist.Normal(μ_p, σ_p))
        b = sample("b", dist.Normal(μ_b, σ_b))

    σ = sample("σ", dist.HalfNormal(5))

    softmargin_est =(
            b[querry_idx] 
            + α[querry_idx] * n_with_same_person 
            + β[querry_idx] * n_with_same_city 
            + r[querry_idx] * n_with_same_relation 
            + p[querry_idx] * n_entails_querry
    )
    with numpyro.plate("training_runs", n_training_runs):
        sample(
            "obs",
            dist.Normal(softmargin_est, σ),
            obs=softmargin_obs
        )
    
# %%
