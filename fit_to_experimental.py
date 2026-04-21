from collections import defaultdict
from typing import Callable

import arviz as az
import firetruck as ft
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from firetruck.plots import _predictive_check_discrete
from numpyro.infer.reparam import LocScaleReparam
from plotly.subplots import make_subplots
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils.models import wba, weighted_mean

numpyro.set_host_device_count(4)


def init_models(models: dict[str, Callable], input_data: dict) -> dict[str, Callable]:
    new_models = dict()
    for model_name in models:
        model = models[model_name]
        # Adding input
        model = model.add_input(**input_data)
        new_models[model_name] = model
    return new_models


def sample_models(
    rng_key: jax.random.PRNGKey, models: dict[str, Callable], output_data: jax.Array
) -> tuple:
    mcmc = {}
    samples = {}
    key = rng_key
    for model_name in models:
        print(f"--------------Sampling {model_name}----------------")
        model = models[model_name]
        key, subkey = jax.random.split(key)
        # Sampling the posterior
        # We use a dense mass matrix, because sigma is hard to sample from,
        mcmc[model_name] = model.condition_on(output_data).sample_posterior(
            subkey,
            # dense_mass=True,
            num_chains=4,
        )
        samples[model_name] = mcmc[model_name].get_samples()
    return key, mcmc, samples


def read_data():
    data = pd.read_csv("dat/Simonsen_clean.csv")
    data = data[data["ID"].isin(data["ID"].unique())]
    n_participants = len(data["ID"].unique())
    labeller = LabelEncoder()
    participant_id = labeller.fit_transform(data["ID"])
    y0, yg, y1 = (
        jnp.array(data["FirstRating"]) - 1,
        jnp.array(data["GroupRating"]) - 1,
        jnp.array(data["SecondRating"]) - 1,
    )
    input_data = dict(
        y0=y0,
        yg=yg,
        participant_id=participant_id,
        n_participants=n_participants,
    )
    return input_data, y1


def tree_index(tree, idx):
    return {key: tree[key][idx] for key in tree}


def kfold(input_data, y1, k: int = 5):
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    # Stratifying across participants
    for i, (train_index, test_index) in enumerate(
        kf.split(y1, input_data["participant_id"])
    ):
        yield tree_index(input_data, train_index), y1[train_index], tree_index(
            input_data, test_index
        ), y1[test_index]


config = {"_social_w": LocScaleReparam(0)}
input_data, y1 = read_data()
models = {"wba": wba, "weighted_mean": weighted_mean}
models = {model_name: model.reparam(config) for model_name, model in models.items()}

train_models = init_models(models, input_data)
key = jax.random.key(0)
key, mcmc, samples = sample_models(key, train_models, y1)

ft.plot_trace(mcmc["weighted_mean"])

model_names = list(models.keys())
fig = make_subplots(
    rows=len(models), cols=1, subplot_titles=model_names, vertical_spacing=0.1
)
for i_model, model_name in enumerate(model_names):
    subfig = ft.plot_forest(mcmc[model_name], variables=["social_w"])
    for trace in subfig.data:
        fig.add_trace(trace, row=i_model + 1, col=1)
fig = fig.add_vline(x=0.5, line=dict(dash="dash", color="black", width=2))
fig = fig.update_xaxes(matches="x")
fig = fig.update_layout(
    template="plotly_white", width=800, height=800, margin=dict(t=30, r=10, b=10, l=10)
)
fig.show()


prior_predictives = {}
posterior_predictives = {}
for model_name, model in train_models.items():
    posterior = samples[model_name]
    key, subkey = jax.random.split(key)
    posterior_predictive = model.sample_predictive(subkey, posterior_samples=posterior)
    posterior_predictive["obs"] = jnp.round(posterior_predictive["obs"]).astype(int)
    posterior_predictives[model_name] = posterior_predictive
    key, subkey = jax.random.split(key)
    prior_predictive = model.sample_predictive(subkey)
    prior_predictive["obs"] = jnp.round(prior_predictive["obs"]).astype(int)
    prior_predictives[model_name] = prior_predictive

fig = go.Figure()
for model_name in train_models:
    pred = prior_predictives[model_name]["obs"]
    taus = []
    for draw in pred:
        taus.append(kendalltau(draw, y1).statistic)
    fig.add_box(name=model_name, y=taus, showlegend=False)
fig = fig.update_yaxes(title=" $\\text{Kendall's }\\tau$")
fig = fig.update_layout(
    template="plotly_white",
    width=500,
    height=300,
    margin=dict(t=30, r=10, b=10, l=10),
    font=dict(size=16),
)
fig.show()

idatas = {
    model_name: az.from_numpyro(
        mcmc[model_name],
        posterior_predictive=posterior_predictives[model_name],
        log_likelihood=True,
    )
    for model_name in mcmc
}

az.plot_compare(
    az.compare(
        {"weighted_mean": idatas["weighted_mean"], "directional": idatas["directional"]}
    ),
    backend="plotly",
).show()


model_names = list(train_models.keys())
subplot_titles = []
for t in ["prior", "posterior"]:
    for model_name in model_names:
        subplot_titles.append(f"{model_name} - {t} predictive")
fig = make_subplots(
    rows=2,
    cols=len(models),
    subplot_titles=subplot_titles,
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
)
for i_model, model_name in enumerate(model_names):
    subfig = _predictive_check_discrete(
        go.Figure(), None, prior_predictives[model_name]["obs"], y1.astype(int)
    )
    for trace in subfig.data:
        trace.showlegend = (i_model == 0) and trace.showlegend
        fig.add_trace(
            trace,
            col=i_model + 1,
            row=1,
        )
    subfig = _predictive_check_discrete(
        go.Figure(), None, posterior_predictives[model_name]["obs"], y1.astype(int)
    )
    for trace in subfig.data:
        trace.showlegend = False
        fig.add_trace(trace, col=i_model + 1, row=2)
fig = fig.update_yaxes(matches="y")
fig = fig.update_layout(
    template="plotly_white",
    barmode="overlay",
    margin=dict(l=10, r=10, t=30, b=10),
    width=1200,
    height=600,
    font=dict(size=16),
)
fig.show()

for train_data, train_y, test_data, test_y in kfold(input_data, y1):
    train_models = init_models(
        models,
    )
