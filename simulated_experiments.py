from typing import Callable

import firetruck as ft
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.models import (compare_models, init_models, sample_models,
                          sample_predictives, wba, weighted_mean)
from utils.plots import plot_forests, plot_kendall_taus
from utils.simulations import simulate_experiment


def plot_behavior(input_data, y1):
    fig = make_subplots(rows=input_data["n_participants"], cols=1)
    for i_participant in range(input_data["n_participants"]):
        matches = input_data["participant_id"] == i_participant
        _data = {
            key: value[matches]
            for key, value in input_data.items()
            if key != "n_participants"
        }
        _y1 = y1[matches]
        idx = jnp.arange(_y1.shape[0])
        fig = fig.add_scatter(
            x=idx,
            y=_data["y0"],
            name="y0",
            row=i_participant + 1,
            col=1,
            line=dict(color="blue"),
            opacity=0.5,
            showlegend=i_participant == 0,
        )
        fig = fig.add_scatter(
            x=idx,
            y=_data["yg"],
            name="yg",
            row=i_participant + 1,
            col=1,
            line=dict(color="red"),
            opacity=0.5,
            showlegend=i_participant == 0,
        )
        fig = fig.add_scatter(
            x=idx,
            y=_y1,
            name="y1",
            row=i_participant + 1,
            col=1,
            line=dict(color="black"),
            showlegend=i_participant == 0,
        )
    return fig


key = jax.random.key(0)
N_AGENTS = 40
key, subkey = jax.random.split(key)
input_data, y1, params = simulate_experiment(
    subkey,
    weighted_mean,
    agent_weights=jnp.linspace(0.1, 1.0, N_AGENTS),
    n_trials_per_agent=150,
    n_agents=N_AGENTS,
    agent_bias=0.5,
    group_bias=0.5,
    certainty=4,
)

models = {"wba": wba, "weighted_mean": weighted_mean}
models = init_models(models, input_data)
key, subkey = jax.random.split(key)
mcmcs, samples = sample_models(subkey, models, y1)
key, subkey = jax.random.split(key)
prior_predictives, posterior_predictives = sample_predictives(subkey, models, samples)

compare_models(mcmcs, posterior_predictives)

plot_kendall_taus(posterior_predictives, obs=y1)

posterior_predictives["wba"]["obs"].shape

y1.shape

ft.plot_trace(mcmcs["wba"])

inv_probit = jax.scipy.stats.norm.cdf
inv_probit(-0.21)


plot_forests(samples, variable="social_w", true_params=params)
