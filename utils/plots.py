from typing import Optional

import firetruck as ft
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from firetruck.plots import get_plotly
from numpyro.diagnostics import hpdi
from plotly.subplots import make_subplots
from scipy.stats import kendalltau


def plot_predictives(
    prior_predictives: dict[str, dict],
    posterior_predictives: dict[str, dict],
    obs: jax.Array,
):
    model_names = list(prior_predictives.keys())
    subplot_titles = []
    for t in ["prior", "posterior"]:
        for model_name in model_names:
            subplot_titles.append(f"{model_name} - {t} predictive")
    fig = make_subplots(
        rows=2,
        cols=len(prior_predictives),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    for i_model, model_name in enumerate(model_names):
        subfig = ft.plot_predictive_check(
            prior_predictives[model_name]["obs"], obs=obs.astype(int)
        )
        for trace in subfig.data:
            trace.showlegend = (i_model == 0) and trace.showlegend
            fig.add_trace(
                trace,
                col=i_model + 1,
                row=1,
            )
        subfig = ft.plot_predictive_check(
            posterior_predictives[model_name]["obs"], obs=obs.astype(int)
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
    return fig


def plot_forests(
    samples: dict[str, dict],
    prob: float = 0.94,
    variable: Optional[str] = None,
    true_params: Optional[dict[str, jax.Array]] = None,
):
    px, go, subplots = get_plotly()
    model_names = list(samples.keys())
    fig = subplots.make_subplots(
        rows=len(samples), cols=1, subplot_titles=model_names, vertical_spacing=0.1
    )
    for i_model, model_name in enumerate(model_names):
        model_samples = samples[model_name]
        colors = px.colors.qualitative.Dark24
        var_samples = model_samples[variable]
        n_samples = var_samples.shape[0]
        if len(var_samples.shape) == 2:
            var_samples = var_samples.T
        elif len(var_samples.shape) == 1:
            var_samples = var_samples[None, :]
        else:
            var_samples = var_samples.T
            var_samples = jnp.reshape(var_samples, (-1, n_samples))
        hpis = []
        medians = []
        for i_level, level in enumerate(var_samples):
            hpis.append(list(hpdi(level, prob=prob)))
            medians.append(jnp.median(level))
        hpis = jnp.array(hpis)
        medians = jnp.array(medians)
        if true_params is None:
            level_order = jnp.arange(len(medians))
        else:
            level_order = jnp.argsort(true_params[variable])
        for i_level in level_order:
            i_level = int(i_level)
            center = medians[i_level]
            lower, upper = hpis[i_level]
            name = variable
            if var_samples.shape[0] != 1:
                name += f"[{i_level}]"
            if true_params is None:
                x0 = name
            else:
                x0 = true_params[variable][i_level]
            fig.add_scatter(
                x0=x0,
                y=[center],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[upper - center],
                    arrayminus=[center - lower],
                    width=0,
                    thickness=2.5,
                ),
                marker=dict(color=colors[i_level % len(colors)]),
                name=name,
                showlegend=False,
                mode="markers",
                col=1,
                row=i_model + 1,
            )
            level = var_samples[i_level]
            lower, upper = hpdi(level, prob=0.5)
            fig.add_scatter(
                x0=x0,
                y=[center],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[upper - center],
                    arrayminus=[center - lower],
                    width=0,
                    thickness=5,
                ),
                marker=dict(color=colors[i_level % len(colors)], size=12),
                name=name,
                showlegend=False,
                mode="markers",
                col=1,
                row=i_model + 1,
            )
        if true_params is not None:
            v = true_params[variable]
            fig.add_scatter(
                x=[jnp.min(v), jnp.max(v)],
                y=[jnp.min(v), jnp.max(v)],
                mode="lines",
                name="True Value",
                line=dict(width=2, dash="dash", color="black"),
                showlegend=i_model == 0,
                row=i_model + 1,
                col=1,
            )
    fig = fig.update_layout(template="plotly_white", margin=dict(t=20, b=0, l=0, r=0))
    return fig


def plot_kendall_taus(predictives, obs):
    fig = go.Figure()
    for model_name in predictives:
        pred = predictives[model_name]["obs"]
        taus = []
        for draw in pred:
            taus.append(kendalltau(draw, obs).statistic)
        fig.add_box(name=model_name, y=taus, showlegend=False)
    fig = fig.update_yaxes(title=" $\\text{Kendall's }\\tau$")
    fig = fig.update_layout(
        template="plotly_white",
        width=500,
        height=300,
        margin=dict(t=30, r=10, b=10, l=10),
        font=dict(size=16),
    )
    return fig
