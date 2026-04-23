from pathlib import Path

import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.models import (compare_models, init_models, sample_models,
                          sample_predictives, wba, weighted_mean)
from utils.plots import plot_elpd_kfold, plot_forests, plot_kendall_taus
from utils.simulations import simulate_experiment


def plot_behavior(
    input_data: dict, y1: jax.Array, agent_weights: jax.Array
) -> go.Figure:
    fig = make_subplots(
        rows=input_data["n_participants"],
        cols=1,
        subplot_titles=[f"Social w={w:.2f}" for w in agent_weights],
        vertical_spacing=0.05,
    )
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


def main():
    figures_dir = Path("figures")
    AGENTS = {"wba": wba, "weighted_mean": weighted_mean}
    N_PARTICIPANTS = 10
    AGENT_WEIGHTS = jnp.linspace(0.1, 0.9, N_PARTICIPANTS)
    key = jax.random.key(0)
    for n_trials in [25, 75, 150]:
        print(f"========RUNNING EXPERIMENTS WITH {n_trials} TRIALS=======")
        for true_agent_name, true_agent in AGENTS.items():
            print(f" - True agent = {true_agent_name}")
            out_dir = figures_dir.joinpath(f"{true_agent_name}-{n_trials}")
            out_dir.mkdir(exist_ok=True, parents=True)
            key, subkey = jax.random.split(key)
            print("   - Simulating experiment...")
            input_data, y1, params = simulate_experiment(
                subkey,
                true_agent,
                agent_weights=AGENT_WEIGHTS,
                n_trials_per_agent=n_trials,
                n_agents=N_PARTICIPANTS,
                agent_bias=0.5,
                group_bias=0.5,
                certainty=4,
            )
            print("   - Plotting behaviour...")
            fig = plot_behavior(input_data, y1, AGENT_WEIGHTS)
            fig = fig.update_layout(
                template="plotly_white",
                margin=dict(t=30, r=10, b=10, l=10),
                font=dict(size=16),
                width=1000,
                height=1200,
            )
            fig.write_html(out_dir.joinpath("behaviour.html"))
            models = init_models(
                {"wba": wba, "weighted_mean": weighted_mean}, input_data
            )
            key, subkey = jax.random.split(key)
            print("   - Sampling posteriors...")
            mcmcs, samples = sample_models(subkey, models, y1)
            key, subkey = jax.random.split(key)
            print("   - Sampling predictives...")
            prior_predictives, posterior_predictives = sample_predictives(
                subkey, models, samples
            )
            print("   - Plotting recovery...")
            fig = plot_forests(
                samples, variable="social_w", true_params=params
            ).update_layout(width=1000, height=1000)
            fig.write_image(out_dir.joinpath("recovery.png"), scale=2)
            print("   - Plotting model performance...")
            fig = plot_kendall_taus(posterior_predictives, obs=y1)
            fig.write_image(out_dir.joinpath("kendall_taus.png"), scale=2)
            print("   - Estimating ELPD with LOO...")
            try:
                comparison = compare_models(mcmcs, posterior_predictives)
                comparison.to_csv(out_dir.joinpath("comparison.csv"))
            except Exception as e:
                print(f"      COMPARISON FAILED! Reason: {e}")
                print(" - Estimating ELPD with KFold...")
                key, subkey = jax.random.split(key)
                fig = plot_elpd_kfold(subkey, models, input_data, y1)
                fig.write_image(out_dir.joinpath("kfold_elpd.png"), scale=2)
    print("DONE")


if __name__ == "__main__":
    main()
