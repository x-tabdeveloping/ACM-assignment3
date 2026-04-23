from pathlib import Path

import firetruck as ft
import jax
import jax.numpy as jnp
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.models import (init_models, sample_models, sample_predictives, wba,
                          weighted_mean)
from utils.plots import (plot_elpd_kfold, plot_forests, plot_kendall_taus,
                         plot_predictives)


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


def main():
    out_dir = Path("figures/real_data")
    out_dir.mkdir(exist_ok=True, parents=True)
    print("Loading data")
    input_data, y1 = read_data()
    models = {"wba": wba, "weighted_mean": weighted_mean}
    print("Sampling models")
    train_models = init_models(models, input_data)
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    mcmcs, samples = sample_models(subkey, train_models, y1)
    print("Estimating ELPD with K-fold cross validation")
    key, subkey = jax.random.split(key)
    fig = plot_elpd_kfold(subkey, models, input_data, y1)
    fig.write_image(out_dir.joinpath("kfold_elpd.png"), scale=2)
    print("Plotting social weights.")
    fig = plot_forests(samples, variable="social_w", true_params=None).update_layout(
        width=1000, height=1000
    )
    fig.write_image(out_dir.joinpath("forests.png"), scale=2)
    print("Plotting prior and posterior predictives")
    key, subkey = jax.random.split(key)
    prior_predictives, posterior_predictives = sample_predictives(
        subkey, train_models, samples
    )
    fig = plot_predictives(prior_predictives, posterior_predictives, obs=y1)
    fig.write_image(out_dir.joinpath("predictives.png"), scale=2)
    print("Plotting Kendall taus")
    fig = plot_kendall_taus(posterior_predictives, obs=y1)
    fig.write_image(out_dir.joinpath("kendall_taus.png"), scale=2)
    print("Plotting prior-posterior updates")
    for model_name in models:
        fig = ft.plot_prior_posterior_update(
            train_models[model_name], mcmcs[model_name]
        )
        fig = fig.update_layout(width=1000, height=1000).update_traces(showlegend=False)
        fig.write_image(
            out_dir.joinpath(f"{model_name}-prior_posterior_update.png"), scale=2
        )
    print("DONE")


if __name__ == "__main__":
    main()
