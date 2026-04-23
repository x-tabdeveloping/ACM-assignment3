from typing import Callable

import arviz as az
import firetruck as ft
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute
from numpyro.infer.reparam import LocScaleReparam

numpyro.set_host_device_count(4)

REPARAM_CONFIG = {"_social_w": LocScaleReparam(0)}


def compute_loglikelihood(model, samples, observations):
    def _log_prob(carry, parameters):
        m = seed(substitute(model, parameters), jax.random.key(0))
        return carry, m().log_prob(observations)

    _, log_p = jax.lax.scan(_log_prob, None, samples)
    return log_p


def reparametrize(model: Callable):
    return model.reparam(REPARAM_CONFIG)


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
    return mcmc, samples


def sample_predictives(
    rng_key, models: dict[str, Callable], samples: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, dict]]:
    prior_predictives = {}
    posterior_predictives = {}
    for model_name, model in models.items():
        posterior = samples[model_name]
        key, subkey = jax.random.split(rng_key)
        posterior_predictive = model.sample_predictive(
            subkey, posterior_samples=posterior
        )
        posterior_predictive["obs"] = jnp.round(posterior_predictive["obs"]).astype(int)
        posterior_predictives[model_name] = posterior_predictive
        key, subkey = jax.random.split(key)
        prior_predictive = model.sample_predictive(subkey)
        prior_predictive["obs"] = jnp.round(prior_predictive["obs"]).astype(int)
        prior_predictives[model_name] = prior_predictive
    return prior_predictives, posterior_predictives


def compare_models(mcmcs, posterior_predictives):
    idatas = {
        model_name: az.from_numpyro(
            mcmcs[model_name],
            posterior_predictive=posterior_predictives[model_name],
            log_likelihood=True,
        )
        for model_name in mcmcs
    }
    return az.compare(idatas)


def update_beliefs(y, a, b, total_count=7):
    a1 = y + a
    b1 = total_count - y + b
    return a1, b1


@ft.compact
def belief_model(self, y0, yg, participant_id, n_participants: int, total_count=7):
    inv_probit = jax.scipy.stats.norm.cdf
    self.pop_mu = dist.Normal(0, 0.5)
    self.pop_sigma = dist.HalfNormal(0.5)
    self.pop_k = dist.Normal(0, 0.5)
    self.pop_ks = dist.HalfNormal(0.5)
    with numpyro.plate("n_participants", n_participants):
        self.probit_mu = dist.Normal(self.pop_mu, self.pop_sigma)
        # self.mu_sigma = dist.HalfNormal(0.5)
        self.log_kappa = dist.Normal(
            self.pop_k,
            self.pop_ks,
        )
    self.mu = inv_probit(self.probit_mu)
    self.kappa = jnp.exp(self.log_kappa)
    self._trial_mu = dist.Normal(
        self.probit_mu[participant_id], self.mu_sigma[participant_id]
    )
    self.trial_mu = inv_probit(self.trial_mu)
    self.a0 = (self.kappa[participant_id] - 1) * self.trial_mu
    self.b0 = (self.kappa[participant_id] - 1) * (1 - self.trial_mu)
    self.a1, self.b1 = update_beliefs(yg, self.a0, self.b0, total_count=total_count)
    numpyro.sample(
        "y0", dist.BetaBinomial(self.a0, self.b0, total_count=total_count), obs=y0
    )
    return dist.BetaBinomial(self.a1, self.b1, total_count=total_count)


@reparametrize
@ft.compact
def weighted_mean(self, y0, yg, participant_id, n_participants: int, total_count=7):
    inv_probit = jax.scipy.stats.norm.cdf
    self.mu_w = dist.Normal(0, 1.0)
    self.sigma_w = dist.HalfNormal(1.0)
    with numpyro.plate("n_participants", n_participants):
        self._social_w = dist.Normal(self.mu_w, self.sigma_w)
        self.epsilon = dist.HalfNormal(0.5)
    self.social_w = inv_probit(self._social_w)
    mu_y = yg * self.social_w[participant_id] + y0 * (1 - self.social_w[participant_id])
    return dist.TruncatedNormal(
        mu_y, self.epsilon[participant_id], low=0, high=total_count
    )


@reparametrize
@ft.compact
def wba(
    self, y0, yg, participant_id, n_participants: int, use_kappa=False, total_count=7
):
    inv_probit = jax.scipy.stats.norm.cdf
    self.mu_w = dist.Normal(0, 1.0)
    self.sigma_w = dist.HalfNormal(1.0)
    with numpyro.plate("n_participants", n_participants):
        if use_kappa:
            self.kappa = dist.LogNormal(0, 1)
        else:
            self.kappa = jnp.ones(n_participants)
        self._social_w = dist.Normal(self.mu_w, self.sigma_w)
    self.social_w = inv_probit(self._social_w)
    a0, b0 = 1, 1
    w0 = self.kappa * (1 - self.social_w)
    wg = self.kappa * self.social_w
    a1 = w0[participant_id] * y0 + wg[participant_id] * yg + a0
    b1 = total_count - w0[participant_id] * y0 - wg[participant_id] * yg + b0
    return dist.BetaBinomial(a1, b1, total_count=total_count)


@reparametrize
@ft.compact
def directional(self, y0, yg, participant_id, n_participants: int, total_count=7):
    self.mu_w = dist.Normal(0.5, 0.5)
    self.sigma_w = dist.HalfNormal(0.5)
    with numpyro.plate("n_participants", n_participants):
        self._social_w = dist.Normal(self.mu_w, self.sigma_w)
        self.epsilon = dist.HalfNormal(0.5)
    diff = yg - y0
    self.social_w = self._social_w
    mu_y = y0 + diff * self.social_w[participant_id]
    return dist.TruncatedNormal(
        mu_y, self.epsilon[participant_id], low=0, high=total_count
    )
