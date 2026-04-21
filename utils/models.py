import firetruck as ft
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

numpyro.set_host_device_count(4)


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


@ft.compact
def weighted_mean(self, y0, yg, participant_id, n_participants: int, total_count=7):
    inv_probit = jax.scipy.stats.norm.cdf
    self.mu_w = dist.Normal(0, 0.5)
    self.sigma_w = dist.HalfNormal(0.5)
    with numpyro.plate("n_participants", n_participants):
        self._social_w = dist.Normal(self.mu_w, self.sigma_w)
        self.epsilon = dist.HalfNormal(0.5)
    self.social_w = inv_probit(self._social_w)
    mu_y = yg * self.social_w[participant_id] + y0 * (1 - self.social_w[participant_id])
    return dist.TruncatedNormal(
        mu_y, self.epsilon[participant_id], low=0, high=total_count
    )


@ft.compact
def wba(
    self, y0, yg, participant_id, n_participants: int, use_kappa=False, total_count=7
):
    inv_probit = jax.scipy.stats.norm.cdf
    self.mu_w = dist.Normal(0, 0.5)
    self.sigma_w = dist.HalfNormal(0.5)
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
    self.a1 = w0[participant_id] * y0 + wg[participant_id] * yg + a0
    self.b1 = total_count - w0[participant_id] * y0 - wg[participant_id] * yg + b0
    return dist.BetaBinomial(self.a1, self.b1, total_count=total_count)


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
