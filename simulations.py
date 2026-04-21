from typing import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import ndtri as probit
from numpyro.handlers import seed


def simulate_outcomes(
    rng_key, bias: float = 0.5, certainty: int = 4, n_trials: int = 100
):
    a0 = (certainty - 1) * bias
    b0 = (certainty - 1) * (1 - bias)
    return dist.BetaBinomial(a0, b0, total_count=7).expand((n_trials,)).sample(rng_key)


def set_weights(agent, social_w):
    return agent.do({"_social_w": probit(social_w)})


def simulate_experiment(
    rng_key,
    agent: Callable,
    n_trials_per_agent: int,
    n_agents: int,
    agent_weights=None,
    group_bias=0.5,
    agent_bias=0.5,
):
    key = rng_key
    key, subkey = jax.random.split(key)
    y0 = simulate_outcomes(
        subkey, bias=agent_bias, certainty=4, n_trials=n_trials_per_agent * n_agents
    )
    key, subkey = jax.random.split(key)
    yg = simulate_outcomes(
        subkey, bias=group_bias, certainty=4, n_trials=n_trials_per_agent * n_agents
    )
    participant_id = jnp.repeat(jnp.arange(n_agents), n_trials_per_agent)
    input_data = {
        "y0": y0,
        "yg": yg,
        "participant_id": participant_id,
        "n_participants": n_agents,
    }
    # Creating agent with input data
    agents = agent.add_input(**input_data)
    if agent_weights is not None:
        agents = set_weights(agents, social_w=jnp.array(agent_weights))
    key, subkey = jax.random.split(key)
    y1_dist = seed(agents, subkey)()
    key, subkey = jax.random.split(key)
    y1 = y1_dist.sample(subkey)
    return input_data, y1
