import jax
import jax.numpy as jnp
from jax.scipy.stats import sem
from sklearn.model_selection import StratifiedKFold

from utils.models import compute_loglikelihood


def tree_index(tree, idx):
    return {
        key: tree[key][idx] if len(jnp.array(tree[key]).shape) > 0 else tree[key]
        for key in tree
    }


def kfold(input_data, y1, k: int = 5):
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    # Stratifying across participants
    for i, (train_index, test_index) in enumerate(
        kf.split(y1, input_data["participant_id"])
    ):
        yield tree_index(input_data, train_index), y1[train_index], tree_index(
            input_data, test_index
        ), y1[test_index]


def elpd_kfold(rng_key, model, input_data, y1, k: int = 10):
    lpds = []
    key = rng_key
    for train_data, train_y1, test_data, test_y1 in kfold(input_data, y1):
        key, subkey = jax.random.split(key)
        mcmc = (
            model.add_input(**train_data)
            .condition_on(train_y1)
            .sample_posterior(subkey)
        )
        samples = mcmc.get_samples()
        log_prob = compute_loglikelihood(model.add_input(**test_data), samples, test_y1)
        lpd = jnp.sum(jnp.mean(log_prob, axis=0))
        lpds.append(lpd)
    lpds = jnp.array(lpds)
    return jnp.mean(lpds), sem(lpds)
