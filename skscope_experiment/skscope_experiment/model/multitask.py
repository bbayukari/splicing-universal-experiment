import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from sklearn.datasets import make_regression


def data_generator(n, p, k, m, seed):
    X, y, coef = make_regression(n_samples=n, n_features=p, n_informative=k, n_targets=m, coef=True, random_state=seed)
    return coef.flatten(), (X, y)

def loss_jax(params, data):
    p = data[0].shape[1]
    m = data[1].shape[1]
    return jnp.sum(jnp.square(data[1] - data[0] @ params.reshape((p, m))))

def loss_cvxpy(params, data):
    return cp.sum_squares(data[1] - data[0] @ params)

if __name__ == "__main__":
    true_params, data = data_generator(20, 5, 3, 2, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable((5, 2))
    true_params_cvxpy.value = true_params.reshape((5, 2))
    print(loss_cvxpy(true_params_cvxpy, data).value)
