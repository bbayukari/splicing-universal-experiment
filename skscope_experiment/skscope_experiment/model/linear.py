import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import abess
import _skscope_experiment

def data_generator(n, p, k, seed):
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([100, -100], k)
    data = abess.make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.1,
        family="gaussian",
        corr_type="exp",
        snr=10 * np.log10(60),
        sigma=0,
        coef_=coef
    )
    return coef, (data.x, data.y)

def data_cpp_wrapper(data):
    return _skscope_experiment.RegressionData(data[0], data[1])

def loss_cpp(params, data):
    return _skscope_experiment.linear_loss(params, data)

def grad_cpp(params, data):
    return _skscope_experiment.linear_grad(params, data)

def loss_jax(params, data):
    return jnp.sum(jnp.square(data[1] - data[0] @ params))

def loss_cvxpy(params, data):
    return cp.sum_squares(data[1] - data[0] @ params)

if __name__ == "__main__":
    true_params, data = data_generator(20, 10, 5, 0)
    print(true_params)
    print(loss_jax(jnp.array(true_params), data))
    true_params_cvxpy = cp.Variable(len(true_params))
    true_params_cvxpy.value = true_params
    print(loss_cvxpy(true_params_cvxpy, data).value)