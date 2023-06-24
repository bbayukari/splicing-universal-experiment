from skscope_experiment import (
    linear,
    logistic,
    ising,
)
import time
import numpy as np
import cvxpy as cp
import parallel_experiment_util

model_dict = {
    "linear": linear,
    "logistic": logistic,
    "ising": ising,
}


def task(model, n, p, k, seed, relax_ratio):
    sample_size = n
    dim = p
    sparsity_level = k * relax_ratio
    L1_init = 1.0

    results = []
    true_params, data = model_dict[model].data_generator(sample_size, dim, k, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    loss_jax_data = lambda x: model_dict[model].loss_jax(x, data)
    loss_cvxpy_data = lambda x: model_dict[model].loss_cvxpy(x, data)
    data_cpp = model_dict[model].data_cpp_wrapper(data)
    loss_cpp_data = lambda x: np.array(model_dict[model].loss_cpp(x, data_cpp))
    grad_cpp_data = lambda x: np.array(model_dict[model].grad_cpp(x, data_cpp))

    # CVXPY
    def object_fn(x, lambd):
        return loss_cvxpy_data(x) + lambd * cp.norm1(x)

    x = cp.Variable(dim)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(object_fn(x, lambd)))
    lambd_lowwer = 0.0
    lambd.value = L1_init
    params = None
    start = time.perf_counter()
    for _ in range(100):
        try:
            problem.solve()
        except:
            if params is None:
                raise ValueError("CVXPY failed")
            else:
                break
        params = x.value
        support_size_est = np.array(abs(params) > 1e-2).sum()

        if support_size_est > sparsity_level:
            lambd_lowwer = lambd.value
            lambd.value = 2 * lambd.value
        elif support_size_est < sparsity_level:
            lambd.value = (lambd_lowwer + lambd.value) / 2
        else:
            break
    end = time.perf_counter()
    results.append(
        {
            "method": "CVXPY",
            "time": end - start,
            "accuracy": len(
                set(np.argpartition(np.abs(params), -k)[-k:]) & true_support_set
            )
            / k,
        }
    )

    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "n", "p", "k", "seed", "relax_ratio"],
        out_keys=["method", "time", "accuracy"],
        processes=8,
        name="cvxpy",
        memory_limit=0.9,
    )

    if False:
        experiment.check(model="logistic", n=100, p=50, k=5, seed=1099, relax_ratio=1.0)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["logistic"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
                "relax_ratio": [1.0],
            },
            repeat=5,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save("cvxpy"+"_1_2")

"""
            {
                "model": ["linear", "logistic"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
            },
            {
                "model": ["ising"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [190],
                "k": [40],
            },
"""
