from skscope_experiment import (
    linear,
    logistic,
    ising,
)
import scope
import time
import numpy as np
import cvxpy as cp
import parallel_experiment_util
import jax

model_dict = {
    "linear": linear,
    "logistic": logistic,
    "ising": ising,
}


def task(model, n, p, k, seed):
    results = []
    true_params, data = model_dict[model].data_generator(n, p, k, seed)
    data_cpp = model_dict[model].data_cpp_wrapper(data)
    loss_cpp_data = lambda x: np.array(model_dict[model].loss_cpp(x, data_cpp))
    grad_cpp_data = lambda x: np.array(model_dict[model].grad_cpp(x, data_cpp))

    for max_exchange_num in [1, 2, 5, 10, 20, 40]:
        solver = scope.ScopeSolver(p, k, max_exchange_num=max_exchange_num)
        t1 = time.perf_counter()
        solver.solve(loss_cpp_data, gradient=grad_cpp_data)
        t2 = time.perf_counter()
        results.append(
            {
                "max_exchange_num": max_exchange_num,
                "time": t2 - t1,
                "accuracy": len(set(solver.get_support()) & set(np.nonzero(true_params)[0])) / k,
            }
        )

    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "n", "p", "k", "seed"],
        out_keys=["max_exchange_num", "time", "accuracy"],
        processes=16,
        name="scope_k_max",
        memory_limit=0.9,
    )

    if False:
        # experiment.check(model="linear", n=200, p=500, k=50, seed=11)
        experiment.check(model="logistic", n=1000, p=500, k=50, seed=1099)
        # experiment.check(model="ising", n=200, p=190, k=40, seed=708, relax_ratio=1.0)
    else:
        parameters = parallel_experiment_util.para_generator(
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
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()

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
