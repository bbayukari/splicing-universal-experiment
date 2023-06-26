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
    sample_size = n
    dim = p
    
    L1_init = 1.0

    results = []
    true_params, data = model_dict[model].data_generator(sample_size, dim, k, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    data_cpp = model_dict[model].data_cpp_wrapper(data)
    loss_cpp_data = lambda x: np.array(model_dict[model].loss_cpp(x, data_cpp))
    grad_cpp_data = lambda x: np.array(model_dict[model].grad_cpp(x, data_cpp))
    accuracy = []
    accuracy_contain = []

    for relax_ratio in [1.0, 1.5, 2.0]:
        sparsity_level = int(k * relax_ratio)

        def count_hess(x, *data):
            accuracy_contain.append(len(set(np.argpartition(np.abs(x), -sparsity_level)[-sparsity_level:]) & true_support_set))
            accuracy.append(len(set(np.argpartition(np.abs(x), -k)[-k:]) & true_support_set))
            return np.eye(x.size)


        solver = scope.ScopeSolver(dim, sparsity_level, max_exchange_num=k, use_hessian=True)

        solver.solve(loss_cpp_data, gradient=grad_cpp_data, hessian=count_hess)

        results.extend([
            {
                "relax_ratio": relax_ratio,
                "iteration": i,
                "accuracy": accuracy[i],
                "accuracy_contain": accuracy_contain[i],
            }
            for i in range(len(accuracy))
        ])
    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "n", "p", "k", "seed"],
        out_keys=["iteration", "accuracy_contain", "accuracy", "relax_ratio"],
        processes=8,
        name="scope_relax_sparsity",
        memory_limit=0.9,
    )

    if False:
        # experiment.check(model="linear", n=200, p=500, k=50, seed=11)
        experiment.check(model="logistic", n=1000, p=500, k=50, seed=1099)
        # experiment.check(model="ising", n=200, p=190, k=40, seed=708, relax_ratio=1.0)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["logistic"],
                "n": [100, 500, 1000],
                "p": [500],
                "k": [50],
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
