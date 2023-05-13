import scope
import time
import numpy as np
from parallel_experiment_util import accuracy as Accuracy
from skscope_experiment import (
    linear,
    logistic,
    ising,
    trend_filtering_1d,
    non_linear_additive_example,
    non_linear_non_additive_example,
    multitask,
)

def task(loss_jax, data_generator, sample_size, dim, sparsity_level, seed):
    true_params, data = data_generator(sample_size, dim, sparsity_level, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    loss_jax_data = lambda x: loss_jax(x, data)

    for method, solver in {
        "SCOPE": scope.GraspSolver(dim, sparsity_level),
    }.items():
        t1 = time.perf_counter()
        solver.solve(loss_jax_data, jit=True)
        t2 = time.perf_counter()
        support_set = set(solver.get_support())
        for supp in solver.results.keys():
            print(len(set(supp) & true_support_set) / sparsity_level)
        print(solver.n_iters)
        print(len(support_set & true_support_set) / sparsity_level)
        """
        return {
            "method": method + "_jit" if jit else method,
            "time": t2 - t1,
            "accuracy": len(support_set & true_support_set) / sparsity_level,
            "n_iters": solver.n_iters,
        }
        """
            
 
if __name__ == "__main__":
    task(
        loss_jax=linear.loss_jax,
        data_generator=linear.data_generator,
        sample_size=200,
        dim=500,
        sparsity_level=50,
        seed=110,
    )
