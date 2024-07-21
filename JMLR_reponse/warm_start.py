import parallel_experiment_util
from skscope import (
    ScopeSolver,
    GraspSolver,
    IHTSolver,
    HTPSolver,
    FobaSolver,
    OMPSolver,
)
from skscope.utilities import LinearSIC, SIC
import time
import numpy as np
from skscope_experiment import model as Model
import cvxpy as cp

model_dict = {
    "linear": Model.linear,
    "logistic": Model.logistic,
    "ising": Model.ising,
    "multitask": Model.multitask,
    "non_linear": Model.non_linear_non_additive_example,
    "trend_filter": Model.trend_filtering_1d,
    "robust_ESL": Model.robust_ESL,
}


def task(model: str, sample_size, dim, sparsity_level, seed):
    results = []
    true_params, data = model_dict[model].data_generator(sample_size, dim, sparsity_level, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    loss_jax = lambda x: model_dict[model].loss_jax(x, data)
    ic_method = LinearSIC if model == "linear" else SIC
    for warm_start in [True, False]:
        solver = ScopeSolver(dim, sample_size=sample_size, ic_method=ic_method)
        solver.warm_start = warm_start
        t1 = time.time()
        solver.solve(loss_jax, jit=True)
        t2 = time.time()
        results.append(
            {
                "method": "warm_start" if warm_start else "no_warm_start",
                "time": t2 - t1,
                "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level,
                "support_size": len(solver.get_support()),
            }
        )

    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "sample_size", "dim", "sparsity_level", "seed"],
        out_keys=["method", "time", "accuracy", "support_size"],
        processes=8,
        name="warm_start",
        memory_limit=0.9,
    )
    if False:
        # experiment.check(model="multitask", sample_size=600, dim=500, sparsity_level=50, seed=1)
        # experiment.check(model="non_linear", sample_size=600, dim=500, sparsity_level=50, seed=1)
        experiment.check(model="linear", sample_size=600, dim=500, sparsity_level=50, seed=294)
        experiment.check(model="logistic", sample_size=600, dim=500, sparsity_level=50, seed=100)
        # experiment.check(model="ising", sample_size=600, dim=190, sparsity_level=40, seed=200)
        experiment.check(model="trend_filter", sample_size=600, dim=600, sparsity_level=50, seed=90)
        # experiment.check(model="robust_ESL", sample_size=60, dim=50, sparsity_level=5, seed=900)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["linear"],
                "sample_size": [600],
                "dim": [500],
                "sparsity_level": [50],
            },
            {
                "model": ["logistic"],
                "sample_size": [600],
                "dim": [500],
                "sparsity_level": [50],
            },
            {
                "model": ["trend_filter"],
                "sample_size": [600],
                "dim": [600],
                "sparsity_level": [50],
            },
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()

"""
            {
                "model": ["robust_ESL"],
                "sample_size": [600],
                "dim": [500],
                "sparsity_level": [50],
            },
            {
                "model": ["non_linear"],
                "sample_size": [600],
                "dim": [50],
                "sparsity_level": [10],
            },
            {
                "model": ["trend_filter"],
                "sample_size": [600],
                "dim": [600],
                "sparsity_level": [50],
            },
            {
                "model": ["linear", "logistic"],
                "sample_size": [600],
                "dim": [500],
                "sparsity_level": [50],
            },
            {
                "model": ["ising"],
                "sample_size": [600],
                "dim": [190],
                "sparsity_level": [40],
            },
"""
