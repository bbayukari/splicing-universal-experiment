import parallel_experiment_util
from skscope import (
    ScopeSolver,
    GraspSolver,
    IHTSolver,
    HTPSolver,
    FobaSolver,
    OMPSolver,
)
import time
import numpy as np
from skscope_experiment import model as Model

model_dict = {
    "linear": Model.linear,
    "logistic": Model.logistic,
    "ising": Model.ising,
    "multitask": Model.multitask,
    "non_linear": Model.non_linear_non_additive_example,
    "trend_filter": Model.trend_filtering_1d,
}


def task(model: str, sample_size, dim, sparsity_level, seed):
    results = []
    n_outputs = 3 if model == "multitask" else 1
    true_params, data = model_dict[model].data_generator(
        sample_size, dim, sparsity_level, seed
    )
    true_support_set = set(np.nonzero(true_params)[0])
    loss_data = lambda x: model_dict[model].loss_jax(x, data)
    group = [i for i in range(dim) for _ in range(n_outputs)]
    dim = dim * n_outputs
    sparsity_level = sparsity_level * n_outputs
    for method, solver in {
        "SCOPE": ScopeSolver(dim, sparsity_level, group=group),
        #"GraSP": GraspSolver(dim, sparsity_level, group=group),
        #"FoBa": FobaSolver(dim, sparsity_level, group=group),
        #"OMP": OMPSolver(dim, sparsity_level, group=group),
    }.items():
        for is_jit in [True, False]:
            t1 = time.perf_counter()
            solver.solve(loss_data, jit=True)
            t2 = time.perf_counter()
            results.append(
                {
                    "jit": is_jit,
                    "method": method,
                    "time": t2 - t1,
                    "accuracy": len(set(solver.get_support()) & true_support_set)
                    / sparsity_level,
                }
            )
    """
    # IHT, HTP
    for method, solver in {
        "IHT": IHTSolver(dim, sparsity_level, group=group),
        "HTP": HTPSolver(dim, sparsity_level, group=group),
    }.items():
        for is_jit in [True, False]:
            t1 = time.perf_counter()
            best_loss = np.inf
            for step_size in [1.0, 0.1, 0.01, 0.001, 0.0001]:
                solver.set_params(step_size=step_size)
                try:
                    solver.solve(loss_data, jit=is_jit)
                except:
                    continue
                loss = solver.objective_value
                if loss < best_loss:
                    best_loss = loss
                    result = {
                        "jit": is_jit,
                        "method": method,
                        "time": time.perf_counter() - t1,
                        "accuracy": len(set(solver.get_support()) & true_support_set)
                        / sparsity_level,
                    }
            results.append(result)
    """
    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "sample_size", "dim", "sparsity_level", "seed"],
        out_keys=["jit", "method", "time", "accuracy"],
        processes=16,
        name="skscope_experiment_jit",
        memory_limit=0.9,
    )
    import cProfile, pstats
    if True:
        cProfile.run(
            'experiment.check(model="trend_filter", sample_size=600, dim=600, sparsity_level=5, seed=901)',
            "profile_data",
        )
        p = pstats.Stats("profile_data")
        p.sort_stats("cumulative").print_stats(200)
        # experiment.check(model="multitask", sample_size=600, dim=500, sparsity_level=50, seed=1)
        # experiment.check(model="non_linear", sample_size=600, dim=50, sparsity_level=10, seed=1)
        # experiment.check(model="linear", sample_size=600, dim=500, sparsity_level=50, seed=294)
        # experiment.check(model="logistic", sample_size=600, dim=500, sparsity_level=50, seed=100)
        # experiment.check(model="ising", sample_size=600, dim=190, sparsity_level=40, seed=200)
        # experiment.check(model="trend_filter", sample_size=600, dim=600, sparsity_level=5, seed=901)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["multitask"],
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
                "sparsity_level": [5],
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
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()

"""
            {
                "model": ["multitask"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
            },
            {
                "model": ["additive"],
                "n": [i * 20 + 20 for i in range(10)],
                "p": [100],
                "k": [4],
            },
            {
                "model": ["non_additive"],
                "n": [i * 10 + 10 for i in range(10)],
                "p": [100],
                "k": [3],
            },
            {
                "model": ["trend_filter"],
                "n": [100],
                "p": [100],
                "k": [i+1 for i in range(10)],
            },
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
                "k": [30],
            },
"""
