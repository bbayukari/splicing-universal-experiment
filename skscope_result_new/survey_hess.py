from skscope_experiment import (
    linear,
    logistic,
    ising,
)
from scope import ScopeSolver
from scope.numeric_solver import convex_solver_nlopt
import numpy as np
import time
import parallel_experiment_util


def task(n, seed):
    results = []
    optim_times = 0

    def count_convex_solver(*args, **kwargs):
        nonlocal optim_times
        optim_times += 1
        return convex_solver_nlopt(*args, **kwargs)

    for name, model, p, s in [
        #("linear", linear, 500, 50),
        #("logistic", logistic, 500, 50),
        ("ising", ising, 190, 40),
    ]:
        true_params, data = model.data_generator(n, p, s, seed)
        true_support_set = set(np.nonzero(true_params)[0])
        loss_jax_data = lambda x: model.loss_jax(x, data)
        for hessian in [None, lambda x: np.eye(p)]:
            solver = ScopeSolver(p, s, numeric_solver=count_convex_solver)
            optim_times = 0
            t1 = time.perf_counter()
            solver.solve(
                loss_jax_data,
                hessian=hessian,
                jit=True,
            )
            t2 = time.perf_counter()
            support_set = set(solver.get_support())
            results.append(
                {
                    "type": "hess" if hessian is None else "non-hess",
                    "time": t2 - t1,
                    "accuracy": len(support_set & true_support_set) / s,
                    "n_iters": solver.n_iters,
                    "model": name,
                    "optim_times": optim_times,
                }
            )

    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["n", "seed"],
        out_keys=["type", "time", "accuracy", "n_iters", "model", "optim_times"],
        processes=8,
        name="survey_hess_ising",
        memory_limit=0.8,
    )

    if False:
        import cProfile
        import pstats                           
        cProfile.run("experiment.check(n=600, seed=144)", "profile_data")
        p = pstats.Stats('profile_data')
        p.sort_stats('cumulative').print_stats(200)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "n": [i * 100 + 100 for i in range(10)],
            },
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()
