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
        ("linear", linear, 500, 50),
        ("logistic", logistic, 500, 50),
        ("ising", ising, 190, 50),
    ]:
        true_params, data = model.data_generator(n, p, s, seed)
        true_support_set = set(np.nonzero(true_params)[0])
        loss_jax_data = lambda x: model.loss_jax(x, data)
        for dynamic in [True, False]:
            for splicing_type in ["halve", "taper"]:
                for k_max in [1, 5, 10, 30, 50]:
                    for greedy in [True, False]:
                        solver = ScopeSolver(
                            p,
                            s,
                            numeric_solver=count_convex_solver,
                            greedy=greedy,
                            max_exchange_num=k_max,
                            splicing_type=splicing_type,
                            use_hessian=True,
                            is_dynamic_max_exchange_num=dynamic,
                        )
                        optim_times = 0
                        t1 = time.perf_counter()
                        solver.solve(
                            loss_jax_data,
                            hessian=lambda x: np.eye(p),
                            jit=True,
                        )
                        t2 = time.perf_counter()
                        support_set = set(solver.get_support())
                        results.append(
                            {
                                "dynamic": dynamic,
                                "splicing_type": splicing_type,
                                "k_max": k_max,
                                "greedy": greedy,
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
        out_keys=[
            "dynamic",
            "splicing_type",
            "k_max",
            "greedy",
            "time",
            "accuracy",
            "n_iters",
            "model",
            "optim_times",
        ],
        processes=7,
        name="survey_exchange_num_strategy-2",
        memory_limit=0.9,
    )

    if False:
        import cProfile
        import pstats

        cProfile.run("experiment.check(n=600, seed=144)", "profile_data")
        p = pstats.Stats("profile_data")
        p.sort_stats("cumulative").print_stats(200)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "n": [i * 100 + 100 for i in range(10)],
            },
            repeat=7,
            seed=100,
        )

        experiment.run(parameters)
        experiment.save()
