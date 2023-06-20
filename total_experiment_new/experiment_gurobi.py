import scope
import time
import numpy as np
import abess
import parallel_experiment_util

from skscope_experiment import linear




def task(n, p, k, seed):
    sample_size=n
    dim=p
    sparsity_level=k

    results = []
    #true_params, data = model_dict[model].data_generator(sample_size, dim, sparsity_level, seed)
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([100, -100], k)
    data = abess.make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.2,
        family="gaussian",
        corr_type="exp",
        snr=10 * np.log10(60),
        sigma=0,
        coef_=coef
    )
    true_params, data = coef, (data.x, data.y)

    true_support_set = set(np.nonzero(true_params)[0])
    data_cpp = linear.data_cpp_wrapper(data)
    loss_cpp_data = lambda x: np.array(linear.loss_cpp(x, data_cpp))
    grad_cpp_data = lambda x: np.array(linear.grad_cpp(x, data_cpp))
  
    for method, solver in {
        "SCOPE": scope.ScopeSolver(dim, sparsity_level, max_exchange_num=k),
    }.items():
        t1 = time.perf_counter()
        solver.solve(loss_cpp_data, gradient=grad_cpp_data)
        t2 = time.perf_counter()
        params = solver.get_estimated_params()
        support_set = set(np.argpartition(np.abs(params), -k)[-k:])
        results.append(
            {
                "method": method,
                "time": t2 - t1,
                "accuracy": len(support_set & true_support_set) / k,
            }
        )

    return results


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["n", "p", "k", "seed"],
        out_keys=["method", "time", "accuracy"],
        processes=10,
        name="gurobi_experiment",
        memory_limit=0.9,
    )

    if False:
        experiment.check(n=200, p=500, k=50, seed=11)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [i * 100 + 100 for i in range(10)], "p": [100], "k": [10]},
            {"n": [100], "p": [i * 10 + 10 for i in range(10)], "k": [10]},
            {"n": [50], "p": [50], "k": [i + 1 for i in range(10)]},
            repeat=100,
            seed=1000,
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