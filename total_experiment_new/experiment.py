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
    # SCOPE, GraSP, HTP_1
    for method, solver in {
        "SCOPE": scope.ScopeSolver(dim, sparsity_level, max_exchange_num=k),
        "GraSP": scope.GraspSolver(dim, sparsity_level),
        "GraHTP_1": scope.HTPSolver(dim, sparsity_level),
    }.items():
        t1 = time.perf_counter()
        if method == "GraHTP_1":
            step_size = (
                1
                / 2
                / jax.numpy.max(
                    jax.numpy.diagonal(jax.hessian(loss_jax_data)(np.zeros(dim)))
                ).item()
            )
            solver.set_params(step_size=step_size)
        if method == "SCOPE":
            solver.solve(loss_cpp_data, gradient=grad_cpp_data)
        else:
            solver.solve(loss_jax_data)
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

    # HTP_2
    step_sizes = [1.0, 0.1, 0.01, 0.001, 0.0001]
    solver = scope.HTPSolver(dim, sparsity_level)
    best_loss = np.inf
    result = {"method": "GraHTP_2"}
    t1 = time.perf_counter()
    for step_size in step_sizes:
        solver.set_params(step_size=step_size)
        solver.solve(loss_jax_data)
        loss = solver.value_of_objective
        if loss < best_loss:
            best_loss = loss
            params = solver.get_estimated_params()
            support_set = set(np.argpartition(np.abs(params), -k)[-k:])
            result["accuracy"] = len(support_set & true_support_set) / k
    result["time"] = time.perf_counter() - t1
    results.append(result)

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
        name="total_relax_sparsity-2",
        memory_limit=0.9,
    )

    if False:
        # experiment.check(model="linear", n=200, p=500, k=50, seed=11)
        # experiment.check(model="logistic", n=1000, p=500, k=50, seed=1099)
        experiment.check(model="ising", n=200, p=190, k=40, seed=708, relax_ratio=1.0)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["linear", "logistic"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
                "relax_ratio": [1.0, 1.5, 2.0],
            },
            {
                "model": ["ising"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [190],
                "k": [40],
                "relax_ratio": [1.0, 1.5, 2.0],
            },
            repeat=95,
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
