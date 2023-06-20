import scope
import time
import numpy as np
import cvxpy as cp

def task(loss_jax, loss_cvxpy, data_generator, sample_size, dim, sparsity_level, seed, cvxpy_constraints, L1_init=10.0):
    results = []
    true_params, data = data_generator(sample_size, dim, sparsity_level, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    loss_jax_data = lambda x: loss_jax(x, data)
    loss_cvxpy_data = lambda x: loss_cvxpy(x, data)
    # SCOPE, GraSP
    for method, solver in {
        "SCOPE": scope.ScopeSolver(dim, sparsity_level, greedy=False),
        "GraSP": scope.GraspSolver(dim, sparsity_level),
    }.items():
        for jit in [True]:#, False]:
            t1 = time.perf_counter()
            solver.solve(loss_jax_data, jit=jit)
            t2 = time.perf_counter()
            support_set = set(solver.get_support())
            results.append(
                {
                    "method": method + "_jit" if jit else method,
                    "time": t2 - t1,
                    "accuracy": len(support_set & true_support_set) / sparsity_level,
                    "n_iters": solver.n_iters,
                }
            )
    # Foba
    t1 = time.perf_counter()
    solver.solve(loss_jax_data, jit=True)
    t2 = time.perf_counter()
    results.append(
        {
            "method": "Foba_jit",
            "time": t2 - t1,
            "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level,
            "n_iters": 0,
        }
    )
    # OMP
    for jit in [True]:#, False]:
        t1 = time.perf_counter()
        solver.solve(loss_jax_data, jit=jit)
        t2 = time.perf_counter()
        results.append(
            {
                "method": "OMP_jit",
                "time": t2 - t1,
                "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level,
                "n_iters": 0,
            }
        )
    # IHT, HTP
    step_sizes = np.logspace(-4, -2, 10)
    for method, solver in {
        "IHT": scope.IHTSolver(dim, sparsity_level),
        "HTP": scope.HTPSolver(dim, sparsity_level),
    }.items():
        for jit in [True]:#, False]:
            best_loss = np.inf
            for step_size in step_sizes:
                solver.set_params(step_size=step_size)
                t1 = time.perf_counter()
                solver.solve(loss_jax_data, jit=jit)
                t2 = time.perf_counter()
                loss = solver.value_of_objective
                if loss < best_loss:
                    best_loss = loss
                    result = {
                        "method": method + "_jit" if jit else method,
                        "time": t2 - t1,
                        "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level,
                        "n_iters": solver.n_iters + step_size,
                    }
            results.append(result)
    # CVXPY
    def object_fn(x, lambd):
        return loss_cvxpy_data(x) + lambd * cp.norm1(x)
    x = cp.Variable(dim)
    lambd = cp.Parameter(nonneg=True)
    if cvxpy_constraints is None:
        problem = cp.Problem(cp.Minimize(object_fn(x, lambd)))
    else:
        problem = cp.Problem(cp.Minimize(object_fn(x, lambd)), cvxpy_constraints(x))
    lambd_lowwer = 0.0
    lambd.value = L1_init
    for _ in range(100):
        try:
            start = time.perf_counter()
            problem.solve()
            end = time.perf_counter()
        except:
            return results
        params = x.value
        support_size_est = np.array(abs(params) > 1e-2).sum() 

        if support_size_est > sparsity_level:
            lambd_lowwer = lambd.value
            lambd.value = 2 * lambd.value
        elif support_size_est < sparsity_level:
            lambd.value = (lambd_lowwer + lambd.value) / 2
        else:
            break
    results.append(
        {
            "method": "CVXPY",
            "time": end - start,
            "accuracy": len(set(np.where(abs(params) > 1e-2)[0]) & true_support_set) / sparsity_level,
            "n_iters": lambd.value,
        }
    )

    return results

def multi_target_task(loss_jax, loss_cvxpy, data_generator, sample_size, n_features, sparsity_level, n_targets, seed):
    results = []
    true_params, data = data_generator(sample_size, n_features, sparsity_level, n_targets, seed)
    true_support_set = set(np.nonzero(true_params)[0])
    group = [i for i in range(n_features) for _ in range(n_targets)]
    loss_jax_data = lambda x: loss_jax(x, data)
    loss_cvxpy_data = lambda x: loss_cvxpy(x, data)
    # SCOPE, GraSP
    for method, solver in {
        "SCOPE": scope.ScopeSolver(n_features * n_targets, sparsity_level, group=group),
        "GraSP": scope.GraspSolver(n_features * n_targets, sparsity_level, group=group),
    }.items():
        for jit in [True]:#, False]:
            t1 = time.perf_counter()
            solver.solve(loss_jax_data, jit=jit)
            t2 = time.perf_counter()
            support_set = set(solver.get_support())
            results.append(
                {
                    "method": method + "_jit" if jit else method,
                    "time": t2 - t1,
                    "accuracy": len(support_set & true_support_set) / sparsity_level / n_targets,
                    "n_iters": solver.n_iters,
                }
            )
    # Foba
    t1 = time.perf_counter()
    solver.solve(loss_jax_data, jit=True)
    t2 = time.perf_counter()
    results.append(
        {
            "method": "Foba_jit",
            "time": t2 - t1,
            "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level / n_targets,
            "n_iters": 0,
        }
    )
    # OMP
    for jit in [True]:#, False]:
        t1 = time.perf_counter()
        solver.solve(loss_jax_data, jit=jit)
        t2 = time.perf_counter()
        results.append(
            {
                "method": "OMP_jit" if jit else method,
                "time": t2 - t1,
                "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level / n_targets,
                "n_iters": 0,
            }
        )
    # IHT, HTP
    step_sizes = np.logspace(-4, -2, 10)
    for method, solver in {
        "IHT": scope.IHTSolver(n_features * n_targets, sparsity_level, group=group),
        "HTP": scope.HTPSolver(n_features * n_targets, sparsity_level, group=group),
    }.items():
        for jit in [True]:#, False]:
            best_loss = np.inf
            for step_size in step_sizes:
                solver.set_params(step_size=step_size)
                t1 = time.perf_counter()
                solver.solve(loss_jax_data, jit=jit)
                t2 = time.perf_counter()
                loss = solver.value_of_objective
                if loss < best_loss:
                    best_loss = loss
                    result = {
                        "method": method + "_jit" if jit else method,
                        "time": t2 - t1,
                        "accuracy": len(set(solver.get_support()) & true_support_set) / sparsity_level / n_targets,
                        "n_iters": solver.n_iters,
                    }
            results.append(result)
    # CVXPY
    def object_fn(x, lambd):
        return loss_cvxpy_data(x) + lambd * cp.mixed_norm(x)
    x = cp.Variable((n_features, n_targets))
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(object_fn(x, lambd)))
    lambd_lowwer = 0.0
    lambd.value = 100.0
    for _ in range(100):
        try:
            start = time.perf_counter()
            problem.solve()
            end = time.perf_counter()
        except:
            return results
        params = x.value.flatten()
        support_size_est = np.array(abs(params) > 1e-2).sum() 

        if support_size_est > sparsity_level * n_targets:
            lambd_lowwer = lambd.value
            lambd.value = 2 * lambd.value
        elif support_size_est < sparsity_level * n_targets:
            lambd.value = (lambd_lowwer + lambd.value) / 2
        else:
            break
    results.append(
        {
            "method": "CVXPY",
            "time": end - start,
            "accuracy": len(set(np.where(abs(params) > 1e-2)[0]) & true_support_set) / sparsity_level / n_targets,
            "n_iters": 0,
        }
    )

    return results   