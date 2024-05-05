from itertools import product
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import skscope
import time
import parallel_experiment_util
from skscope_experiment.model import linear

TIME_LIMIT = 1000


def linear_gurobi(features, response, non_zero):
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    with gp.Env() as env, gp.Model(env=env) as regressor:
        regressor.setParam("output_flag", 0)
        regressor.setParam("threads", 1)
        regressor.setParam("node_file_start", 0.5)
        regressor.setParam("time_limit", TIME_LIMIT)
        # Append a column of ones to the feature matrix to account for the y-intercept
        X = features

        # Decision variables
        beta = regressor.addVars(dim, lb=-GRB.INFINITY, name="beta")  # Weights
        # iszero[i] = 1 if beta[i] = 0
        iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

        # Objective Function (OF): minimize 1/2 * RSS using the fact that
        # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
        Quad = np.dot(X.T, X)
        lin = np.dot(response.T, X)
        obj = sum(
            0.5 * Quad[i, j] * beta[i] * beta[j]
            for i, j in product(range(dim), repeat=2)
        )
        obj -= sum(lin[i] * beta[i] for i in range(dim))
        obj += 0.5 * np.dot(response, response)
        regressor.setObjective(obj, GRB.MINIMIZE)

        # Constraint sets
        for i in range(dim):
            # If iszero[i]=1, then beta[i] = 0
            regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
        regressor.addConstr(iszero.sum() == dim - non_zero)  # Budget constraint

        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
    return coeff


def linear_scope(features, response, non_zero):
    n, p = features.shape
    model = skscope.ScopeSolver(p, non_zero)
    model.cpp = True
    return model.solve(
        linear.loss_cpp,
        gradient=linear.grad_cpp,
        data=linear.data_cpp_wrapper((features, response)),
    )


def task(n, p, k, seed, rho, snr):
    true_params, (features, response) = linear.data_generator(n, p, k, seed, rho, snr)

    t1 = time.perf_counter()
    gurobi_coef = linear_gurobi(features, response, k)
    t2 = time.perf_counter()
    scope_coef = linear_scope(features, response, k)
    t3 = time.perf_counter()

    return {
        "GUROBI_time": t2 - t1,
        "SCOPE_time": t3 - t2,
        "GUROBI_accuracy": len(
            set(np.nonzero(gurobi_coef)[0]) & set(np.nonzero(true_params)[0])
        )
        / k,
        "SCOPE_accuracy": len(
            set(np.nonzero(scope_coef)[0]) & set(np.nonzero(true_params)[0])
        )
        / k,
    }


if __name__ == "__main__":
    in_keys = ["n", "p", "k", "seed", "rho", "snr"]
    out_keys = [
        "SCOPE_accuracy",
        "SCOPE_time",
        "GUROBI_accuracy",
        "GUROBI_time",
    ]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=20,
        name="gurobi_experiment_final_2",
        memory_limit=0.9,
    )

    if False:
        experiment.check(n=50, p=100, k=8, seed=719, rho=0.6, snr=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "n": [i * 100 + 600 for i in range(5)],
                "p": [100],
                "k": [10],
                "rho": [0.6],
                "snr": [1],
            },
            repeat=100,
            seed=10000,
        )

        experiment.run(parameters)
        experiment.save()


"""
            {
                "n": [50],
                "p": [100],
                "k": [i * 1 + 1 for i in range(10)],
                "rho": [0.6],
                "snr": [1],
            },
            {
                "n": [50],
                "p": [i * 10 + 10 for i in range(10)],
                "k": [10],
                "rho": [0.6],
                "snr": [1],
            },
            {
                "n": [i * 50 + 50 for i in range(10)],
                "p": [100],
                "k": [10],
                "rho": [0.6],
                "snr": [1],
            },
"""
