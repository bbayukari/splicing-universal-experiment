from itertools import product
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import abess
import statistic_model_pybind
import time
import parallel_experiment_util


def miqp(features, response, non_zero):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    with gp.Env() as env, gp.Model(env=env) as regressor:
        regressor.setParam("output_flag",0)
        regressor.setParam("threads", 1)
        regressor.setParam("node_file_start", 0.5)
        regressor.setParam("time_limit",10000)
        # Append a column of ones to the feature matrix to account for the y-intercept
        X = np.concatenate([features, np.ones((samples, 1))], axis=1)

        # Decision variables
        beta = regressor.addVars(dim + 1, lb=-GRB.INFINITY, name="beta")  # Weights
        intercept = beta[dim]  # Last decision variable captures the y-intercept
        intercept.varname = "intercept"
        # iszero[i] = 1 if beta[i] = 0
        iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

        # Objective Function (OF): minimize 1/2 * RSS using the fact that
        # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
        Quad = np.dot(X.T, X)
        lin = np.dot(response.T, X)
        obj = sum(
            0.5 * Quad[i, j] * beta[i] * beta[j]
            for i, j in product(range(dim + 1), repeat=2)
        )
        obj -= sum(lin[i] * beta[i] for i in range(dim + 1))
        obj += 0.5 * np.dot(response, response)
        regressor.setObjective(obj, GRB.MINIMIZE)

        # Constraint sets
        for i in range(dim):
            # If iszero[i]=1, then beta[i] = 0
            regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
        regressor.addConstr(iszero.sum() == dim - non_zero)  # Budget constraint

        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
        return intercept.X, coeff

def scope_linear(X, y, k):
    n, p = X.shape
    model = abess.ConvexSparseSolver(model_size=p, support_size=k)
    model.set_loss_custom(
        statistic_model_pybind.linear_loss_no_intercept,
        statistic_model_pybind.linear_gradient_no_intercept,
        statistic_model_pybind.linear_hessian_no_intercept,
    )
    data_set = statistic_model_pybind.RegressionData(X, y)

    model.fit(data_set)
    return model.get_solution()

def task(n, p, k, seed):
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
        snr=10 * np.log10(1),
        standardize=True,
        coef_=coef
    )

    t1 = time.time()
    _, gurobi_coef = miqp(data.x, data.y, k)
    t2 = time.time()
    scope_coef = scope_linear(data.x, data.y, k)
    t3 = time.time()

    return {
        "GUROBI_time" : t2 - t1,
        "SCOPE_time" : t3 - t2,
        "GUROBI_accuracy" : parallel_experiment_util.accuracy(gurobi_coef, coef),
        "SCOPE_accuracy" : parallel_experiment_util.accuracy(scope_coef, coef),
    }

if __name__ == "__main__":
    in_keys = ["n", "p", "k", "seed"]
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
        processes=40,
        name="gurobi_experiment",
        memory_limit=80,
    )

    if False:
        experiment.check(n=10, p=5, k=3, seed=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [i * 100 + 100 for i in range(10)], "p": [100], "k": [10]},
            {"n": [100], "p": [i * 10 + 10 for i in range(10)], "k": [10]},
            {"n": [50], "p": [50], "k": [i + 1 for i in range(10)]},
            repeat=100,
            seed=1
        )
            
        experiment.run(parameters)
        experiment.save()
