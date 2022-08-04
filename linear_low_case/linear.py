import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error as mse

import gurobipy as gp
from gurobipy import GRB

from abess import pybind_cabess
from abess.universal import ConvexSparseSolver
from abess import LinearRegression


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

def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask, :]
    xtest = features[~train_mask, :]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest

def cross_validate(features, response, non_zero, folds, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(samples / folds)
    assert folds * fold_size <= samples
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_cv = 0
    # Exclude folds from training, one at a time,
    # to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size : (fold + 1) * fold_size]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)

        intercept, beta = miqp(xtrain, ytrain, non_zero)
        ypred = np.dot(xtest, beta) + intercept
        mse_cv += mse(ytest, ypred) / folds
    # Report the average out-of-sample MSE
    return mse_cv

def gurobi_linear(features, response, support_size, folds=5, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_mse = np.inf
    best = 0
    # Grid search to find best number of features to consider
    for i in support_size:
        val = cross_validate(features, response, i, folds=folds, seed=seed)
        if val < best_mse:
            best_mse = val
            best = i

    return miqp(features, response, best)

def autodiff_linear(features, response, non_zero):
    model = ConvexSparseSolver(model_size=features.shape[1], sample_size=features.shape[0], intercept_size=1, support_size=non_zero)
    model.set_model_autodiff(
        pybind_cabess.loss_linear,
        pybind_cabess.gradient_linear,
        pybind_cabess.hessian_linear,
    )
    model.fit(pybind_cabess.Data(features, response))
    return model.intercept_, model.coef_

def abess_linear(features, response, non_zero):
    model = LinearRegression(support_size=non_zero)
    model.fit(features, response)
    return model.intercept_, model.coef_


   
