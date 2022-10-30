import numpy as np
import gflsegpy as seg
from abess.universal import ConvexSparseSolver
from abess import pybind_cabess

def mse(Y, cp):
    cp = np.append(np.insert(np.sort(cp), 0, 0), Y.shape[0]).astype(np.int32)
    mse = 0
    for i in range(1,len(cp)):
        mse += np.square(Y[cp[i-1]:cp[i], :] - np.mean(Y[cp[i-1]:cp[i], :], axis=0)).sum()
    return mse


def gfl(Y, support_size):
    # select lambda with cross validation
    lambdas = np.logspace(0, 2, 5)
    min_mse = np.inf
    best_lambda = 0
    best_cp = []
    for lambda_ in lambdas:
        cp = seg.gfl_coord(Y,lambda_,support_size)
        if lambda_==lambdas[0]:
            cp_1 = cp.copy()
        if lambda_==lambdas[2]:
            cp_10 = cp.copy()
        mse_ = mse(Y, cp)
        if mse_ < min_mse:
            min_mse = mse_
            best_lambda = lambda_
            best_cp = cp.copy()
    return best_cp, best_lambda, cp_1, cp_10

def abess(X, Y, support_size):
    n, p = Y.shape
    group = [i for i in range(n-1) for j in range(p)]
    model = ConvexSparseSolver(
        model_size=p * (n-1), sample_size=n, intercept_size=p, group=group, support_size=support_size
    )
    model.set_data(pybind_cabess.Data(X, Y))
    model.set_model_autodiff(
        pybind_cabess.loss_linear,
        pybind_cabess.gradient_linear,
        pybind_cabess.hessian_linear,
    )
    model.fit()
    return np.unique(np.floor(np.nonzero(model.coef_)[0] / p) + 1).astype(np.int32)


if __name__ == "__main__":
    n = 100
    p = 5
    k = 9
    sigma = np.sqrt(1)
    change_point = np.arange(n,step = int(n/(k+1)))[1:]
    Z = np.zeros((n, p))
    W = np.random.normal(size=(n, p)) * sigma
    for i in range(1,n):
        Z[i, :] = (Z[i-1, :] + np.random.normal(size=(1, p))) if i in change_point else Z[i-1, :]
    Y = Z + W
    Y = Y - np.mean(Y, axis=0)
    Y.reshape(n,p)

    X = np.zeros((n, n - 1))
    for j in range(1,n):        
        for i in range(1,j+1):
            X[i-1, j-1] = -np.sqrt((n-j) / (n * j))
        for i in range(j+1,n+1):
            X[i-1, j-1] = np.sqrt(j / (n * (n - j)))
    
    print(gfl(Y,len(change_point))[0])
