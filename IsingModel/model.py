from random import sample
import numpy as np
import util as util_cpp
from abess.universal import ConvexSparseSolver
from sklearn.linear_model import LogisticRegression


def data_generate(P, N, Edges, seed):
    """
    Generate data for Ising model
    :param P: number of variables
    :param N: number of samples
    :param Edges: number of Edges
    :return: data (2^p * p+1), theta, coef
    """
    np.random.seed(seed)
    theta = np.diag((np.random.randint(3,size = P) - 1) / 2)
    E = np.random.choice(int(P*(P-1)/2), Edges, replace=False)
    flatten_theta = np.zeros(int(P*(P-1)/2))
    flatten_theta[E] = np.random.randint(2, size=Edges) - 0.5

    idx = 0
    for i in range(P):
        for j in range(i+1,P):
            if idx in E:
                theta[i,j] = flatten_theta[idx]
                theta[j,i] = theta[i,j]
            idx += 1

    data = util_cpp.sample_by_conf(N, theta, seed)
    return data[np.where(data[:,0] > 0.5)[0],], theta, flatten_theta


def abess(data, P, Edges):
    """
    Fit Ising model using abess
    :param data: data
    :param P: number of variables, model_size is P*(P-1)/2
    :param Edges: number of Edges, support_size is Edges
    :return: estimated theta
    """
    model = ConvexSparseSolver(model_size=int(P*(P-1)/2), support_size=Edges, intercept_size=P)
    model.set_data(util_cpp.IsingData(data))
    model.set_model_autodiff(util_cpp.loss_ising_model, util_cpp.gradient_ising_model, util_cpp.hessian_ising_model)
    model.fit()
    return model.coef_

def wainwright(data, Edges, method='min', C=1.0, tol=1e-4):
    p = data.shape[1] - 1
    sample_weight = data[:,0]
    data = data[:,1:]
    theta = np.zeros((p,p))
    C_up = 3 * C
    C_low = 0.0

    while True:
        for s in range(p):
            # estimate s-th row of theta
            mask = np.full(p, True, dtype=bool)
            mask[s] = False
            theta[s,mask] = LogisticRegression(penalty='l1',solver='liblinear',C=C).fit(data[:,mask], data[:,s], sample_weight=sample_weight).coef_
        E = np.array(abs(theta) > tol)
        E = np.logical_and(E, E.T) if method == 'min' else np.logical_or(E, E.T)
        edges_estimate = np.sum(E) / 2
        if edges_estimate > Edges:
            C_up = C
            C = (C_low + C) / 2
        elif edges_estimate < Edges:
            C_low = C
            C = (C_up + C) / 2
        else:
            break
    
    ## flatten theta
    flatten_theta = np.zeros(int(p*(p-1)/2))
    idx = 0
    for i in range(p):
        for j in range(i+1,p):
            flatten_theta[idx] = E[i,j]
            idx += 1
    return flatten_theta


if __name__ == "__main__":
    P = 20
    N = 100
    Edges = 30
    seed = 113
    data, theta, coef = data_generate(P, N, Edges, seed)
    print('theta:\n', np.nonzero(coef))
    print(np.nonzero(wainwright(data,Edges)))
