from math import exp, log
import numpy as np


def loss_quadratic(x, data, active_index=None):
    """
    compute value of quadratic function
        x: array with the shape of (p,)
        data: dictionary
            Q: array with the shape of (p,p)
            v: array with the shape of (p,)
        active_index:  int array, the index of nonzore features
    Returns:
        value: x'Qx + x'v
    """
    if active_index is None:
        Q = data["Q"]
        v = data["v"]
    else:
        x = x[active_index]
        Q = data["Q"][np.ix_(active_index, active_index)]
        v = data["v"][active_index]
    return np.dot(x, np.matmul(Q, x)) + np.dot(x, v)


def grad_quadratic(x, data, active_index=None, compute_index=None):
    """
    compute grad of quadratic function
    Args:
        x: array with the shape of (p,)
        data: dictionary
            Q: array with the shape of (p,p)
            v: array with the shape of (p,)
        active_index:  int array, the index of nonzore features
        compute_index:  int array, the index of grad to compute
    Returns:
        grad: 2Qx + v on compute_index
    """
    if active_index is None:
        active_index = np.arange(x.size)
    if compute_index is None:
        compute_index = active_index

    x = x[active_index]
    Q = data["Q"][np.ix_(compute_index, active_index)]
    v = data["v"][compute_index]

    return 2 * np.matmul(Q, x) + v


def hessian_quadratic(x, data, compute_index):
    """
    compute hessian of quadratic function
    Args:
        x: array with the shape of (p,)
        data: dictionary
            Q: array with the shape of (p,p)
        compute_index:  int array, the index of hessian to compute
    Returns:
        hessian: the diage of 2Q on compute_index
    """
    return 2 * np.diag(data["Q"])[compute_index]


def loss_logistic(beta, data, active_index=None):
    """
    compute value of logistic function
    Args:
        beta: array with the shape of (p,)
        data: class
            x: array with the shape of (n,p)
            y: array with the shape of (n,)
        active_index:  int array, the index of nonzore features
    Returns:
        value: 1'log(1 + exp(Xbeta)) - y'Xbeta
    """
    if active_index is None:
        active_index = np.arange(beta.size)
    Xbeta = np.matmul(data.x[:, active_index], beta[active_index])
    return sum([x if x > 100 else 0.0 if x < -100 else log(1+exp(x)) for x in Xbeta]) - np.dot(data.y, Xbeta)


def grad_logistic(beta, data, active_index=None, compute_index=None):
    """
    compute grad of logistic function
    Args:
        beta: array with the shape of (p,)
        data: class
            x: array with the shape of (n,p)
            y: array with the shape of (n,)
        active_index:  int array, the index of nonzore features
        compute_index:  int array, the index of grad to compute
    Returns:
        grad: X(1/(1+exp(-Xbeta)) - y) on compute_index
    """
    if active_index is None:
        active_index = np.arange(beta.size)
    if compute_index is None:
        compute_index = active_index

    Xbeta = np.matmul(data.x[:, active_index], beta[active_index])
    return np.matmul(
        data.x[:, compute_index].T,
        np.array([0.0 if x<-100 else 1.0 if x>100 else 1/(1+exp(-x)) for x in Xbeta]) - data.y
    )

def hessian_logistic(beta, data, active_index=None, compute_index=None):
    """
    compute hessian of logistic function
    Args:
        beta: array with the shape of (p,)
        data: class
            x: array with the shape of (n,p)
            y: array with the shape of (n,)
        active_index:  int array, the index of nonzore features
        compute_index:  int array, the index of hessian to compute
    Returns:
        hessian: X'diag(1/(exp(Xbeta)+exp(-Xbeta)+2))X on compute_index
    """
    if active_index is None:
        active_index = np.arange(beta.size)
    if compute_index is None:
        compute_index = active_index

    Xbeta = np.matmul(data.x[:, active_index], beta[active_index])
    return np.matmul(
        data.x[:, compute_index].T,
        np.array([1/(exp(x)+exp(-x)+2) if abs(x)<100 else 0.0 for x in Xbeta])[:,np.newaxis] 
        * data.x[:, compute_index]
    )
   