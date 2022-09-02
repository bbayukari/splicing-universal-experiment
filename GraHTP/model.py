import numpy as np

def quadratic(x, supp, data):
    """
    compute value of quadratic function and its gradient on supp
    Args:
        x: array with the shape of (p,)
        data: dictionary
            A: array with the shape of (p,p)
            B: array with the shape of (p,)
        supp:  int array with the shape of (k,)
    Returns:
        value: x'Ax + x'B
        grad: 2Ax + B on supp
    """
    value = x.T @ data["A"] @ x + x.T @ data["B"]
    if supp.size > 0:
        grad = 2 * data["A"] @ x + data["B"]
        grad = grad[supp]
    else:
        grad = None

    return (value, grad)

def logistic(x, supp, data):
    """
    compute value of logistic function and its gradient on supp
    Args:
        x: array with the shape of (p,)
        data: dictionary
            X: array with the shape of (n,p)
            y: array with the shape of (n,)
        supp:  int array with the shape of (k,)
    Returns:
        value: log(1 + exp(-yXx))
        grad: -yX'exp(-yXx)/(1 + exp(-yXx)) on supp
    """
    X = data["X"]
    y = data["y"]
    value = np.log(1 + np.exp(-y * X @ x))
    if supp.size > 0:
        grad = -y * X[:, supp].T @ (np.exp(-y * X @ x) / (1 + np.exp(-y * X @ x)))
    else:
        grad = None

    return (value, grad)
