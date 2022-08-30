import time
import numpy as np
import nlopt
import MyTest


def GraHTP(
    f,
    p,
    k,
    data=None,
    fast=False,
    relaxed_sparsity_level=0,
    init_x=None,
    step_size=0.1,
    max_iter=10000,
):
    """
    Select features by GraHTP algorithm

    Args:
        f: f(x, supp, data) -> (value, gradient_supp)

        p: length of the args 'x' in f

        k: the number of selected features

        fast: bool, whether to use FGraHTP algorithm or not

        relaxed_sparsity_level: int, must larger than k, algorithm will select relaxed_sparsity_level features firstly,
            then preserve the top k entries of the output as the final estimation.

        init_x: the initial value of the args 'x' in f

    Returns:
        suppport set: int array with the shape of (k,)
        estimator: array with the shape of (p,) which contains k nonzero entries
    """
    if init_x is None:
        init_x = np.zeros(p)
    
    final_support_size = k
    if relaxed_sparsity_level > k:   
        k = relaxed_sparsity_level
    
    # init
    _, grad = f(init_x, np.arange(p), data)
    x_old = - step_size * grad
    suppport_old = np.argpartition(np.abs(x_old), -k)[-k:]

    for _ in range(max_iter):
        # S1: gradient descent
        _, grad = f(x_old, np.arange(p), data)
        x_new = x_old - step_size * grad
        # S2: Gradient Hard Thresholding
        suppport_new = np.argpartition(np.abs(x_new), -k)[-k:]
        # S3: debise
        if fast:
            non_supp = np.array([True] * p)
            non_supp[suppport_new] = False
            x_new[non_supp] = 0.0
        else:
            def opt_f(x_supp, grad_supp):
                x_full = np.zeros(p)
                x_full[suppport_new] = x_supp
                if grad_supp.size > 0:
                    value, grad_supp[:] = f(x_full, suppport_new, data)
                    return value
                else:
                    value, _ = f(x_full, np.zeros(0), data)
                    return value

            x_opt_init = x_new[suppport_new].copy()
            opt = nlopt.opt(nlopt.LD_LBFGS, k)
            opt.set_min_objective(opt_f)
            x_new = np.zeros(p)
            x_new[suppport_new] = opt.optimize(x_opt_init)
        # terminating condition
        if np.all(set(suppport_old) == set(suppport_new)):
            break

    if k > final_support_size:
        suppport_new = np.argpartition(np.abs(x_new), -final_support_size)[
            -final_support_size:
        ]
        non_supp = np.array([True] * p)
        non_supp[suppport_new] = False
        x_new[non_supp] = 0.0

    return suppport_new, x_new

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

def GraHTP_quadratic(p,k, data):
    return GraHTP(quadratic, p, k, data)

if __name__ == "__main__":
    np.random.seed(2334)
    n = 500
    p = 500
    k = 20
    from abess import make_glm_data

    data_set = make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.2,
        family="gaussian",
        corr_type="exp",
        snr=10 * np.log10(6),
        standardize=True,
    )

    t1 = time.time()
    supp, beta = GraHTP(
        quadratic,
        p,
        k,
        #fast=True,
        #relaxed_sparsity_level=k,
        data={"A": data_set.x.T @ data_set.x, "B": -2 * data_set.x.T @ data_set.y}
    )
    t2 = time.time()

    print(t2 - t1)
    print(MyTest.accuracy(beta, data_set.coef_))
