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
    step_size=0.01,
    max_iter=100,
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
        estimator: array with the shape of (p,) which contains k nonzero entries
        iter: the number of iterations
    """
    if init_x is None:
        init_x = np.zeros(p)
    
    final_support_size = k
    if relaxed_sparsity_level > k:   
        k = relaxed_sparsity_level
    
    # init
    x_old = init_x
    support_old = np.argpartition(np.abs(x_old), -k)[-k:]

    for iter in range(max_iter):
        # S1: gradient descent
        objective_value_old, grad = f(x_old, np.arange(p), data)
        x_new_full = x_old - step_size * grad
        # S2: Gradient Hard Thresholding
        support_new = np.argpartition(np.abs(x_new_full), -k)[-k:]
        # S3: debise
        if fast:
            x_new = np.zeros(p)
            x_new[support_new] = x_new_full[support_new]
        else:
            try:
                def opt_f(x_supp, grad_supp):
                    x_full = np.zeros(p)
                    x_full[support_new] = x_supp
                    if grad_supp.size > 0:
                        value, grad_supp[:] = f(x_full, support_new, data)
                        return value
                    else:
                        value, _ = f(x_full, np.zeros(0), data)
                        return value

                x_opt_init = x_new_full[support_new].copy()
                opt = nlopt.opt(nlopt.LD_LBFGS, k)
                opt.set_min_objective(opt_f)
                x_new = np.zeros(p)
                x_new[support_new] = opt.optimize(x_opt_init)
            except RuntimeError:
                raise
        # terminating condition
        if np.all(set(support_old) == set(support_new)):
            break
        else:
            x_old = x_new.copy()
            support_old = support_new.copy()

    if k > final_support_size:
        support_new = np.argpartition(np.abs(x_new), -final_support_size)[
            -final_support_size:
        ]
        non_supp = np.array([True] * p)
        non_supp[support_new] = False
        x_new[non_supp] = 0.0

    return x_new, iter

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




if __name__ == "__main__":
    np.random.seed(1)
    n = 200
    p = 200
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
    beta, iter = GraHTP(
        quadratic,
        p,
        k,
        step_size=0.001,
        #fast=True,
        #relaxed_sparsity_level=k,
        max_iter=100,
        data={"A": data_set.x.T @ data_set.x, "B": -2 * data_set.x.T @ data_set.y}
    )
    t2 = time.time()
    print(iter)
    print(t2 - t1)
    print(MyTest.accuracy(beta, data_set.coef_))
