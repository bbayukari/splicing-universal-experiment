import cvxpy as cp
import numpy as np
import nlopt

def GraHTP(
    loss,
    grad,
    dim,
    support_size,
    data=None,
    fast=False,
    final_support_size=-1,
    x_init=None,
    step_size=0.01,
    max_iter=100,
):
    """GraHTP algorithm
    Args:
        loss: function (x, data, active_index) -> loss_value
        grad: function (x, data, active_index, compute_index) -> gradient_vector
            x: array with the shape of (dim,)
            data: dictionary, data for loss and grad
            active_index: int array, the index of nonzore features, default is None which means it's np.arange(dim)
            compute_index: int array, the index of features which gradient is computed
            gradient_vector.shape = compute_index.shape, default is None which means it's same with active_index
        dim: int, dimension of the model
        support_size: the number of selected features for algorithm
        data: dictionary, data for loss and grad
        fast: bool, whether to use FGraHTP algorithm or not
        final_support_size: int, must less than support_size, algorithm will select support_size features firstly,
            then preserve the top final_support_size entries of the output as the final estimation.
        x_init: the initial value of the estimator
    Returns:
        estimator: array with the shape of (dim,) which contains k nonzero entries
    """
    if x_init is None:
        x_init = np.zeros(dim)
    
    if final_support_size < 0:
        final_support_size = support_size
    
    # init
    x_old = x_init
    support_old = np.argpartition(np.abs(x_old), -support_size)[-support_size:]

    for iter in range(max_iter):
        # S1: gradient descent
        x_bias = x_old - step_size * grad(x_old, data)
        # S2: Gradient Hard Thresholding
        support_new = np.argpartition(np.abs(x_bias), -support_size)[-support_size:]
        # S3: debise
        if fast:
            x_new = np.zeros(dim)
            x_new[support_new] = x_bias[support_new]
        else:
            try:
                def object_fn(x):
                    x_full = np.zeros(dim)
                    x_full[support_new] = x
                """
                def opt_f(x, gradient):
                    x_full = np.zeros(dim)
                    x_full[support_new] = x
                    if gradient.size > 0:
                        gradient[:] = grad(x_full, data, support_new)
                    return loss(x_full, data, support_new)    

                opt = nlopt.opt(nlopt.LD_SLSQP, support_size)
                opt.set_min_objective(opt_f)
                opt.set_ftol_rel(0.001)
                x_new = np.zeros(dim)
                x_new[support_new] = opt.optimize(x_bias[support_new])
                """
            except RuntimeError:
                raise
        # terminating condition
        if np.all(set(support_old) == set(support_new)):
            break
        x_old = x_new
        support_old = support_new

    final_support = np.argpartition(np.abs(x_new), -final_support_size)[-final_support_size:]
    final_estimator = np.zeros(dim)
    final_estimator[final_support] = x_new[final_support]

    return final_estimator

def GraHTP_cv(loss_f, grad_f, p, k, data):
    step_size_cv = [0.0001, 0.0005, 0.05, 0.1] + [(s + 1) / 1000 for s in range(10)]

    min_estimator = np.zeros(p)
    min_loss = loss_f(min_estimator, data)
    min_step_size = 0.0
    fail_times = 0
    for step_size in step_size_cv:
        try:
            x = GraHTP(loss_f, grad_f, p, k, step_size=step_size, data=data)
            loss = loss_f(x, data)
            if loss < min_loss:
                min_loss = loss
                min_estimator = x
                min_step_size = step_size
        except RuntimeError:
            fail_times += 1
            if fail_times > 4:
                raise

    return min_estimator, min_step_size

if __name__ == "__main__":
    import model
    from abess import make_glm_data
    import MyTest
    import time

    np.random.seed(12)
    n = 10
    p = 5
    k = 3
    
    quadratic = False
    logistic = True

    if quadratic:
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
        beta = GraHTP(
            model.loss_quadratic,
            model.grad_quadratic,
            p,
            k,
            #step_size=0.001,
            #fast=True,
            #relaxed_sparsity_level=k,
            #max_iter=1000,
            data={"Q": data_set.x.T @ data_set.x, "v": -2 * data_set.x.T @ data_set.y}
        )
        t2 = time.time()

        print(t2 - t1)
        print(MyTest.accuracy(beta, data_set.coef_))
    
    if logistic:
        data_set = make_glm_data(
            n=n,
            p=p,
            k=k,
            rho=0.2,
            family="binomial",
            corr_type="exp",
            snr=10 * np.log10(6),
            standardize=True,
        )

        t1 = time.time()
        beta = GraHTP(
            model.loss_logistic,
            model.grad_logistic,
            p,
            k,
            step_size=0.001,
            #fast=True,
            #relaxed_sparsity_level=k,
            #max_iter=1000,
            data=data_set
        )
        t2 = time.time()

        print(t2 - t1)
        print(MyTest.accuracy(beta, data_set.coef_))
