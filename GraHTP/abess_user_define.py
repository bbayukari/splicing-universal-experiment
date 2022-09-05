from abess.universal import ConvexSparseSolver
import numpy as np
import model as M

def abess_quadratic(model_size, support_size, data):
    model = ConvexSparseSolver(model_size, support_size=support_size)
    model.set_data(None)
    model.set_loss(lambda x, intercept, _data: M.loss_quadratic(x,data))
    model.set_gradient(lambda x, intercept, _data, supp: M.grad_quadratic(x, data, compute_index = supp))
    model.set_hessian(lambda x, intercept, _data, supp: M.hessian_quadratic(x, data, supp))
    model.fit()

    return model.coef_

def abess_logistic(model_size, support_size, data):
    model = ConvexSparseSolver(model_size, support_size=support_size)
    model.set_data(None)
    model.set_loss(lambda x, intercept, _data: M.loss_logistic(x,data))
    model.set_gradient(lambda x, intercept, _data, supp: M.grad_logistic(x, data, compute_index = supp))
    model.set_hessian(lambda x, intercept, _data, supp: M.hessian_logistic(x, data, compute_index = supp))
    model.fit()

    return model.coef_


if __name__ == "__main__":
    import MyTest
    import time
    from abess import make_glm_data 

    np.random.seed(234)
    n = 1000
    p = 500
    k = 300
    
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
        beta = abess_quadratic(p, k, data = {'Q' : data_set.x.T @ data_set.x, 'v' : -2 * data_set.x.T @ data_set.y})
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
        beta = abess_logistic(p, k, data = data_set)
        t2 = time.time()
        print(t2 - t1)
        print(MyTest.accuracy(beta, data_set.coef_))
