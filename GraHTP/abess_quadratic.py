import time
from abess.universal import ConvexSparseSolver
import numpy as np
import MyTest


def abess_quadratic_user_define(model_size, support_size, data):
    """
    Args:
        f: loss
        g: gradient
        h: hessian
    """
    def quadratic(x, intercept, data):
        return x.T @ data['A'] @ x + x.T @ data['B']

    def quadratic_grad(x, intercept, data, supp):
        grad = 2 * data['A'] @ x + data['B']
        return grad[supp]

    def quadratic_hessian(x, intercept, data, supp):
        return 2 * data['A'][np.ix_(supp,supp)]

    model = ConvexSparseSolver(model_size, support_size=support_size)
    model.set_data(data)
    model.set_loss(quadratic)
    model.set_gradient(quadratic_grad)
    model.set_hessian(quadratic_hessian)
    t1 = time.time()
    model.fit()
    t2 = time.time()
    return model.coef_, t2-t1

if __name__ == "__main__":
    np.random.seed(234)
    n = 5000
    p = 500
    k = 50
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

    beta = abess_quadratic_user_define(p, k, data = {'A' : data_set.x.T @ data_set.x, 'B' : -2 * data_set.x.T @ data_set.y})
    t2 = time.time()
    print(t2 - t1)
    print(MyTest.accuracy(beta, data_set.coef_))
