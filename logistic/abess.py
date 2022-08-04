import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

from abess import make_glm_data
from abess import pybind_cabess

import MyTest
from MyTest import merge_dict
from MyTest import product_dict

pybind_cabess.init_spdlog(console_log_level=6, file_log_level=6)


def task(n, p, k):
    result = {}
    # make dataset
    data = make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.2,
        family="gaussian",
        corr_type="exp",
        snr=10 * np.log10(6),
        standardize=True,
    )
    X_train = data.x
    y_train = data.y

    t2 = time.time()
    autodiff_intercept, autodiff_coef = autodiff_linear(X_train, y_train, k)
    t3 = time.time()
    # return
    #result["gurobi_accuracy"] = MyTest.accuracy(gurobi_coef, data.coef_)
    result["autodiff_accuracy"] = MyTest.accuracy(autodiff_coef, data.coef_)

    #result["gurobi_time"] = t2 - t1
    result["autodiff_time"] = t3 - t2

    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k"]
    out_keys = [
        method + "_" + term
        for term in ["accuracy", "time"]
        for method in ["autodiff"]
        #for method in ["autodiff", "gurobi"]
    ]
    test = MyTest.Test(
        task, in_keys, out_keys, processes=40, name="tem"
    )
    # if n is very small, out of samples mse cann't be computed.
    #test.check(n=20,p=3,k=3)

    para = list(MyTest.del_duplicate(
        product_dict(n=[100], p=np.arange(10,1000,10), k=[10]),
    )) * 5
  
    #test start
    test.run(para)
    test.save()
    
