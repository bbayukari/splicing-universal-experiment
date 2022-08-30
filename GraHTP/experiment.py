import numpy as np
import time

from abess import make_glm_data
from abess import pybind_cabess

import MyTest
from GraHTP import GraHTP_quadratic
from abess_quadratic import abess_quadratic_user_define

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
    dataset = {"A": data.x.T @ data.x, "B": -2 * data.x.T @ data.y}
    t1 = time.time()
    GraHTP_coef = GraHTP_quadratic(p,k,dataset)
    t2 = time.time()
    abess_coef = abess_quadratic_user_define(p,k,dataset)
    t3 = time.time()

    result["GraHTP_accuracy"] = MyTest.accuracy(GraHTP_coef, data.coef_)
    result["abess_accuracy"] = MyTest.accuracy(abess_coef, data.coef_)
    result["GraHTP_time"] = t2 - t1
    result["abess_time"] = t3 - t2

    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k"]
    out_keys = [
        method + "_" + term
        for term in ["accuracy", "time"]
        #for method in ["autodiff"]
        for method in ["abess", "GraHTP"]
    ]
    test = MyTest.Test(
        task, in_keys, out_keys, processes=40, name="GraHTP_linear"
    )
    
    #test.check(n=20,p=3,k=3)

    para = list(MyTest.del_duplicate(
        MyTest.product_dict(n=[100], p=np.arange(10,1000,10), k=[10]),
    )) * 5
  
    #test start
    test.run(para)
    test.save()
  
