import numpy as np
import time

import model
from abess import pybind_cabess
import sys
sys.path.append("/data/home/wangzz/github/splicing-universal-experiment/")
import MyTest
from MyTest import merge_dict
from MyTest import product_dict

pybind_cabess.init_spdlog(console_log_level=6, file_log_level=6)


def task(n ,p, k, seed):
    result = {'true_seed': n+seed}
    # make dataset
    data, theta, coef = model.data_generate(p, n, k, result['true_seed'])
    # run model
    t1 = time.time()
    coef_abess = model.abess(data, p, k)
    t2 = time.time()
    coef_wainwright_min = model.wainwright(data, k, method='min')
    t3 = time.time()
    coef_wainwright_max = model.wainwright(data, k, method='max')
    t4 = time.time()

    
    # return
    result["abess_accuracy"] = MyTest.accuracy(coef_abess, coef)
    result["abess_time"] = t2 - t1
    result["wainwright_min_accuracy"] = MyTest.accuracy(coef_wainwright_min, coef)
    result["wainwright_min_time"] = t3 - t2
    result["wainwright_max_accuracy"] = MyTest.accuracy(coef_wainwright_max, coef)
    result["wainwright_max_time"] = t4 - t3
    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k", "seed"]
    out_keys = [
        "abess_accuracy", "abess_time", "wainwright_min_accuracy", "wainwright_min_time", "wainwright_max_accuracy", "wainwright_max_time"
    ] 
    test = MyTest.Test(
        task, in_keys, out_keys, processes=40, name="ising_model"
    )
    # if n is very small, out of samples mse cann't be computed.
    #test.check(n=20,p=10,k=10,seed=1)

    para = list(MyTest.del_duplicate(
            product_dict(n=np.arange(20)*20+20, p=[20], k=[40], seed=np.arange(10))),
        ) 

    #test start
    test.run(para)
    test.save()
