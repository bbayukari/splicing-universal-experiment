import numpy as np
import time

from abess import make_glm_data
from abess import pybind_cabess
from abess.universal import ConvexSparseSolver
import sys
sys.path.append("/data/home/wangzz/github/splicing-universal-experiment/")
import MyTest
from MyTest import merge_dict
from MyTest import product_dict

pybind_cabess.init_spdlog(console_log_level=6, file_log_level=6)


def task(n ,p, k):
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
    dataset = pybind_cabess.Data(data.x, data.y)
    # set model
    model = ConvexSparseSolver(model_size=p, sample_size=n, intercept_size=1, support_size=k)
    model.set_model_autodiff(
        pybind_cabess.loss_linear,
        pybind_cabess.gradient_linear,
        pybind_cabess.hessian_linear,
    )
    model.set_data(dataset)
    # run model
    t1 = time.time()
    model.fit()
    margin_coef = model.coef_
    t2 = time.time()
    model.init_active_set = np.arange(p)
    model.fit()
    random_coef = model.coef_
    t3 = time.time()
    model.init_active_set = [0]
    model.fit()
    sacrifice_coef = model.coef_
    t4 = time.time()

    # return
    result["margin_accuracy"] = MyTest.accuracy(margin_coef, data.coef_)
    result["margin_time"] = t2 - t1
    result["random_accuracy"] = MyTest.accuracy(random_coef, data.coef_)
    result["random_time"] = t3 - t2
    result["sacrifice_accuracy"] = MyTest.accuracy(sacrifice_coef, data.coef_)
    result["sacrifice_time"] = t4 - t3
    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k"]
    out_keys = [
        "margin_accuracy", "margin_time", "random_accuracy", "random_time", "sacrifice_accuracy", "sacrifice_time"
    ] 
    test = MyTest.Test(
        task, in_keys, out_keys, processes=40, name="init_strategy"
    )
    # if n is very small, out of samples mse cann't be computed.
    #test.check(n=200,p=1000,k=100)

    para = (
        list(
            MyTest.del_duplicate(
                MyTest.product_dict(
                    n=[i * 100 + 100 for i in range(10)], p=[500], k=[50]
                ),
                MyTest.product_dict(n=[500], p=[i * 100 + 100 for i in range(10)], k=[50])
            )
        )
        * 10
    )

    #test start
    test.run(para)
    test.save()
