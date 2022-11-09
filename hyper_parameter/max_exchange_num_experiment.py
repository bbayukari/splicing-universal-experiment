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
    dataset = pybind_cabess.Data(data.x, data.y)
    # set model
    model = ConvexSparseSolver(
        model_size=p, sample_size=n, intercept_size=1, support_size=k
    )
    model.set_model_autodiff(
        pybind_cabess.loss_linear,
        pybind_cabess.gradient_linear,
        pybind_cabess.hessian_linear,
    )
    model.set_data(dataset)
    # run model
    for max_exchange_num in [2, 5, 10, 15, 20]:
        model.max_exchange_num = max_exchange_num
        t1 = time.time()
        model.fit()
        t2 = time.time()
        result["accuracy_{}".format(max_exchange_num)] = MyTest.accuracy(
            model.coef_, data.coef_
        )
        result["time_{}".format(max_exchange_num)] = t2 - t1

    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k"]
    out_keys = [
        "accuracy_{}".format(max_exchange_num)
        for max_exchange_num in [2, 5, 10, 15, 20]
    ] + ["time_{}".format(max_exchange_num) for max_exchange_num in [2, 5, 10, 15, 20]]

    test = MyTest.Test(task, in_keys, out_keys, processes=40, name="max_exchange_num")
    # if n is very small, out of samples mse cann't be computed.
    # test.check(n=200,p=1000,k=100)

    para = (
        list(
            MyTest.del_duplicate(
                MyTest.product_dict(
                    n=[i * 40 + 40 for i in range(10)], p=[200], k=[20]
                ),
                MyTest.product_dict(
                    n=[200], p=[i * 40 + 40 for i in range(10)], k=[20]
                ),
            )
        )
        * 10
    )

    # test start
    test.run(para)
    test.save()
