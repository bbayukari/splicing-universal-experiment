import numpy as np
import time

from abess import make_glm_data
from abess import pybind_cabess

import MyTest
import model
from GraHTP import GraHTP, GraHTP_cv
from abess_user_define import abess_quadratic

#pybind_cabess.init_spdlog(console_log_level=6, file_log_level=2)

def task(n, p, k):
    result = {}
    # repeat until no exception
    for iter in range(100):
        try:
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
            dataset = {"Q": data.x.T @ data.x, "v": -2 * data.x.T @ data.y}

            t0 = time.time()
            (
                GraHTP_cv_coef,
                result["min_step_size"],
            ) = GraHTP_cv(model.loss_quadratic, model.grad_quadratic, p, k, dataset)
            t1 = time.time()
            GraHTP_1_coef = GraHTP(
                model.loss_quadratic,
                model.grad_quadratic,
                p,
                k,
                step_size=0.001,
                data=dataset,
            )
            t2 = time.time()
            GraHTP_5_coef = GraHTP(
                model.loss_quadratic,
                model.grad_quadratic,
                p,
                k,
                step_size=0.005,
                data=dataset,
            )
            t3 = time.time()
            abess_coef, result["abess_time"] = abess_quadratic(
                p, k, dataset
            )

            result["GraHTP_cv_time"] = t1 - t0
            result["GraHTP_1_time"] = t2 - t1
            result["GraHTP_1_accuracy"] = MyTest.accuracy(GraHTP_1_coef, data.coef_)
            result["GraHTP_5_time"] = t3 - t2
            result["GraHTP_5_accuracy"] = MyTest.accuracy(GraHTP_5_coef, data.coef_)
            result["GraHTP_cv_accuracy"] = MyTest.accuracy(GraHTP_cv_coef, data.coef_)
            result["abess_accuracy"] = MyTest.accuracy(abess_coef, data.coef_)
            break
        except RuntimeError:
            continue
    if iter >= 10:
        # fill result with NaN if no solution is found
        result = {
            key: np.nan
            for key in [
                method + "_" + term
                for term in ["accuracy", "time"]
                for method in ["abess", "GraHTP", "GraHTP_cv"]
            ]
        }
    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k"]
    out_keys = [
        method + "_" + term
        for term in ["accuracy", "time"]
        for method in ["abess", "GraHTP_1", "GraHTP_5", "GraHTP_cv"]
    ] + ["min_step_size"]
    test = MyTest.Test(task, in_keys, out_keys, processes=40, name="GraHTP_linear_2")

    # test.check(n=20,p=3,k=3)

    para = (
        list(
            MyTest.del_duplicate(
                MyTest.product_dict(
                    n=[i * 100 + 100 for i in range(10)], p=[500], k=[50]
                )
                # , MyTest.product_dict(n=[i * 500 + 500 for i in range(10)], p=[2000], k=[100])
            )
        )
        * 10
    )

    # test start
    test.run(para)
    test.save()
