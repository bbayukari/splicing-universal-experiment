import numpy as np
import time

from abess import make_glm_data
from abess import pybind_cabess

import MyTest
import GraHTP
from abess_quadratic import abess_quadratic_user_define

pybind_cabess.init_spdlog()


def GraHTP_quadratic_cv(p, k, data):
    step_size_cv = [0.0001, 0.0005, 0.05, 0.1] + [(s + 1) / 1000 for s in range(10)]

    min_estimator = np.zeros(p)
    min_loss = GraHTP.quadratic(min_estimator, np.zeros(0), data)[0]
    min_step_size = 0.0
    fail_times = 0
    t1 = time.time()
    for step_size in step_size_cv:
        try:
            x = GraHTP.GraHTP(GraHTP.quadratic, p, k, step_size=step_size, data=data)[0]
            loss = GraHTP.quadratic(x, np.zeros(0), data)[0]
            if loss < min_loss:
                min_loss = loss
                min_estimator = x
                min_step_size = step_size
        except RuntimeError:
            fail_times += 1
            if fail_times > 4:
                raise
    t2 = time.time()

    return min_estimator, t2 - t1, min_step_size


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
            dataset = {"A": data.x.T @ data.x, "B": -2 * data.x.T @ data.y}

            GraHTP_cv_coef, result["GraHTP_cv_time"] = GraHTP_quadratic_cv(
                p, k, dataset
            )
            abess_coef, result["abess_time"] = abess_quadratic_user_define(
                p, k, dataset
            )
            t1 = time.time()
            GraHTP_coef = GraHTP.GraHTP(GraHTP.quadratic, p, k, data=data)[0]
            t2 = time.time()

            result["GraHTP_time"] = t2 - t1
            result["GraHTP_accuracy"] = MyTest.accuracy(GraHTP_coef, data.coef_)
            result["GraHTP_cv_accuracy"] = MyTest.accuracy(GraHTP_cv_coef, data.coef_)
            result["abess_accuracy"] = MyTest.accuracy(abess_coef, data.coef_)
            break
        except RuntimeError:
            continue
    if iter >= 100:
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
        for method in ["abess", "GraHTP", "GraHTP_cv"]
    ]
    test = MyTest.Test(task, in_keys, out_keys, processes=40, name="GraHTP_linear")

    # test.check(n=20,p=3,k=3)

    para = (
        list(
            MyTest.del_duplicate(
                MyTest.product_dict(
                    n=[i * 50 + 100 for i in range(15)], p=[200], k=[20]
                ),
                MyTest.product_dict(
                    n=[i * 500 + 500 for i in range(10)], p=[2000], k=[100]
                )
            )
        )
        * 10
    )

    # test start
    test.run(para)
    test.save()
