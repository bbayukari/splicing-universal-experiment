import statistic_model
import statistic_model_pybind
import variable_select_algorithm
import parallel_experiment_util
import abess

import numpy as np
import time

abess.set_log_level(console_log_level=6, file_log_level=6)


def task(n, seed):
    result = {}
    # make dataset
    data, theta, coef = statistic_model.ising_generator(P=20, N=n, Edges=40, seed=seed)
    dim = 190
    support_size = 40
    # set model
    model = abess.ConvexSparseSolver(model_size=dim, sample_size=n, support_size=support_size)
    model.set_model_autodiff(
        statistic_model_pybind.ising_model
    )
    model.set_data(statistic_model_pybind.IsingData(data))

    # run model
    t1 = time.time()
    model.fit()
    SCOPE_coef = model.coef_
    t2 = time.time()
    GraHTP_coef = variable_select_algorithm.GraHTP(
        loss_fn=statistic_model.ising_loss_no_intercept,
        dim=dim,
        data=statistic_model.IsingData(data),
        support_size=support_size,
    )
    t3 = time.time()
    GraHTP_cv_coef, best_step_size = variable_select_algorithm.GraHTP_cv(
        loss_fn=statistic_model.ising_loss_no_intercept,
        dim=dim,
        data=statistic_model.IsingData(data),
        support_size=support_size,
    )
    t4 = time.time()
    GraSP_coef = variable_select_algorithm.GraSP(
        loss_fn=statistic_model.ising_loss_no_intercept,
        dim=dim,
        data=statistic_model.IsingData(data),
        support_size=support_size,
    )
    t5 = time.time()

    # return
    result["SCOPE_accuracy"] = parallel_experiment_util.accuracy(SCOPE_coef, data.coef_)
    result["SCOPE_time"] = t2 - t1
    result["GraHTP_accuracy"] = parallel_experiment_util.accuracy(
        GraHTP_coef, data.coef_
    )
    result["GraHTP_time"] = t3 - t2
    result["GraHTP_cv_accuracy"] = parallel_experiment_util.accuracy(
        GraHTP_cv_coef, data.coef_
    )
    result["GraHTP_best_step_size"] = best_step_size
    result["GraHTP_cv_time"] = t4 - t3
    result["GraSP_accuracy"] = parallel_experiment_util.accuracy(GraSP_coef, data.coef_)
    result["GraSP_time"] = t5 - t4

    return result


if __name__ == "__main__":
    in_keys = ["n", "seed"]
    out_keys = [
        "SCOPE_accuracy",
        "SCOPE_time",
        "GraHTP_accuracy",
        "GraHTP_time",
        "GraHTP_cv_accuracy",
        "GraHTP_best_step_size",
        "GraHTP_cv_time",
        "GraSP_accuracy",
        "GraSP_time",
    ]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=40,
        name="logistic_experiment",
    )

    if False:
        experiment.check(n=[i*100 +100 for i in range(20)][0], seed=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [i*100 +100 for i in range(20)]},
            repeat=2,
            seed=100
        )

        experiment.run(parameters)
        experiment.save()
