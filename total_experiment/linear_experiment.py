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
    np.random.seed(seed)
    data = abess.make_glm_data(
        n=n,
        p=500,
        k=50,
        rho=0.2,
        family="gaussian",
        corr_type="exp",
        snr=10 * np.log10(6),
        standardize=True,
    )

    # set model
    model = abess.ConvexSparseSolver(model_size=500, sample_size=n, support_size=50)
    model.set_model_user_defined(
        statistic_model_pybind.linear_loss_no_intercept,
        statistic_model_pybind.linear_gradient_no_intercept,
        statistic_model_pybind.linear_hessian_no_intercept,
    )
    model.set_data(statistic_model_pybind.RegressionData(data.x, data.y))

    # run model
    t1 = time.time()
    model.fit()
    SCOPE_coef = model.coef_
    t2 = time.time()
    GraHTP_coef = variable_select_algorithm.GraHTP(
        loss_fn=statistic_model.linear_loss_no_intercept,
        grad_fn=statistic_model.linear_grad_no_intercept,
        dim=500,
        data=data,
        support_size=50,
    )
    t3 = time.time()
    GraHTP_cv_coef, best_step_size = variable_select_algorithm.GraHTP_cv(
        loss_fn=statistic_model.linear_loss_no_intercept,
        grad_fn=statistic_model.linear_grad_no_intercept,
        dim=500,
        data=data,
        support_size=50,
    )
    t4 = time.time()
    GraSP_coef = variable_select_algorithm.GraSP(
        loss_fn=statistic_model.linear_loss_no_intercept,
        grad_fn=statistic_model.linear_grad_no_intercept,
        dim=500,
        data=data,
        support_size=50,
    )
    t5 = time.time()
    Lasso_coef, Lasso_best_lambda = variable_select_algorithm.Lasso(
        loss_cvxpy=statistic_model.linear_cvxpy_no_intercept,
        dim=500,
        data=data,
        support_size=50,
    )
    t6 = time.time()

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
    result["Lasso_accuracy"] = parallel_experiment_util.accuracy(Lasso_coef, data.coef_)
    result["Lasso_best_lambda"] = Lasso_best_lambda
    result["Lasso_time"] = t6 - t5

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
        "Lasso_accuracy",
        "Lasso_best_lambda",
        "Lasso_time",
    ]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=40,
        name="linear_experiment",
    )

    #experiment.check(n=1000, seed=1)

    parameters = parallel_experiment_util.para_generator(
        {"n": np.arange(20)*100+100},
        repeat=2
    )

    experiment.run(parameters)
    experiment.save()
