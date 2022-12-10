import statistic_model
import statistic_model_pybind
import variable_select_algorithm
import parallel_experiment_util
import abess

import numpy as np
import time


def task(n, seed):
    result = {}
    # make dataset
    p = 500
    k = 50
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([100, -100], k)
    data = abess.make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.2,
        family="binomial",
        corr_type="exp",
        snr=10 * np.log10(6),
        standardize=True,
        coef_=coef
    )

    # set model
    model = abess.ConvexSparseSolver(model_size=p, support_size=k)
    model.set_loss_custom(
        statistic_model_pybind.logistic_loss_no_intercept,
        statistic_model_pybind.logistic_gradient_no_intercept,
        statistic_model_pybind.logistic_hessian_no_intercept,
    )
    data_set = statistic_model_pybind.RegressionData(data.x, data.y)

    # run model
    t1 = time.time()
    Lasso_coef, Lasso_best_lambda = variable_select_algorithm.Lasso(
        loss_cvxpy=statistic_model.logistic_cvxpy_no_intercept,
        dim=p,
        data=data,
        support_size=k,
        init_lambda=0.1
    )
    t2 = time.time()
    GraHTP_coef = variable_select_algorithm.GraHTP(
        loss_fn=statistic_model.logistic_loss_no_intercept,
        grad_fn=statistic_model.logistic_grad_no_intercept,
        dim=p,
        data=data_set,
        support_size=k,
        step_size=2e-3,
    )
    t3 = time.time()
    GraHTP_cv_coef, best_step_size = variable_select_algorithm.GraHTP_cv(
        loss_fn=statistic_model.logistic_loss_no_intercept,
        grad_fn=statistic_model.logistic_grad_no_intercept,
        dim=p,
        data=data_set,
        support_size=k,
    )
    t4 = time.time()
    GraSP_coef = variable_select_algorithm.GraSP(
        loss_fn=statistic_model.logistic_loss_no_intercept,
        grad_fn=statistic_model.logistic_grad_no_intercept,
        dim=p,
        data=data_set,
        support_size=k,
    )
    t5 = time.time()
    model.fit(data_set)
    SCOPE_coef = model.get_solution()
    t6 = time.time()

    # return
    result["Lasso_accuracy"] = parallel_experiment_util.accuracy(Lasso_coef, data.coef_)
    result["Lasso_best_lambda"] = Lasso_best_lambda
    result["Lasso_time"] = t2 - t1
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
    result["SCOPE_accuracy"] = parallel_experiment_util.accuracy(SCOPE_coef, data.coef_)
    result["SCOPE_time"] = t6 - t5

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
        processes=20,
        name="logistic_experiment",
        memory_limit=80
    )

    if False:
        experiment.check(n=200, seed=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [200]},
            {"n": [250]},
            {"n": [300]},
            {"n": [350]},
            repeat=[15, 80, 25, 80],
            seed=10000
        )

        experiment.run(parameters)
        experiment.save()
