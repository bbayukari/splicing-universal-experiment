import statistic_model
import statistic_model_pybind
import variable_select_algorithm
import parallel_experiment_util
import abess

import numpy as np
import time

abess.set_log_level(console_log_level=6, file_log_level=6)


def task(n, k, seed):
    result = {}
    # make dataset
    data, theta, coef = statistic_model.ising_generator(P=20, N=n, Edges=k, seed=seed)
    dataset = statistic_model_pybind.IsingData(data)
    dim = 190
    support_size = k
    # set model
    model = abess.ConvexSparseSolver(
        model_size=dim, sample_size=n, support_size=support_size
    )
    model.set_model_user_defined(
        loss = statistic_model_pybind.ising_loss,
        gradient = statistic_model_pybind.ising_grad,
        hessian = statistic_model_pybind.ising_hess_diag
    )
    model.set_data(dataset)

    # run model
    t1 = time.time()
    Lasso_coef, Lasso_best_lambda = variable_select_algorithm.Lasso(
        loss_cvxpy=statistic_model.ising_cvxpy,
        dim=dim,
        data=statistic_model.IsingData(data),
        support_size=support_size,
        tol=0.05,
        init_lambda=200
    )
    t2 = time.time()
    GraHTP_coef = variable_select_algorithm.GraHTP(
        loss_fn=statistic_model.ising_loss,
        grad_fn=statistic_model.ising_grad,
        dim=dim,
        data=dataset,
        support_size=support_size,
        step_size=2e-3,
    )
    t3 = time.time()
    GraHTP_cv_coef, best_step_size = variable_select_algorithm.GraHTP_cv(
        loss_fn=statistic_model.ising_loss,
        grad_fn=statistic_model.ising_grad,
        dim=dim,
        data=dataset,
        support_size=support_size,
    )
    t4 = time.time()
    GraSP_coef = variable_select_algorithm.GraSP(
        loss_fn=statistic_model.ising_loss,
        grad_fn=statistic_model.ising_grad,
        dim=dim,
        data=dataset,
        support_size=support_size,
    )
    t5 = time.time()
    model.fit()
    SCOPE_coef = model.coef_
    t6 = time.time()


    # return
    result["Lasso_accuracy"] = parallel_experiment_util.accuracy(Lasso_coef, coef)
    result["Lasso_best_lambda"] = Lasso_best_lambda
    result["Lasso_time"] = t2 - t1
    result["GraHTP_accuracy"] = parallel_experiment_util.accuracy(GraHTP_coef, coef)
    result["GraHTP_time"] = t3 - t2
    result["GraHTP_cv_accuracy"] = parallel_experiment_util.accuracy(
        GraHTP_cv_coef, coef
    )
    result["GraHTP_best_step_size"] = best_step_size
    result["GraHTP_cv_time"] = t4 - t3
    result["GraSP_accuracy"] = parallel_experiment_util.accuracy(GraSP_coef, coef)
    result["GraSP_time"] = t5 - t4
    result["SCOPE_accuracy"] = parallel_experiment_util.accuracy(SCOPE_coef, coef)
    result["SCOPE_time"] = t6 - t5
    return result


if __name__ == "__main__":
    in_keys = ["n", "k", "seed"]
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
        processes=3,
        name="ising_supplement",
        memory_limit=80
    )

    if False:
        experiment.check(n=1000, k=40, seed=1)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [800], "k": [40]},
            repeat=10,
            seed=31000,
        )
        
        experiment.run(parameters)
        experiment.save()
