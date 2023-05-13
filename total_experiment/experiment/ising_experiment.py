import statistic_model
import statistic_model_pybind
import variable_select_algorithm
import parallel_experiment_util
import abess
import scope
import numpy as np
import time


def task(n, k, seed):
    result = {}
    # make dataset
    data, theta, coef = statistic_model.ising_generator(P=20, N=n, Edges=k, seed=seed)
    dataset = statistic_model_pybind.IsingData(data)
    dim = 190
    support_size = k
    print(coef)
    """
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
    """
    t5 = time.time()

    SCOPE_coef = scope.ScopeSolver(
        dim,
        support_size,
        n,
        file_log_level="debug"
    ).solve(
        statistic_model_pybind.ising_loss,
        data=dataset,
        gradient=statistic_model_pybind.ising_grad,
        hessian=statistic_model_pybind.ising_hess_diag,
        cpp=True,
    )
    t6 = time.time()

    """
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
    """
    result["SCOPE_accuracy"] = parallel_experiment_util.accuracy(SCOPE_coef, coef)
    result["SCOPE_time"] = t6 - t5
    return result


if __name__ == "__main__":
    in_keys = ["n", "k", "seed"]
    out_keys = [
        "SCOPE_accuracy",
        "SCOPE_time",]
    """
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
    """
    

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=8,
        name="ising_supplement-3",
        memory_limit=0.8,
    )

    if False:
        experiment.check(n=700, k=40, seed=708)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [i * 100 + 100 for i in range(10)], "k": [40]},
            repeat=10,
            seed=1234,
        )

        experiment.run(parameters)
        experiment.save()
