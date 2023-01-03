import numpy as np
import time
import statistic_model
import statistic_model_pybind
import parallel_experiment_util

from abess import make_glm_data, ConvexSparseSolver


def task(n, seed, model):
    result = {}
    # make dataset
    if model == "Ising":
        p = 190
        k = 20
        data, theta, coef = statistic_model.ising_generator(P=20, N=n, Edges=k, seed=seed)
        data_set = statistic_model_pybind.IsingData(data)
    else:
        p = 500
        k = 50
        coef = np.zeros(p)
        np.random.seed(seed)
        coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([100, -100], k)
        data = make_glm_data(
            n=n,
            p=p,
            k=k,
            rho=0.2,
            family="gaussian" if model == "Linear" else "binomial",
            corr_type="exp",
            snr=10 * np.log10(6),
            standardize=True,
            coef_=coef
        )
        data_set = statistic_model_pybind.RegressionData(data.x, data.y)

    # set model
    solver = ConvexSparseSolver(
        model_size=p, support_size=k
    )

    if model == "Ising":
        solver.set_loss_custom(
            loss = statistic_model_pybind.ising_loss,
            gradient = statistic_model_pybind.ising_grad,
            hessian = statistic_model_pybind.ising_hess_diag
        )
    elif model == "Classification":
        solver.set_loss_custom(
            statistic_model_pybind.logistic_loss_no_intercept,
            statistic_model_pybind.logistic_gradient_no_intercept,
            statistic_model_pybind.logistic_hessian_no_intercept,
        )
    elif model == "Linear":
        solver.set_loss_custom(
            statistic_model_pybind.linear_loss_no_intercept,
            statistic_model_pybind.linear_gradient_no_intercept,
            statistic_model_pybind.linear_hessian_no_intercept,
        )
    else:
        raise ValueError("model must be one of Ising, Classification, Linear")    

    # run model
    t1 = time.time()
    solver.fit(data_set)
    sacrifice_coef = solver.coef_
    t2 = time.time()
    solver.init_active_set = np.arange(p)
    solver.fit(data_set)
    random_coef = solver.coef_
    t3 = time.time()

    # return
    result["random_accuracy"] = parallel_experiment_util.accuracy(random_coef, coef)
    result["random_time"] = t3 - t2
    result["sacrifice_accuracy"] = parallel_experiment_util.accuracy(sacrifice_coef, coef)
    result["sacrifice_time"] = t2 - t1
    return result


if __name__ == "__main__":
    in_keys = ["n", "seed", "model"]
    out_keys = [
        "random_accuracy", "random_time", "sacrifice_accuracy", "sacrifice_time"
    ]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=40,
        name="init_strategy_2",
        memory_limit=80
    )

    if False:
        experiment.check(n=1000, model="Linear", seed=1)
        experiment.check(n=500, model="Classification", seed=10)
        experiment.check(n=100, model="Ising", seed=100)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [i * 200 + 200 for i in range(10)], "model": ["Linear", "Classification", "Ising"]},
            repeat=100,
            seed=1,
        )
        
        experiment.run(parameters)
        experiment.save()
