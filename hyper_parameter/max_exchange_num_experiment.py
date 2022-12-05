import numpy as np
import time
import statistic_model
import statistic_model_pybind
import parallel_experiment_util

from abess import make_glm_data, ConvexSparseSolver

max_exchange_num_list = [2, 5, 10, 20, 30, 40, 50]


def task(n, seed, model):
    result = {}
    # make dataset
    if model == "ising":
        p = 190
        k = 40
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
            family=model,
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

    if model == "ising":
        solver.set_loss_custom(
            loss = statistic_model_pybind.ising_loss,
            gradient = statistic_model_pybind.ising_grad,
            hessian = statistic_model_pybind.ising_hess_diag
        )
    elif model == "binomial":
        solver.set_loss_custom(
            statistic_model_pybind.logistic_loss_no_intercept,
            statistic_model_pybind.logistic_gradient_no_intercept,
            statistic_model_pybind.logistic_hessian_no_intercept,
        )
    elif model == "gaussian":
        solver.set_loss_custom(
            statistic_model_pybind.linear_loss_no_intercept,
            statistic_model_pybind.linear_gradient_no_intercept,
            statistic_model_pybind.linear_hessian_no_intercept,
        )
    else:
        raise ValueError("model must be one of ising, binomial, gaussian")    

    # run model
    for max_exchange_num in max_exchange_num_list:
        solver.max_exchange_num = max_exchange_num
        t1 = time.time()
        solver.fit(data_set)
        t2 = time.time()
        result["accuracy_{}".format(max_exchange_num)] = parallel_experiment_util.accuracy(
            solver.get_solution(), coef
        )
        result["time_{}".format(max_exchange_num)] = t2 - t1

    return result


if __name__ == "__main__":
    in_keys = ["n", "seed", "model"]
    out_keys = [
        "accuracy_{}".format(max_exchange_num)
        for max_exchange_num in max_exchange_num_list
    ] + ["time_{}".format(max_exchange_num) for max_exchange_num in max_exchange_num_list]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=20,
        name="max_exchange_num",
        memory_limit=40
    )

    if False:
        experiment.check(n=1000, model="gaussian", seed=1)
        experiment.check(n=500, model="binomial", seed=10)
        experiment.check(n=100, model="ising", seed=100)
    else:
        parameters = parallel_experiment_util.para_generator(
            {"n": [i * 100 + 100 for i in range(10)], "model": ["gaussian", "binomial", "ising"]},
            repeat=100,
            seed=1,
        )
        
        experiment.run(parameters)
        experiment.save()
