from abess import ConvexSparseSolver, make_glm_data
import pandas as pd
import statistic_model_pybind
import statistic_model
import variable_select_algorithm

import numpy as np

results = []

for n in [i*100 + 100 for i in range(20)]:
    p = 500
    k = 50
    seed = 1045 + n
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([100, -100], k)
    data = make_glm_data(
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
    model = ConvexSparseSolver(model_size=p, sample_size=n, support_size=k)
    model.set_model_user_defined(
        statistic_model_pybind.logistic_loss_no_intercept,
        statistic_model_pybind.logistic_gradient_no_intercept,
        statistic_model_pybind.logistic_hessian_no_intercept,
    )
    model.set_data(statistic_model_pybind.RegressionData(data.x, data.y))

    # run model
    model.fit(console_log_level="off", file_log_level="debug")
    GraHTP_coef, GraHTP_time, GraHTP_grad, GraHTP_optim, GraHTP_iter = variable_select_algorithm.GraHTP(
        loss_fn=statistic_model.logistic_loss_no_intercept,
        grad_fn=statistic_model.logistic_grad_no_intercept,
        dim=p,
        data=statistic_model_pybind.RegressionData(data.x, data.y),
        support_size=k,
        step_size=2e-3,
    )
    GraSP_coef, GraSP_time, GraSP_grad, GraSP_optim, GraSP_iter = variable_select_algorithm.GraSP(
        loss_fn=statistic_model.logistic_loss_no_intercept,
        grad_fn=statistic_model.logistic_grad_no_intercept,
        dim=p,
        data=statistic_model_pybind.RegressionData(data.x, data.y),
        support_size=k,
    )

    results.append({
        "GraHTP_iter": GraHTP_iter,
        "GraHTP_time": GraHTP_time,
        "GraHTP_grad": GraHTP_grad / GraHTP_time,
        "GraHTP_optim": GraHTP_optim / GraHTP_time,
        "GraSP_iter": GraSP_iter,
        "GraSP_time": GraSP_time,
        "GraSP_grad": GraSP_grad / GraSP_time,
        "GraSP_optim": GraSP_optim / GraSP_time,
    })

key = [
    "GraHTP_iter", "GraHTP_time", "GraHTP_grad", "GraHTP_optim", 
    "GraSP_iter", "GraSP_time", "GraSP_grad", "GraSP_optim"
]

pd.DataFrame(
    {
        para: [result[para] for result in results]
        for para in key
    }
).to_csv("logs/algo_time.csv")
