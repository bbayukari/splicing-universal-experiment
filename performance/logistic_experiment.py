from abess import ConvexSparseSolver, make_glm_data
import statistic_model_pybind
import parallel_experiment_util

import numpy as np
import time


n = 100
p = 50
k = 5
seed = 10
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
t1 = time.time()
model.fit(console_log_level="off", file_log_level="debug")
t2 = time.time()

print("Time: ", t2 - t1)
print("Accurary: ", parallel_experiment_util.accuracy(model.coef_, data.coef_))
