from struct import pack
import numpy as np
import time
import pandas as pd
import os

from abess import pybind_cabess
from abess import make_glm_data
from abess.universal import ConvexSparseSolver
from MyTest import accuracy

pybind_cabess.init_spdlog(console_log_level=6, file_log_level=6)

#info = pd.read_csv(os.path.dirname(__file__) + '/data/info.csv')
#data = pd.read_csv(os.path.dirname(__file__) + '/data/data.csv')

p = 10000
k = 100

results = []
for n in range(500,2501,400):
    print(n)
    data = make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.2,
        family="binomial",
        corr_type="exp",
        snr=10 * np.log10(6),
        standardize=True,
    )
    model = ConvexSparseSolver(model_size=p, sample_size=n, intercept_size=1, support_size=k)  
    model.set_model_autodiff(
        pybind_cabess.loss_logistic,
        pybind_cabess.gradient_logistic,
        pybind_cabess.hessian_logistic,
    )
    t1 = time.time()
    model.fit(pybind_cabess.Data(data.x, data.y))
    t2 = time.time()

    results.append([n,t2-t1,accuracy(model.coef_, data.coef_)])

df = pd.DataFrame(results, columns=['n','time','accuracy'])
df.insert(0,'method','abess')
df.to_csv(os.path.dirname(__file__) + '/results/abess_2.csv')
