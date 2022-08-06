import numpy as np
import time
import pandas as pd
import os

from abess import pybind_cabess
from abess.universal import ConvexSparseSolver
from abess.linear import LogisticRegression
from MyTest import accuracy

pybind_cabess.init_spdlog(console_log_level=6, file_log_level=6)

info = pd.read_csv(os.path.dirname(__file__) + '/data/info.csv')
data = pd.read_csv(os.path.dirname(__file__) + '/data/data.csv')

p = 500
k = 20

results_autodiff = []
results = []
for id in info['id']:
    x = data.loc[data.id == id, ["X" + str(i + 1) for i in range(p)]].to_numpy()
    y = data.loc[data.id == id, 'y'].to_numpy().ravel()
    coef = info.loc[info.id == id, ["B" + str(i + 1) for i in range(p)]].to_numpy().ravel()

    model = ConvexSparseSolver(model_size=p, sample_size=len(y), intercept_size=1, support_size=k)  
    model.set_model_autodiff(
        pybind_cabess.loss_logistic,
        pybind_cabess.gradient_logistic,
        pybind_cabess.hessian_logistic,
    )
    t1 = time.time()
    model.fit(pybind_cabess.Data(x, y))
    t2 = time.time()

    results_autodiff.append([len(y),t2-t1,accuracy(model.coef_, coef)])

    model2 = LogisticRegression(support_size = k)
    t1 = time.time()
    model2.fit(x, y)
    t2 = time.time()
    results.append([len(y),t2-t1,accuracy(model2.coef_, coef)])
    print(accuracy(model2.coef_, model.coef_))

df = pd.DataFrame(results_autodiff, columns=['n','time','accuracy'])
df.insert(0,'method','autodiff')
df2 = pd.DataFrame(results, columns=['n','time','accuracy'])
df2.insert(0,'method','abess')
pd.concat([df, df2]).to_csv(os.path.dirname(__file__) + '/results/abess_2.csv')
