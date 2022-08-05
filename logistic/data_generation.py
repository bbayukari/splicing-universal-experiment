import numpy as np
from abess import make_glm_data
import pandas as pd

df1 = []
df2 = []
n_list = [n*100+100 for n in range(10)]

for n in n_list:
    for _ in range(5):
        p = 500
        data = make_glm_data(
            n=n,
            p=p,
            k=10,
            rho=0.2,
            family="binomial",
            corr_type="exp",
            snr=10 * np.log10(6),
            standardize=True,
        )
        df = pd.DataFrame(
            data.coef_.reshape(1, -1),
            columns=["B" + str(i + 1) for i in range(p)],
        )
        df.insert(0, "n", n)
        df.insert(0, 'id', len(df1))
        df1.append(df)

        df = pd.DataFrame(
            data.x, columns=["X" + str(i + 1) for i in range(p)]
        )
        df.insert(0, "y", data.y)
        df.insert(0, 'id', len(df2))
        df2.append(df)

pd.concat(df1, ignore_index=True).to_csv('info.csv')
pd.concat(df2, ignore_index=True).to_csv('data.csv')
