from scope import ScopeSolver
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt
import os

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = ""  # force use CPU because there is an error when using GPU in my environment
# RuntimeError: jaxlib/cusolver_kernels.cc:44: operation cusolverDnCreate(&handle) failed: cuSolver internal error
n, p = 200, 15

np_rdm = np.random.RandomState(0)
pre = make_sparse_spd_matrix(
    p, alpha=0.8, smallest_coef=0.5, largest_coef=1, random_state=np_rdm
)
k = int((np.count_nonzero(pre) - p) / 2)
cov = np.linalg.inv(pre)
d = np.sqrt(np.diag(cov))
cov /= d
cov /= d[:, np.newaxis]
pre *= d
pre *= d[:, np.newaxis]
X = np_rdm.multivariate_normal(np.zeros(p), cov, size=n)
X -= X.mean(axis=0)
emp_cov = np.dot(X.T, X) / n
emp_pre = np.linalg.inv(emp_cov)

def graphical_guassian_objective(params):
    Omega = jnp.zeros((p, p))
    Omega = Omega.at[np.triu_indices(p)].set(params)
    Omega = jnp.where(Omega, Omega, Omega.T)
    return jnp.sum(emp_cov * Omega) - jnp.log(jnp.linalg.det(Omega))


solver = ScopeSolver(
    int(p * (p + 1) / 2),
    k + p,
    always_select=np.where(np.triu_indices(p)[0] == np.triu_indices(p)[1])[0],
    #file_log_level="debug",
    #regular_coef=1.0
)  # always select diagonal elements
solver.solve(graphical_guassian_objective, init_params=np.eye(p)[np.triu_indices(p)])
params = solver.get_params()

print(np.nonzero(pre[np.triu_indices(p)])[0])
print(solver.get_support_set())
print(pre[np.triu_indices(p)])
print(params)
