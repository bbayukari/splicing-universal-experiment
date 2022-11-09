import numpy as np
import time
from abess.universal import ConvexSparseSolver


import model
from abess import pybind_cabess
import sys
sys.path.append("/data/home/wangzz/github/splicing-universal-experiment/")
import MyTest
from MyTest import merge_dict
from MyTest import product_dict

pybind_cabess.init_spdlog(console_log_level=6, file_log_level=6)

def task(n ,p, k, sigma):
    result = {}
    # make dataset
    change_point = np.arange(n,step = int(n/(k+1)))[1:]
    Z = np.zeros((n, p))
    W = np.random.normal(size=(n, p)) * sigma
    for i in range(1,n):
        Z[i, :] = (Z[i-1, :] + np.random.normal(size=(1, p))) if i in change_point else Z[i-1, :]
    Y = Z + W
    Y = Y - np.mean(Y, axis=0)
    Y.reshape(n,p)

    X = np.zeros((n, n - 1))
    for j in range(1,n):        
        for i in range(1,j+1):
            X[i-1, j-1] = -np.sqrt((n-j) / (n * j))
        for i in range(j+1,n+1):
            X[i-1, j-1] = np.sqrt(j / (n * (n - j)))
    
    # run model
    t1 = time.time()
    cp_abess = model.abess(X, Y, len(change_point))
    t2 = time.time()
    cp_lasso, result['best_lambda'], cp_1, cp_10 = model.gfl(Y,len(change_point))
    t3 = time.time()
    
    # return
    result["lasso_accuracy"] = len(set(cp_lasso) & set(change_point)) / len(change_point)
    result["autodiff_accuracy"] = len(set(cp_abess) & set(change_point)) / len(change_point)
    result["lasso_1_accuracy"] = len(set(cp_1) & set(change_point)) / len(change_point)
    result["lasso_10_accuracy"] = len(set(cp_10) & set(change_point)) / len(change_point)
    result["autodiff_time"] = t2 - t1
    result["lasso_time"] = t3 - t2

    return result


if __name__ == "__main__":
    in_keys = ["n", "p", "k", "sigma"]
    out_keys = [
        method + "_" + term
        for term in ["accuracy", "time"]
        for method in ["autodiff", "lasso"]
        #for method in ["autodiff", "gurobi"]
    ] + ["best_lambda", "lasso_1_accuracy", "lasso_10_accuracy"]
    test = MyTest.Test(
        task, in_keys, out_keys, processes=40, name="group_linear"
    )
    # if n is very small, out of samples mse cann't be computed.
    test.check(n=100,p=3,k=3,sigma=1)
    #print("check done")
"""
    para = list(MyTest.del_duplicate(
        product_dict(n=[100], p=np.arange(20)+1, k=[1,5], sigma=np.sqrt([0.2]))),
    ) * 20
  
    #test start
    test.run(para)
    test.save()
"""
    
