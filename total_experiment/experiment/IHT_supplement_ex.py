import numpy as np
import pandas as pd
import time

import variable_select_algorithm
import parallel_experiment_util
import statistic_model
import abess

def linear_loss(para, data):
    return np.sum(np.square(data.y - data.x @ para))

def linear_grad(para, data, compute_para_index):
    return 2 * data.x[:,compute_para_index].T @ (data.x @ para - data.y)


class IsingData:
    def __init__(self, data):
        self.n = data.shape[0]
        self.p = data.shape[1] - 1
        self.table = data[:, 1:]
        self.freq = data[:, 0]

        self.index_translator = np.zeros(shape=(self.p, self.p), dtype=np.int32)
        idx = 0
        for i in range(self.p):
            for j in range(i+1, self.p):
                self.index_translator[i,j] = idx
                self.index_translator[j,i] = idx
                idx += 1

def ising_loss(para, data):
    loss = 0.0
    for i in range(data.n):
        for k in range(data.p):
            tmp = 0.0
            for j in range(data.p):
                if j != k:
                    tmp -= 2 * para[data.index_translator[k,j]] * data.table[i,j] * data.table[i,k]
            loss += data.freq[i] * np.logaddexp(0.0, tmp)
    return loss

def ising_grad(para, data, compute_para_index):
    grad_para = np.zeros(para.shape)

    for i in range(data.n):
        for k in range(data.p):
            tmp = 0.0
            for j in range(data.p):
                if j == k:
                    continue
                tmp += data.table[i,k] * data.table[i,j] * para[data.index_translator[k,j]]
            exp_tmp = 2 * data.freq[i] * data.table[i,k] / (1 + np.exp(np.clip(2 * tmp, -100, 100)))
            for j in range(data.p):
                if j == k:
                    continue
                grad_para[data.index_translator[k,j]] -= exp_tmp * data.table[i,j]

    return grad_para[compute_para_index]


def linear_task(n, seed):
    result = {}
    # make dataset
    p = 500
    k = 50
    coef = np.zeros(p)
    np.random.seed(seed)
    coef[np.random.choice(np.arange(p), k, replace=False)] = np.random.choice([100, -100], k)
    data = abess.make_glm_data(
        n=n,
        p=p,
        k=k,
        rho=0.2,
        family="gaussian",
        corr_type="exp",
        snr=10 * np.log10(6),
        standardize=True,
        coef_=coef
    )   


    GraHTP_coef = variable_select_algorithm.GraHTP(
        loss_fn=linear_loss,
        grad_fn=linear_grad,
        dim=p,
        data=data,
        fast=True,
        support_size=k,
        step_size=2e-3,
    )

    GraHTP_cv_coef, best_step_size = variable_select_algorithm.GraHTP_cv(
        loss_fn=linear_loss,
        grad_fn=linear_grad,
        dim=p,
        data=data,
        fast=True,
        support_size=k,
    )
 

    result["IHT_accuracy"] = parallel_experiment_util.accuracy(
        GraHTP_coef, data.coef_
    )
    result["IHT_cv_accuracy"] = parallel_experiment_util.accuracy(
        GraHTP_cv_coef, data.coef_
    )
    result["model"] = "linear"
    return result

def ising_task(n, seed):
    result = {}
    # make dataset
    k = 40
    data, theta, coef = statistic_model.ising_generator(P=20, N=n, Edges=k, seed=seed)
    dim = 190
    support_size = k
    dataset = IsingData(data)

    # run model
    GraHTP_coef = variable_select_algorithm.GraHTP(
        loss_fn=ising_loss,
        grad_fn=ising_grad,
        dim=dim,
        data=dataset,
        fast=True,
        support_size=support_size,
        step_size=2e-3,
    )
    GraHTP_cv_coef, best_step_size = variable_select_algorithm.GraHTP_cv(
        loss_fn=ising_loss,
        grad_fn=ising_grad,
        dim=dim,
        data=dataset,
        fast=True,
        support_size=support_size,
    )

    # return
    result["IHT_accuracy"] = parallel_experiment_util.accuracy(GraHTP_coef, coef)
    result["IHT_cv_accuracy"] = parallel_experiment_util.accuracy(
        GraHTP_cv_coef, coef
    )
    result["model"] = "ising"
    return result


def experiment(task, para_file, isTest=False):
    in_keys = ["n", "seed"]
    out_keys = [
        "IHT_accuracy",
        "IHT_cv_accuracy",
        "model"
    ]

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=in_keys,
        out_keys=out_keys,
        processes=40,
        name=para_file.replace("experiment", "supplement_IHT").split(".")[0],
        memory_limit=80
    )

    if isTest:
        experiment.check(n=100, seed=1)
        return

    parameters = pd.read_csv(
        "/data/home/wangzz/github/splicing-universal-experiment/total_experiment/results/" + para_file, 
        usecols=['n', 'seed'])
    
    experiment.run(parameters.to_dict("records"))
    experiment.save()


if __name__ == "__main__":
    experiment(linear_task, "linear_experiment.csv")
    experiment(ising_task, "ising_experiment.csv")
