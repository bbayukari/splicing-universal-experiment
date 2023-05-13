from skscope_experiment import task as Task
from skscope_experiment import multi_target_task as MTask
from skscope_experiment import (
    linear,
    logistic,
    ising,
    trend_filtering_1d,
    non_linear_additive_example,
    non_linear_non_additive_example,
    multitask,
)
import parallel_experiment_util

model_dict = {
    "linear": linear,
    "logistic": logistic,
    "ising": ising,
    "trend_filter": trend_filtering_1d,
    "additive": non_linear_additive_example,
    "non_additive": non_linear_non_additive_example,
    "multitask": multitask,
}


def task(model, n, p, k, seed):
    if model in ['multitask']:
        return MTask(
            loss_jax=model_dict[model].loss_jax,
            loss_cvxpy=model_dict[model].loss_cvxpy,
            data_generator=model_dict[model].data_generator,
            sample_size=n,
            n_features=p,
            sparsity_level=k,
            n_targets=3,
            seed=seed,
        )
    return Task(
        loss_jax=model_dict[model].loss_jax,
        loss_cvxpy=model_dict[model].loss_cvxpy,
        data_generator=model_dict[model].data_generator,
        sample_size=n,
        dim=p,
        sparsity_level=k,
        seed=seed,
        cvxpy_constraints=getattr(model_dict[model], "cvxpy_constraints", None),
    )


if __name__ == "__main__":
    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["model", "n", "p", "k", "seed"],
        out_keys=["method", "time", "accuracy", "n_iters"],
        processes=8,
        name="linear-logistic-2",
        memory_limit=0.8,
    )

    if False:
        #experiment.check(model="multitask", n=100, p=50, k=5, seed=1)
        #experiment.check(model="additive", n=120, p=100, k=4, seed=1)
        # experiment.check(model="non_additive", n=100, p=100, k=3, seed=1)
        # experiment.check(model="trend_filter", n=100, p=100, k=1, seed=1)
        #experiment.check(model="linear", n=200, p=500, k=50, seed=10)
        #experiment.check(model="logistic", n=10, p=20, k=5, seed=100)
        experiment.check(model="ising", n=100, p=45, k=5, seed=200)
    else:
        parameters = parallel_experiment_util.para_generator(
            {
                "model": ["linear", "logistic"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
            },
            repeat=100,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()

"""
            {
                "model": ["multitask"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
            },
            {
                "model": ["additive"],
                "n": [i * 20 + 20 for i in range(10)],
                "p": [100],
                "k": [4],
            },
            {
                "model": ["non_additive"],
                "n": [i * 10 + 10 for i in range(10)],
                "p": [100],
                "k": [3],
            },
            {
                "model": ["trend_filter"],
                "n": [100],
                "p": [100],
                "k": [i+1 for i in range(10)],
            },
            {
                "model": ["linear", "logistic"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [500],
                "k": [50],
            },
            {
                "model": ["ising"],
                "n": [i * 100 + 100 for i in range(10)],
                "p": [190],
                "k": [30],
            },
"""
