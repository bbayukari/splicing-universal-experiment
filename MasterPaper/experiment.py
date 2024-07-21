import time
from data_utils import load_mice, load_activity, load_isolet
import jax.numpy as jnp
import jax
import pandas as pd
from skscope import (
    ScopeSolver,
    HTPSolver,
    GraspSolver,
    IHTSolver,
    BaseSolver,
    FobaSolver,
)
import numpy as np
import parallel_experiment_util


def load_data(rng: np.random.Generator):
    X_mice, Y_mice = load_mice()
    X_activity, Y_activity = load_activity()
    X_isolet, Y_isolet = load_isolet()
    Y_mice = jax.nn.one_hot(Y_mice, len(set(Y_mice)))
    Y_activity = jax.nn.one_hot(Y_activity, len(set(Y_activity)))
    Y_isolet = jax.nn.one_hot(Y_isolet, len(set(Y_isolet)))

    indices = rng.permutation(len(X_mice))
    x_train_mice, y_train_mice = (
        X_mice[indices[: int(0.8 * len(X_mice))]],
        Y_mice[indices[: int(0.8 * len(X_mice))]],
    )
    x_test_mice, y_test_mice = (
        X_mice[indices[int(0.8 * len(X_mice)) :]],
        Y_mice[indices[int(0.8 * len(X_mice)) :]],
    )
    indices = rng.permutation(len(X_activity))
    x_train_activity, y_train_activity = (
        X_activity[indices[: int(0.8 * len(X_activity))]],
        Y_activity[indices[: int(0.8 * len(X_activity))]],
    )
    x_test_activity, y_test_activity = (
        X_activity[indices[int(0.8 * len(X_activity)) :]],
        Y_activity[indices[int(0.8 * len(X_activity)) :]],
    )
    indices = rng.permutation(len(X_isolet))
    x_train_isolet, y_train_isolet = (
        X_isolet[indices[: int(0.8 * len(X_isolet))]],
        Y_isolet[indices[: int(0.8 * len(X_isolet))]],
    )
    x_test_isolet, y_test_isolet = (
        X_isolet[indices[int(0.8 * len(X_isolet)) :]],
        Y_isolet[indices[int(0.8 * len(X_isolet)) :]],
    )

    return {
        "Mice": (x_train_mice, y_train_mice, x_test_mice, y_test_mice),
        "Activity": (
            x_train_activity,
            y_train_activity,
            x_test_activity,
            y_test_activity,
        ),
        "Isolet": (x_train_isolet, y_train_isolet, x_test_isolet, y_test_isolet),
    }


def accuaracy(params, x, y, p, m):
    return jnp.mean(jnp.argmax(jnp.dot(x, params.reshape((p, m))), axis=1) == jnp.argmax(y, axis=1)).item()


def run_experiment(solver_generator, num_selected_features: int, data, rng):
    x_train, y_train, x_test, y_test = data
    p, m = x_train.shape[1], y_train.shape[1]
    match p:
        case 77:
            dataset = "Mice"
        case 561:
            dataset = "Activity"
        case 617:
            dataset = "Isolet"
        case _:
            raise ValueError("Unknown dataset")

    def multinomial_regression_loss(params):
        return -jnp.sum(jax.nn.log_softmax(jnp.dot(x_train, params.reshape((p, m)))) * y_train)

    if solver_generator.__name__ == "BaseSolver":
        selected_features = rng.choice(p, num_selected_features, replace=False)
        x_train = x_train[:, selected_features]
        x_test = x_test[:, selected_features]
        p = num_selected_features

    solver = solver_generator(p * m, num_selected_features, group=[i for i in range(p) for j in range(m)])
    t1 = time.time()
    params = solver.solve(multinomial_regression_loss, jit=True)
    res = {
        "dataset": dataset,
        "method": solver_generator.__name__,
        "num_selected_features": num_selected_features,
        "accuracy": accuaracy(params, x_test, y_test, p, m),
        "time": time.time() - t1,
    }
    print(res, flush=True)
    return res


def task(seed):
    results = []
    rng = np.random.default_rng(seed + 321545648942)
    Data = load_data(rng)
    for solver_generator in (
        FobaSolver,
        # ScopeSolver,
        # HTPSolver,
        # GraspSolver,
        # IHTSolver,
        # BaseSolver,
    ):
        for dataset in ["Mice", "Activity"]:  # "Isolet"
            match dataset:
                case "Mice":
                    data = Data["Mice"]
                    features_list = list(range(2, 21, 2))
                case "Activity":
                    data = Data["Activity"]
                    features_list = list(range(10, 101, 10))
                case "Isolet":
                    data = Data["Isolet"]
                    features_list = list(range(25, 617 // 2, 25))

            for num_selected_features in features_list:
                results.append(
                    run_experiment(
                        solver_generator,
                        num_selected_features,
                        data,
                        rng,
                    )
                )
    return results


if __name__ == "__main__":

    experiment = parallel_experiment_util.ParallelExperiment(
        task=task,
        in_keys=["seed"],
        out_keys=["method", "dataset", "accuracy", "num_selected_features", "time"],
        processes=10,
        name="foba_t",
        memory_limit=0.9,
    )
    if False:
        experiment.check(seed=100)
    else:
        parameters = parallel_experiment_util.para_generator(
            {},
            repeat=10,
            seed=0,
        )

        experiment.run(parameters)
        experiment.save()
