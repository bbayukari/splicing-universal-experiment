from data_utils import load_mice, load_activity
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# 屏蔽 ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_data(seed):
    X_mice, Y_mice = load_mice()
    X_activity, Y_activity = load_activity()

    np.random.seed(seed)
    indices = np.random.permutation(len(X_mice))
    x_train_mice, y_train_mice = (
        X_mice[indices[: int(0.8 * len(X_mice))]],
        Y_mice[indices[: int(0.8 * len(X_mice))]],
    )
    x_test_mice, y_test_mice = (
        X_mice[indices[int(0.8 * len(X_mice)) :]],
        Y_mice[indices[int(0.8 * len(X_mice)) :]],
    )
    indices = np.random.permutation(len(X_activity))
    x_train_activity, y_train_activity = (
        X_activity[indices[: int(0.8 * len(X_activity))]],
        Y_activity[indices[: int(0.8 * len(X_activity))]],
    )
    x_test_activity, y_test_activity = (
        X_activity[indices[int(0.8 * len(X_activity)) :]],
        Y_activity[indices[int(0.8 * len(X_activity)) :]],
    )

    return (
        (x_train_mice, y_train_mice),
        (x_test_mice, y_test_mice),
        (x_train_activity, y_train_activity),
        (x_test_activity, y_test_activity),
    )


def run_experiment(train_data, test_data, num_selected_features, seed):
    x_train, y_train = train_data
    x_test, y_test = test_data
    # random selection of features
    np.random.seed(seed)
    selected_features = np.random.choice(
        x_train.shape[1], num_selected_features, replace=False
    )
    x_train = x_train[:, selected_features]
    x_test = x_test[:, selected_features]
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    return acc


if __name__ == "__main__":
    results = {
        "dataset": [],
        "method": [],
        "num_selected_features": [],
        "accuracy": [],
        "seed": [],
    }
    for seed in range(10):
        (
            (x_train_mice, y_train_mice),
            (x_test_mice, y_test_mice),
            (x_train_activity, y_train_activity),
            (x_test_activity, y_test_activity),
        ) = load_data(seed)
        # Activity
        for num_selected_features in list(range(25, 561 // 2, 25)) + [561]:
            acc = run_experiment(
                (x_train_activity, y_train_activity),
                (x_test_activity, y_test_activity),
                num_selected_features,
                seed,
            )
            results["dataset"].append("Activity")
            results["method"].append("random")
            results["num_selected_features"].append(num_selected_features)
            results["accuracy"].append(acc)
            results["seed"].append(seed)
            print(f"Activity {seed} random {num_selected_features} {acc}", flush=True)
        """
        # Mice
        for num_selected_features in list(range(5, 77 // 2, 5)) + [77]:
            acc = run_experiment(
                train_mice,
                test_mice,
                num_selected_features,
                seed,
            )
            results["dataset"].append("Mice")
            results["method"].append("random")
            results["num_selected_features"].append(num_selected_features)
            results["accuracy"].append(acc)
            results["seed"].append(seed)
            print(f"Mice {seed} random {num_selected_features} {acc}", flush=True)


        # Isolet
        for num_selected_features in list(range(25, 617 // 2, 25)) + [617]:
            acc = run_experiment(
                train_isolet,
                test_isolet,
                num_selected_features,
                seed,
            )
            results["dataset"].append("Isolet")
            results["method"].append("random")
            results["num_selected_features"].append(num_selected_features)
            results["accuracy"].append(acc)
            results["seed"].append(seed)
            print(f"Isolet {seed} random {num_selected_features} {acc}", flush=True)
        """
    # df = pd.DataFrame(results)
    # df.to_csv("results.csv", index=False)
