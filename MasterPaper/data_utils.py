import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler

PATH = "/home/wangzz/github/splicing-universal-experiment/MasterPaper/"
MICE_PATH = PATH + "dataset/MICE/MICE/Data_Cortex_Nuclear.csv"
ISOLET_PATH = PATH + "dataset/isolet/isolet/"
ACT_PATH = PATH + "dataset/dataset_uci/dataset_uci/"


def load_mice(one_hot=False):
    filling_value = -100000
    X = np.genfromtxt(
        MICE_PATH,
        delimiter=",",
        skip_header=1,
        usecols=range(1, 78),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    classes = np.genfromtxt(
        MICE_PATH,
        delimiter=",",
        skip_header=1,
        usecols=range(78, 81),
        dtype=None,
        encoding="UTF-8",
    )

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean(
                    [
                        X[k, j]
                        for k in range(classes.shape[0])
                        if np.all(classes[i] == classes[k])
                    ]
                )

    DY = np.zeros((classes.shape[0]), dtype=np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
            DY[i] += (2**j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]
    classes = classes[indices]

    if not one_hot:
        Y = DY

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    return X, Y


def load_activity():
    x_train = np.loadtxt(ACT_PATH + "final_X_train.txt", encoding="UTF-8")
    x_test = np.loadtxt(ACT_PATH + "final_X_test.txt", encoding="UTF-8")
    y_train = np.loadtxt(ACT_PATH + "final_y_train.txt", encoding="UTF-8")
    y_test = np.loadtxt(ACT_PATH + "final_y_test.txt", encoding="UTF-8")

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    Y = np.concatenate((y_train, y_test))

    return X, Y


def load_isolet():
    x_train = np.genfromtxt(
        ISOLET_PATH + "isolet1+2+3+4.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_train = np.genfromtxt(
        ISOLET_PATH + "isolet1+2+3+4.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    x_test = np.genfromtxt(
        ISOLET_PATH + "isolet5.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_test = np.genfromtxt(
        ISOLET_PATH + "isolet5.data", delimiter=",", usecols=[617], encoding="UTF-8"
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    Y = np.concatenate((y_train, y_test)) - 1

    return X, Y


if __name__ == "__main__":
    load_mice()
    load_activity()
