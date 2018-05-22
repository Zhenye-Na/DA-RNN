"""Helper functions.

@author Zhenye Na 05/21/2018

"""


# from tqdm import tqdm
# import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')


def read_data(input_path):
    """Read nasdaq stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path)
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    return X, y


def train_val_test_split(X, y, is_Val):
    """training, val, test set split.

    Args:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.
        is_Val (boolean): whether need a validation set.
                          default: True

    Returns:
        X_train (np.ndarray): training features.
        y_train (np.ndarray) training set groundtruth.
        X_test (np.ndarray): test features.
        y_test (np.ndarray): groundtruth for prediction.
        X_val (np.ndarray): validation features.
        y_val (np.ndarray): validation set groundtruth.

    ##########################################################
    According to paper:
        The first 35,100 data points as the training set
        The following 2,730 data points as the validation set.
        The last 2,730 data points are used as the test set.
    ##########################################################

    Adjusted by making size divided by batch size (128).

    """
    # Train set
    X_train = X[0:35072, :]
    y_train = y[0:35072]

    # Test set
    X_test = X[37830:40448, :]
    y_test = y[37830:40448]

    # Val set
    if is_Val:
        X_val = X[35072:37760, :]
        y_val = y[35072:37760]
    else:
        X_val = np.zeros_like(X_test)
        y_val = np.zeros_like(y_test)

    return X_train, y_train, X_test, y_test, X_val, y_val


def plot():
    """plotting."""
    pass
