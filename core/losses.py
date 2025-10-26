from enum import Enum
import numpy as np


class LossFun(Enum):
    MSE = "mse"


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return y_pred - y_true


def get_loss(name: LossFun):
    if name == LossFun.MSE:
        return mse, mse_derivative
    else:
        raise ValueError(f"Nieznana funkcja straty: {name}")
