from enum import Enum
import numpy as np


class LossFun(Enum):
    MSE = "mse"
    MAE = "mae"


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mae_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return y_pred - y_true


def get_loss(name: LossFun):
    if name == LossFun.MSE:
        return mse, mse_derivative
    elif name == LossFun.MAE:
        return mae, mae_derivative
    else:
        raise ValueError(f"Nieznana funkcja straty: {name}")
