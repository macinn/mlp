from enum import Enum
import numpy as np


class LossFun(Enum):
    MSE = "mse"
    MAE = "mae"
    CrossEntropy = "cross_entropy"


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mae_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return y_pred - y_true


def cross_entropy(y_true, y_pred, eps: float = 1e-12):
    y_pred_clipped = np.clip(y_pred, eps, 1.0)
    losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
    return np.mean(losses)


def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true


def get_loss(name: LossFun):
    if name == LossFun.MSE:
        return mse, mse_derivative
    elif name == LossFun.MAE:
        return mae, mae_derivative
    elif name == LossFun.CrossEntropy:
        return cross_entropy, cross_entropy_derivative
    else:
        raise ValueError(f"Nieznana funkcja straty: {name}")
