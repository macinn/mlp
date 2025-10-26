from enum import Enum
import numpy as np


class Fun(Enum):
    Sigmoid = "sigmoid"
    ReLU = "relu"
    Tanh = "tanh"
    Linear = "linear"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def linear(x):
    return x


def linear_derivative(x):
    return np.ones_like(x)


def get_activation(name: Fun):
    if name == Fun.Sigmoid:
        return sigmoid, sigmoid_derivative
    elif name == Fun.ReLU:
        return relu, relu_derivative
    elif name == Fun.Tanh:
        return tanh, tanh_derivative
    elif name == Fun.Linear:
        return linear, linear_derivative
    else:
        raise ValueError(f"Nieznana funkcja aktywacji: {name}")
