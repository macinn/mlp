import numpy as np
from core.activations import get_activation


class Layer:
    def __init__(self, n_inputs, n_outputs, activation):
        limit = np.sqrt(6 / (n_inputs + n_outputs))
        self.W = np.random.uniform(-limit, limit, (n_inputs, n_outputs))
        self.b = np.zeros((1, n_outputs))
        self.activation, self.activation_deriv = get_activation(activation)
        self.Z = 0
        self.A = 0

    def forward(self, X):
        self.Z = np.dot(X, self.W) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA, A_prev, learning_rate):
        dZ = dA * self.activation_deriv(self.Z)
        m = A_prev.shape[0]
        dW = np.dot(A_prev.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.W.T)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dA_prev
