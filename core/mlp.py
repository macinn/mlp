import numpy as np
from core.layer import Layer
from core.losses import get_loss


class MLP:
    def __init__(self, layer_sizes, activations, loss_fun):
        assert len(layer_sizes) - 1 == len(
            activations
        ), "Jedna funkcja aktywacji na każdą warstwę ukrytą i wyjściową!"
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            for i in range(len(activations))
        ]
        self.loss, self.loss_deriv = get_loss(loss_fun)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, X, y, y_pred, learning_rate):
        dA = self.loss_deriv(y, y_pred)

        for i, layer in enumerate(reversed(self.layers)):
            idx = len(self.layers) - i - 1
            prev_A = X if idx == 0 else self.layers[idx - 1].A
            dA = layer.backward(dA, prev_A, learning_rate)

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        loss_history = []
        val_loss_history = []
        weight_history = []

        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = self.loss(y_train, y_pred)

            self.backward(X_train, y_train, y_pred, learning_rate)

            y_val_pred = self.forward(X_val)
            val_loss = self.loss(y_val, y_val_pred)

            loss_history.append(loss)
            val_loss_history.append(val_loss)
            avg_weights = [np.mean(layer.W) for layer in self.layers]
            weight_history.append(avg_weights)

            # if (epoch + 1) % max(1, (epochs // 10)) == 0:
            #     print(
            #         f"Epoka {epoch + 1}/{epochs}, błąd treningowy: {loss:.6f}, walidacyjny: {val_loss:.6f}"
            #     )

        return loss_history, val_loss_history, weight_history

    def predict(self, X):
        return self.forward(X)
