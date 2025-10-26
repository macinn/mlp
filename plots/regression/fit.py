import numpy as np
import matplotlib.pyplot as plt
from utils.normalize import normalize


def make_fit_plot(X, Y, mlp, normalize_input: bool = True, resolution: int = 200):
    X_grid = np.linspace(X.min(), X.max(), resolution).reshape(-1, 1)

    if normalize_input:
        _, X_mean, X_std = normalize(X)
        _, Y_mean, Y_std = normalize(Y)
        X_grid_n = (X_grid - X_mean) / X_std
        Y_pred_grid = mlp.forward(X_grid_n)
        Y_pred_grid = Y_pred_grid * Y_std + Y_mean
    else:
        Y_pred_grid = mlp.forward(X_grid)

    plt.figure(figsize=(6, 4))
    plt.scatter(X, Y, label="Dane treningowe", color="gray")
    plt.plot(X_grid, Y_pred_grid, color="red", linewidth=2, label="MLP")
    plt.title("Dopasowanie modelu MLP do danych regresyjnych")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
