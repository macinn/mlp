import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, resolution=200):
    # siatka punktów
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # predykcje dla siatki
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # rysowanie
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
    plt.scatter(
        X[:, 0], X[:, 1], c=np.argmax(y, axis=1), cmap=plt.cm.coolwarm, edgecolors="k"
    )
    plt.title("Granice decyzyjne MLP")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
