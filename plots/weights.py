import numpy as np
import matplotlib.pyplot as plt


def make_weights_avg_plot(weight_history):
    weight_history = np.array(weight_history)

    plt.figure(figsize=(8, 4))
    for i in range(weight_history.shape[1]):
        plt.plot(weight_history[:, i], label=f"Warstwa {i+1}")
    plt.title("Ewolucja średnich wag w czasie")
    plt.xlabel("Epoka")
    plt.ylabel("Średnia wartość wag")
    plt.legend()
    plt.grid(True)
    plt.show()


def make_weights_histogram_plot(weights_before, weights_after):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(weights_before.flatten(), bins=30)
    plt.title("Rozkład wag - przed treningiem")

    plt.subplot(1, 2, 2)
    plt.hist(weights_after.flatten(), bins=30)
    plt.title("Rozkład wag - po treningu")
    plt.show()


def make_weights_heatmap(mlp, epoch=None):
    n_layers = len(mlp.layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    for i, layer in enumerate(mlp.layers):
        ax = axes[i]
        im = ax.imshow(layer.W, cmap="coolwarm", aspect="auto")
        ax.set_title(f"Wagi: warstwa {i+1}")
        ax.set_xlabel("Neurony warstwy wyjściowej")
        ax.set_ylabel("Neurony warstwy wejściowej")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Heatmap wag{' (epoka ' + str(epoch) + ')' if epoch else ''}", fontsize=14
    )
    plt.tight_layout()
    plt.show()
