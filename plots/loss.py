import matplotlib.pyplot as plt


def make_loss_plot(error_history):
    plt.figure(figsize=(6, 4))
    plt.plot(error_history)
    plt.title("Błąd uczenia (MSE)")
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()
