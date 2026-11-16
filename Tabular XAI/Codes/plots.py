import matplotlib.pyplot as plt


class LossPlotter:
    """
    Visualize the training loss over epochs.
    """

    def __init__(self, loss_values):
        """
        Initialize the loss plotter.

        Parameters
        ----------
        loss_values : list of float
            Training loss values recorded at each epoch.

        Returns
        -------
        None
        """

        self.loss_values = loss_values

    def plot_loss(self):
        """
        Plot the training loss as a function of the epochs.

        Returns
        -------
        None
        """

        epochs = range(1, len(self.loss_values) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(epochs, self.loss_values)

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")

        ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()