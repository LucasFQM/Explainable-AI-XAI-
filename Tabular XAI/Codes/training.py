import torch


class Trainer:
    """
    Train a model using mini-batch gradient descent.
    """

    def __init__(self, model, train_dataloader, loss_fn, optimizer):
        """
        Initialize the model trainer.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be trained.
        train_dataloader : torch.utils.data.DataLoader
            DataLoader containing the training data.
        loss_fn : torch.nn.Module
            Loss function used to calculate the training error.
        optimizer : torch.optim.Optimizer
            Optimizer used to update the model parameters.

        Returns
        -------
        None
        """

        self.model = model
        self.train_dataloader = train_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_values = []

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of training epochs.

        Returns
        -------
        list of float
            Average training loss for each epoch.
        """

        for epoch in range(num_epochs):

            epoch_loss = 0.0
            self.model.train()

            for X_batch, y_batch in self.train_dataloader:

                self.optimizer.zero_grad()

                pred = self.model(X_batch)
                loss = self.loss_fn(pred, y_batch.unsqueeze(-1))

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(self.train_dataloader.dataset)
            self.loss_values.append(epoch_loss)

        return self.loss_values