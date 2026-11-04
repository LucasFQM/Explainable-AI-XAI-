import torch


class Trainer:

    def __init__(self, model, train_dataloader, loss_fn, optimizer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_values = []

    def train(self, num_epochs):

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