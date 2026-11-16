import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from model import NeuralNetwork
from training import Trainer
from data import Data


class DropColumnImportance:
    """
    Compute and visualize Drop-Column Feature Importance.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        hidden_dim=10,
        learning_rate=0.01,
        epochs=20,
        batch_size=32,
        random_state=42
    ):
        """
        Initialize the Drop-Column Importance evaluator.

        Parameters
        ----------
        X_train : array-like
            Training feature matrix.
        y_train : array-like
            Training target labels.
        X_test : array-like
            Test feature matrix.
        y_test : array-like
            Test target labels.
        feature_names : list of str
            Names of the input features.
        hidden_dim : int, default=10
            Number of neurons in the hidden layer.
        learning_rate : float, default=0.01
            Learning rate used by the optimizer.
        epochs : int, default=20
            Number of training epochs.
        batch_size : int, default=32
            Number of samples per training batch.
        random_state : int, default=42
            Random seed used for reproducibility.

        Returns
        -------
        None
        """

        self.y_train = torch.as_tensor(
            np.asarray(y_train),
            dtype=torch.float32
        )

        self.y_test = torch.as_tensor(
            np.asarray(y_test),
            dtype=torch.float32
        )

        self.X_train = torch.as_tensor(
            np.asarray(X_train),
            dtype=torch.float32
        )

        self.X_test = torch.as_tensor(
            np.asarray(X_test),
            dtype=torch.float32
        )

        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_model(self, X_train, X_test):
        """
        Train a model using the provided training features
        and evaluate its accuracy on the test dataset.

        Parameters
        ----------
        X_train : torch.Tensor
            Training feature matrix.
        X_test : torch.Tensor
            Test feature matrix.

        Returns
        -------
        float
            Classification accuracy on the test dataset.
        """

        train_data = Data(
            X_train.numpy(),
            self.y_train.numpy()
        )

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator
        )

        model = NeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=1
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate
        )

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            loss_fn=self.loss_fn,
            optimizer=optimizer
        )

        trainer.train(num_epochs=self.epochs)

        model.eval()

        with torch.no_grad():

            outputs = model(X_test)
            predictions = (outputs >= 0).int().squeeze(1)

            accuracy = (
                predictions == self.y_test.int()
            ).float().mean().item()

        return accuracy

    def calculate_importance(self):
        """
        Calculate feature importance by measuring the decrease
        in test accuracy after removing each feature and
        retraining the model.

        Returns
        -------
        baseline_accuracy : float
            Test accuracy of the model trained using all features.
        importances : list of dict
            Features ranked by importance, with each dictionary
            containing the feature name and its decrease
            in test accuracy.
        """

        # Baseline model using all features
        baseline_accuracy = self.train_model(
            self.X_train,
            self.X_test
        )

        importances = []

        # Remove one feature at a time
        for feature_idx in range(self.X_train.shape[1]):

            X_train_reduced = torch.cat(
                (
                    self.X_train[:, :feature_idx],
                    self.X_train[:, feature_idx + 1:]
                ),
                dim=1
            )

            X_test_reduced = torch.cat(
                (
                    self.X_test[:, :feature_idx],
                    self.X_test[:, feature_idx + 1:]
                ),
                dim=1
            )

            reduced_accuracy = self.train_model(
                X_train_reduced,
                X_test_reduced
            )

            importance = (
                baseline_accuracy - reduced_accuracy
            )

            importances.append({
                "feature": self.feature_names[feature_idx],
                "importance_mean": importance
            })

        importances.sort(
            key=lambda x: x["importance_mean"],
            reverse=True
        )

        return baseline_accuracy, importances

    def plot_importance(self, importances):
        """
        Plot the decrease in test accuracy associated with
        removing each feature.

        Parameters
        ----------
        importances : list of dict
            Feature importance results returned by
            calculate_importance().

        Returns
        -------
        None
        """

        features = [
            item["feature"] for item in importances
        ]

        values = [
            item["importance_mean"] for item in importances
        ]

        fig, ax = plt.subplots(figsize=(10, 9))

        ax.barh(features, values)

        ax.axvline(
            x=0,
            color="black",
            linewidth=0.8
        )

        ax.set_xlabel("Mean Decrease in Accuracy")
        ax.set_ylabel("Features")

        ax.invert_yaxis()

        fig.tight_layout()
        plt.show()