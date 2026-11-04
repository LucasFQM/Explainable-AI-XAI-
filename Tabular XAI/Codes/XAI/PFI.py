import torch
import numpy as np
import matplotlib.pyplot as plt


class PermutationFeatureImportance:
    """
    Permutation Feature Importance for PyTorch models.

    """

    def __init__(
        self,
        model,
        X_test,
        y_test,
        feature_names,
        n_repeats=30,
        random_state=42
    ):

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.random_state = random_state

    def evaluate_accuracy(self, X):
        """Evaluate model accuracy."""

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(X)
            predictions = (outputs >= 0).int().squeeze(1)

            accuracy = (
                predictions == self.y_test.int()
            ).float().mean().item()

        return accuracy

    def calculate_importance(self):
        """Calculate permutation feature importance."""

        baseline_accuracy = self.evaluate_accuracy(self.X_test)

        importances = []

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        for feature_idx in range(self.X_test.shape[1]):

            importance_values = []

            for _ in range(self.n_repeats):

                X_permuted = self.X_test.clone()

                permutation = torch.randperm(
                    X_permuted.shape[0],
                    generator=generator
                )

                X_permuted[:, feature_idx] = (
                    X_permuted[permutation, feature_idx]
                )

                permuted_accuracy = self.evaluate_accuracy(
                    X_permuted
                )

                importance = baseline_accuracy - permuted_accuracy

                importance_values.append(importance)

            importances.append({
                "feature": self.feature_names[feature_idx],
                "importance_mean": np.mean(importance_values)
            })

        importances.sort(
            key=lambda x: x["importance_mean"],
            reverse=True
        )

        return baseline_accuracy, importances

    def plot_importance(self, importances):

        """Visualize permutation feature importance."""

        features = [item["feature"] for item in importances]

        values = [
            item["importance_mean"] for item in importances
        ]

        fig, ax = plt.subplots(figsize=(10, 9))

        ax.barh(
            features,
            values
        )

        ax.axvline(x=0, color="black", linewidth=0.8)

        ax.set_xlabel("Mean Decrease in Accuracy")
        ax.set_ylabel("Features")

        ax.invert_yaxis()

        fig.tight_layout()
        plt.show()

