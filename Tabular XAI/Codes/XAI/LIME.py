
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


class LIME:
    """
    Generate and visualize local explanations using LIME.
    """

    def __init__(
        self,
        X_train,
        X_test,
        feature_names,
        classes,
        mode="classification",
        discretize_continuous=False,
        random_state=42
    ):
        """
        Initialize the LIME explainer.

        Parameters
        ----------
        X_train : array-like
            Training feature matrix used to generate the LIME explainer.
        X_test : array-like
            Test feature matrix containing the instances to be explained.
        feature_names : list of str
            Names of the input features.
        classes : list
            Class names used in the explanations.
        mode : {'classification', 'regression'}, default='classification'
            Type of prediction task.
        discretize_continuous : bool, default=False
            Whether to discretize continuous features into intervals.
        random_state : int, default=42
            Random seed used for reproducibility.

        Returns
        -------
        None
        """

        self.X_train = self._to_numpy(X_train)
        self.X_test = self._to_numpy(X_test)

        self.feature_names = feature_names
        self.classes = classes
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        self.random_state = random_state

        self.explainer = LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=self.classes,
            mode=self.mode,
            discretize_continuous=self.discretize_continuous,
            random_state=self.random_state
        )

    @staticmethod
    def _to_numpy(data):
        """
        Convert the input data into a NumPy array.

        Parameters
        ----------
        data : array-like or torch.Tensor
            Data to be converted.

        Returns
        -------
        numpy.ndarray
            Converted data.
        """

        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()

        return np.asarray(data)

    def explain(
        self,
        predict_fn,
        instance_index,
        num_features=10,
        num_samples=5000
    ):
        """
        Generate local explanations for a test instance
        using perturbed samples and a local surrogate model.

        Parameters
        ----------
        predict_fn : callable
            Function that receives a data matrix and returns
            model predictions or class probabilities.
        instance_index : int
            Index of the test instance to be explained.
        num_features : int, default=10
            Maximum number of features included in the explanation.
        num_samples : int, default=5000
            Number of perturbed samples generated to approximate
            the local decision boundary.

        Returns
        -------
        lime.explanation.Explanation
            Local explanation containing the feature contributions
            and their corresponding weights.
        """

        instance = self.X_test[instance_index]

        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=tuple(range(len(self.classes)))
        )

        return explanation

    @staticmethod
    def plot(explanation, label=1):
        """
        Plot the feature contributions of a local explanation.

        Parameters
        ----------
        explanation : lime.explanation.Explanation
            Local explanation returned by the explain() method.
        label : int, default=1
            Class label whose feature contributions are plotted.

        Returns
        -------
        matplotlib.figure.Figure
            Figure displaying the local feature contributions.
        """

        fig = explanation.as_pyplot_figure(label=label)

        plt.tight_layout()

        return fig
