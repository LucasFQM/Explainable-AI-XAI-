import torch
import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    """
    Custom dataset for tabular data.
    """

    def __init__(self, X, y):
        """
        Initialize the dataset by converting the input data
        into PyTorch tensors.

        Parameters
        ----------
        X : array-like
            Input feature matrix.
        y : array-like
            Target values or labels.

        Returns
        -------
        None
        """

        self.X = torch.from_numpy(
            np.asarray(X, dtype=np.float32)
        )

        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.float32)
        )

    def __getitem__(self, index):
        """
        Retrieve a sample and its corresponding target.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            Input features and corresponding target.
        """

        return self.X[index], self.y[index]

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """

        return len(self.X)