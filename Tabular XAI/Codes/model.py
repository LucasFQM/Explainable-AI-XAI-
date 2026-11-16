from torch import nn


class NeuralNetwork(nn.Module):
    """
    Feedforward neural network for binary classification.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the neural network architecture.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Number of neurons in the hidden layer.
        output_dim : int
            Number of output neurons.

        Returns
        -------
        None
        """

        super().__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Network output (logits).
        """

        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)

        return x