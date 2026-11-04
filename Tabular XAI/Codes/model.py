from torch import nn


class NeuralNetwork(nn.Module):
    """
    Feedforward Neural Network for binary classification.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)

        return x