import torch.nn as nn


class MLP(nn.Module):
    """
    MLP with flexible hidden dimensions.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout, use_residual=False):
        """
        A flexible MLP that supports variable hidden dimensions for each layer.

        :param input_dim: Input feature dimension.
        :param hidden_dims: List of hidden layer dimensions.
        :param output_dim: Output feature dimension.
        :param use_residual: Whether to use residual connections.
        """
        super(MLP, self).__init__()
        self.use_residual = use_residual

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        :param x: Input tensor to the model.
        :return: Output tensor after passing through the network and dropout layers.
        """
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual

        x = self.output_layer(x)
        return x
