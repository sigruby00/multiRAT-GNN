import torch.nn as nn
import torch
import torch.nn.functional as F


class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FNNModel, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.ReLU())  # Output activation
        self.layers = nn.Sequential(*layers)

        # Output scaling parameter (learnable)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.layers(x)
        x = self.alpha * x  # Scaling alpha
        return x


class FNNModel_adv(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation="relu",
        dropout_prob=0.2,
        use_batchnorm=True,
        use_alpha=True,
    ):
        super(FNNModel_adv, self).__init__()
        self.use_alpha = use_alpha
        layers = []
        prev_size = input_size

        # Define activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "softplus":
            self.activation_fn = nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Build hidden layers
        self.hidden_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, size))
            if use_batchnorm:
                self.batchnorm_layers.append(nn.BatchNorm1d(size))
            else:
                self.batchnorm_layers.append(None)
            if dropout_prob > 0:
                self.dropout_layers.append(nn.Dropout(p=dropout_prob))
            else:
                self.dropout_layers.append(None)
            prev_size = size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

        # Output scaling parameter (learnable)
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.alpha = None

    def reset_parameters(self):
        """Initialize weights of the model."""
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)  # Xavier uniform initialization
            nn.init.zeros_(layer.bias)  # Bias initialization to zero

        nn.init.xavier_uniform_(self.output_layer.weight)  # Xavier initialization for output
        nn.init.zeros_(self.output_layer.bias)

        # Initialize BatchNorm layers if present
        for bn_layer in self.batchnorm_layers:
            if bn_layer is not None:
                bn_layer.reset_parameters()

        # Initialize alpha parameter to 1.0 if used
        if self.use_alpha and self.alpha is not None:
            with torch.no_grad():
                self.alpha.fill_(1.0)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batchnorm_layers[i] is not None:
                x = self.batchnorm_layers[i](x)
            x = self.activation_fn(x)
            if self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x)
        x = self.output_layer(x)
        if self.alpha is not None:
            x = self.alpha * x  # Scale output by alpha
        return x
