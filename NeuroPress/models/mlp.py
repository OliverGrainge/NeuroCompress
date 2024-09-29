import torch.nn as nn
import torch.nn.functional as F 


class MLP(nn.Module):
    def __init__(self, layer_type, input_size, hidden_size, num_classes, num_layers):
        """
        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layers.
            num_classes (int): Number of output classes.
            num_layers (int): Number of hidden layers.
        """
        super(MLP, self).__init__()
        
        # First layer (input layer to first hidden layer)
        self.layers = nn.ModuleList([layer_type(input_size, hidden_size)])
        
        # Hidden layers (hidden_size to hidden_size)
        for _ in range(1, num_layers):
            self.layers.append(layer_type(hidden_size, hidden_size))
        
        # Output layer (hidden layer to output)
        self.layers.append(layer_type(hidden_size, num_classes))

    def forward(self, x):
        # Pass through each layer with ReLU activation in between
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Output layer without activation (assumes classification)
        x = self.layers[-1](x)
        return x