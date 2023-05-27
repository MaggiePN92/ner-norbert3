import torch.nn as nn


class Layer(nn.Module):
    def __init__(self, hidden_size):
        super(Layer, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class BaselineANN(nn.Module):
    def __init__(
            self, 
            input_size, 
            n_layers, 
            hidden_size, 
            n_classses
        ):
        super(BaselineANN, self).__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_classses)
        self.layers = nn.ModuleList(
            [Layer(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.projection(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.out(x)
        return x

    def predict(self, x):
        return self.forward(x)
