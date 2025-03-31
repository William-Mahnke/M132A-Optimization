import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=10, num_layers=3, activation='sigmoid'):
        super(NeuralNetwork, self).__init__()
        '''
        Class variables:
            hidden_dim: the number of neurons in the hidden layer
            drop_out: the dropout rate
            Activation_before_batchnorm: whether to apply the activation function before the batch normalization
            activation: the activation function to use (sigmoid, or tanh, can explore more later if needed)
        '''
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # flexible activation function selection
        self.activation = activation.lower()
        if self.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function type, {}".format(self.activation))
        
        # first layer
        self.layer1 = nn.Linear(3, hidden_dim) # x, y, t inputs

        # hidden layers
        self.layers = nn.ModuleList()
        for i in range(num_layers - 2): # subtract 2 for the input and output layers
            if num_layers <= 2:
                raise ValueError("Number of layers must be greater than 2")
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # output layer
        self.layer_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.layer_out(x)
        return x
        
    def get_activation(self):
        return self.activation

    def get_hidden_dim(self):
        return self.hidden_dim

if __name__ == "__main__":
    # test
    model = NeuralNetwork()
    print(model.get_activation())
    print(model.get_hidden_dim())
    x = torch.randn(1000, 3)
    soln = model(x)
    print(soln.shape)
