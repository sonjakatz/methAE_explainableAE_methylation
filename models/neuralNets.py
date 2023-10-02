import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self,
                inputDim,
                hidden_layer_topology,
                outDim=1,
                binary=False,
                relu=True):
        super(NeuralNet, self).__init__()
        self.inputDim = inputDim
        self.hidden_layer_topology = hidden_layer_topology
        self.outDim = outDim
        
        self.topology = [self.inputDim] + self.hidden_layer_topology #+ [self.outDim] 
        
        self.layers = []
        ### Hidden layers
        for i in range(len(self.topology)-1):
            layer = nn.Linear(self.topology[i], self.topology[i+1])
            torch.nn.init.xavier_uniform_(layer.weight)  ## weight initialisation
            self.layers.append(layer)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(self.topology[i+1]))
        ### Final layer
        if hidden_layer_topology:
            final_layer = nn.Linear(self.hidden_layer_topology[-1], self.outDim)
        else:
            final_layer = nn.Linear(self.inputDim, self.outDim)
        self.layers.append(final_layer)
        if binary:
            self.layers.append(nn.Sigmoid())
        elif relu:
            self.layers.append(nn.ReLU())
        
        self.net_topology = nn.Sequential(*self.layers)
        
    def forward(self, x):
        y_hat = self.net_topology(x)
        return y_hat

