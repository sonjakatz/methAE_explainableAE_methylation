import torch
from torch import nn
import numpy as np


class normalAE(nn.Module):
    def __init__(self,
                inputDim,
                latentSize,
                hidden_layer_encoder_topology=[]):
        super(normalAE, self).__init__()
        self.inputDim = inputDim
        self.hidden_layer_encoder_topology = hidden_layer_encoder_topology
        self.latentSize = latentSize

        ### Define encoder
        self.encoder_topology = [self.inputDim] + self.hidden_layer_encoder_topology + [self.latentSize]
        self.encoder_layers = []
        for i in range(len(self.encoder_topology)-1):
            layer = nn.Linear(self.encoder_topology[i],self.encoder_topology[i+1])
            torch.nn.init.xavier_normal_(layer.weight)  ## weight initialisation
            self.encoder_layers.append(layer)
            self.encoder_layers.append(nn.PReLU())
            self.encoder_layers.append(nn.BatchNorm1d(self.encoder_topology[i+1])) ## add this for better training?
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        ### Define decoder
        self.decoder_topology = [self.latentSize] + self.hidden_layer_encoder_topology[::-1] + [self.inputDim]
        self.decoder_layers = []
        for i in range(len(self.decoder_topology)-1):
            layer = nn.Linear(self.decoder_topology[i],self.decoder_topology[i+1])
            torch.nn.init.xavier_uniform_(layer.weight)  ### weight initialisation
            self.decoder_layers.append(layer)
            self.decoder_layers.append(nn.PReLU())
        self.decoder_layers[-1] = nn.Sigmoid() ### replace activation of final layer with Sigmoid()
        self.decoder = nn.Sequential(*self.decoder_layers)

    def encode(self, x):
        hidden = self.encoder(x)
        return hidden
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def generate_embedding(self,x):
        z = self.encode(x)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    
    
class methVAE(nn.Module):
    def __init__(self,
                inputDim,
                latentSize,
                hidden_layer_encoder_topology=[]):
        super(methVAE, self).__init__()
        self.inputDim = inputDim
        self.hidden_layer_encoder_topology = hidden_layer_encoder_topology
        self.latentSize = latentSize

        ### Define encoder
        self.encoder_topology = [self.inputDim] + self.hidden_layer_encoder_topology + [self.latentSize]
        self.encoder_layers = []
        for i in range(len(self.encoder_topology)-1):
            layer = nn.Linear(self.encoder_topology[i],self.encoder_topology[i+1])
            torch.nn.init.xavier_normal_(layer.weight)  ## weight initialisation
            self.encoder_layers.append(layer)
            self.encoder_layers.append(nn.PReLU())
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        ### Latent space        
        self.fc_mu = nn.Sequential(nn.Linear(self.latentSize, self.latentSize), nn.BatchNorm1d(self.latentSize)) ### added BatchNorm because it seems to improve training
        self.fc_var = nn.Sequential(nn.Linear(self.latentSize, self.latentSize), nn.BatchNorm1d(self.latentSize)) ### added BatchNorm because it seems to improve training

        ### Define decoder
        self.decoder_topology = [self.latentSize] + self.hidden_layer_encoder_topology[::-1] + [self.inputDim]
        self.decoder_layers = []
        for i in range(len(self.decoder_topology)-1):
            layer = nn.Linear(self.decoder_topology[i],self.decoder_topology[i+1])
            torch.nn.init.xavier_uniform_(layer.weight)  ### weight initialisation
            self.decoder_layers.append(layer)
            self.decoder_layers.append(nn.PReLU())
        self.decoder_layers[-1] = nn.Sigmoid() ### replace activation of final layer with Sigmoid()
        self.decoder = nn.Sequential(*self.decoder_layers)
    
    def reparametrization(self, mu, log_var):
        '''Sample latent embeddings, reparameterize by adding noise to embedding.'''
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0),mu.size(1)), device='cuda')
        return mu + sigma*z
    
    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def generate_embedding(self,x):
        mu, log_var = self.encode(x)
        z = self.reparametrization(mu, log_var)
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrization(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var    