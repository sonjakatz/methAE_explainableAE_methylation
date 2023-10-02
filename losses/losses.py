import torch
from torch import nn

def lossFun_recon_KL_loss(x, 
                  x_hat,  
                  mu, 
                  log_var,
                  recon_loss_function = nn.MSELoss(reduction="sum"),
                  beta=1):
    # reconstruction loss
    recon_loss = recon_loss_function(x_hat, x)
    # KL loss
    kl_loss = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)
    # sum
    loss = recon_loss + beta*kl_loss
    return loss, recon_loss, kl_loss

def lossFun_recon_loss(x, 
                  x_hat,  
                  recon_loss_function = nn.MSELoss(reduction="sum")):
    # reconstruction loss
    loss = recon_loss_function(x_hat, x)
    return loss

