import numpy as np
import os

import torch
from torch import nn
from torch import optim
from sklearn.metrics import roc_auc_score

from losses.losses import lossFun_recon_loss, lossFun_recon_KL_loss

from torch.utils.tensorboard import SummaryWriter
###################################################################################################
###################################################################################################   
    
class EarlyStopping:
    """c Bjarten - Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.05, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss    
        
        
        
###################################################################################################
###################################################################################################  

def train_NN_clf(logName,
             model, 
             train_loader, 
             val_loader,
             criterion=nn.BCELoss(reduction="sum"),
             n_epochs=100,
             lr=1e-3,
             calcROC=True,
             patienceEarlyStopping=7):
    
    ### Settings for logs
    outPath= f"{logName}/checkpoint"
    os.makedirs(outPath, exist_ok=True)
    writer = SummaryWriter(f"{logName}")
    print(f"\t\tLogging to {outPath}")
    
    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ### initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patienceEarlyStopping, verbose=True, path=f"{outPath}/stateDictModel.pth")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    model.train()
    for epoch in range(1,n_epochs+1):
        ### Training loop
        loss_train = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        ### Validation loop
        loss_val = 0
        roc_val = []
        for data in val_loader:
            model.eval()
            x, y = data[0].to(device), data[1].to(device)
            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y)
            loss_val += loss.item()
            ### Calculate and print AUC ROC score
            if calcROC:
                try: roc_val.append(roc_auc_score(y.detach().numpy(), y_hat.detach().numpy()))
                except: continue

                    
        print(f"Epoch: {epoch}")
        print(f'Training Loss: {round(loss_train, 3)}')
        print(f'Validation Loss: {round(loss_val, 3)}')
        if roc_val:
            print(f'Validation ROC AUC: {round(np.array(roc_val).mean(), 3)}\n')
            
        ### Logging per epoch
        writer.add_scalar('Train - Loss', loss_train, epoch)
        writer.add_scalar('Val - Loss', loss_val, epoch)
        if roc_val:
            writer.add_scalar('Val - ROC-AUC', np.array(roc_val).mean(), epoch)
    
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(loss_val, model) 
        if early_stopping.early_stop:
            print("Early stopping")
            break             
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f"{outPath}/stateDictModel.pth"))
    ### Save final (best) model
    torch.save(model, f"{outPath}/trainedModel.pth")
    ### Delete stateDictModel again to save space; this might be a mistake but we will see
    os.remove(f"{outPath}/stateDictModel.pth")
        
    ### More logging
    # Plot graph of network
    writer.add_graph(model, x)
    writer.close()     

   
###################################################################################################
###################################################################################################          
    
def train_AE(logName,
             model, 
             train_loader, 
             val_loader,
             criterion=nn.MSELoss(reduction="sum"),
             n_epochs=100,
             lr=1e-3,
             patienceEarlyStopping=7):
    
    
    torch.cuda.empty_cache()

    ### Settings for logs
    outPath= f"{logName}/checkpoint"
    os.makedirs(outPath, exist_ok=True)
    writer = SummaryWriter(f"{logName}")
    print(f"\t\tLogging to {outPath}")
    
    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ### initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patienceEarlyStopping, verbose=True, path=f"{outPath}/stateDictModel.pth")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    model.train()
    for epoch in range(1,n_epochs+1):
        ### Training loop
        loss_train = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = lossFun_recon_loss(x, 
                                      x_hat,  
                                      recon_loss_function = criterion)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        ### Validation loop
        loss_val = 0
        for data in val_loader:
            model.eval()
            x, y = data[0].to(device), data[1].to(device)
            x_hat = model(x)
            loss = lossFun_recon_loss(x, 
                                      x_hat,  
                                      recon_loss_function = criterion)
            loss_val += loss.item()
                    
        print(f"Epoch: {epoch}")
        print(f'Training Loss: {round(loss_train, 3)}')
        print(f'Validation Loss: {round(loss_val, 3)}')
            
        ### Logging per epoch
        writer.add_scalar('Train - Loss', loss_train, epoch)
        writer.add_scalar('Val - Loss', loss_val, epoch)
        
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(loss_val, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f"{outPath}/stateDictModel.pth"))
    ### Save final (best) model
    torch.save(model, f"{outPath}/trainedModel.pth")
    ### Delete stateDictModel again to save space; this might be a mistake but we will see
    os.remove(f"{outPath}/stateDictModel.pth")
    
    ### More logging
    # Plot graph of network
    writer.add_graph(model, x)
    writer.close()
    
###################################################################################################
###################################################################################################    
    
def train_AE_l1reg(logName,
             model, 
             train_loader, 
             val_loader,
             criterion=nn.MSELoss(reduction="sum"),
             n_epochs=100,
             lr=1e-3,
             patienceEarlyStopping=7):
    
    ### Settings for logs
    outPath= f"{logName}/checkpoint"
    os.makedirs(outPath, exist_ok=True)
    writer = SummaryWriter(f"{logName}")
    print(f"\t\tLogging to {outPath}")
    
    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ### initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patienceEarlyStopping, verbose=True, path=f"{outPath}/stateDictModel.pth")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    model.train()
    for epoch in range(1,n_epochs+1):
        ### Training loop
        loss_train = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = lossFun_recon_loss(x, 
                                      x_hat,  
                                      recon_loss_function = criterion)
            ### add L1 regularisation loss
            l1_penalty = nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in model.parameters():
                reg_loss += l1_penalty(param, target=torch.zeros_like(param))
            factor_lambda = 0.00005  # value taken from example by medium.com
            loss += reg_loss*factor_lambda
            ###
            
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        ### Validation loop
        loss_val = 0
        for data in val_loader:
            model.eval()
            x, y = data[0].to(device), data[1].to(device)
            x_hat = model(x)
            loss = lossFun_recon_loss(x, 
                                      x_hat,  
                                      recon_loss_function = criterion)
            loss_val += loss.item()
                    
        print(f"Epoch: {epoch}")
        print(f'Training Loss: {round(loss_train, 3)}')
        print(f'Validation Loss: {round(loss_val, 3)}')
            
        ### Logging per epoch
        writer.add_scalar('Train - Loss', loss_train, epoch)
        writer.add_scalar('Val - Loss', loss_val, epoch)
        
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(loss_val, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f"{outPath}/stateDictModel.pth"))
    ### Save final (best) model
    torch.save(model, f"{outPath}/trainedModel.pth")
    
    ### More logging
    # Plot graph of network
    writer.add_graph(model, x)
    writer.close()    
     
###################################################################################################
###################################################################################################

def train_VAE(logName,
             model, 
             train_loader, 
             val_loader,
             criterion=nn.MSELoss(reduction="sum"),
             beta=1,
             n_epochs=100,
             lr=1e-3,
             patienceEarlyStopping=7,
             sleep_earlyStopping=100):
    
    ### Settings for logs
    outPath= f"{logName}/checkpoint"
    os.makedirs(outPath, exist_ok=True)
    writer = SummaryWriter(f"{logName}")
    print(f"\t\tLogging to {outPath}")
    
    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ### initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patienceEarlyStopping, verbose=True, path=f"{outPath}/stateDictModel.pth")  
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    model.train()
    for epoch in range(1,n_epochs+1):
        ### Training loop
        loss_train = 0
        recon_loss_train = 0 
        kl_loss_train = 0
        for batch_idx, data in enumerate(train_loader):
            model.train()
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x)
            loss, recon_loss, kl_loss = lossFun_recon_KL_loss(x, x_hat, mu, log_var, 
                                                              recon_loss_function=criterion, 
                                                              beta=beta)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            recon_loss_train += recon_loss.item()
            kl_loss_train += kl_loss             
        
        ### Validation loop
        loss_val = 0
        recon_loss_val = 0 
        kl_loss_val = 0          
        for data in val_loader:
            model.eval()
            x, y = data[0].to(device), data[1].to(device)
            x_hat, mu, log_var = model(x)
            loss, recon_loss, kl_loss = lossFun_recon_KL_loss(x, x_hat, mu, log_var, 
                                                              recon_loss_function=criterion, 
                                                              beta=beta)
            loss_val += loss.item()
            recon_loss_val += recon_loss.item()
            kl_loss_val += kl_loss 
                    
        print(f"Epoch: {epoch}")
        print(f'Training Loss: {round(loss_train, 3)}')
        print(f'Validation Loss: {round(loss_val, 3)}')
            
        ### Logging per epoch
        writer.add_scalar('Train - Loss', loss_train, epoch)
        writer.add_scalar('Train - Reconstruction Loss', recon_loss_train, epoch)
        writer.add_scalar('Train - KL Loss', kl_loss_train, epoch)   
        writer.add_scalar('Val - Loss', loss_val, epoch)
        writer.add_scalar('Val - Reconstruction Loss', recon_loss_val, epoch)
        writer.add_scalar('Val - KL Loss', kl_loss_val, epoch)   

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if epoch > sleep_earlyStopping:
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f"{outPath}/stateDictModel.pth"))
    ### Save final (best) model
    torch.save(model, f"{outPath}/trainedModel.pth")
    
    ### More logging
    # Plot graph of network
    writer.add_graph(model, x)
    writer.close()
