import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from sklearn import preprocessing

class prepareDataLoader_fromPkl(Dataset):
    def __init__(self,file, colName):
        
        with open(file, "rb") as f: dic = pickle.load(f)

        x = dic['beta'].values
        y = dic['pheno'][colName].values
        # # Encode y labels
        # le = preprocessing.LabelEncoder()
        # y = le.fit_transform(y)
        
        self.x_tensor=torch.tensor(x,dtype=torch.float32)
        self.y_tensor=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_tensor)
    
    def __getitem__(self,idx):
        return self.x_tensor[idx], self.y_tensor[idx]
    
    def returnTensor_(self):
        return self.x_tensor, self.y_tensor
    
    
class prepareDataLoader_fromTensor(Dataset):
    def __init__(self,
                 x_tensor,
                 y_tensor):

        self.x_tensor=x_tensor
        self.y_tensor=y_tensor

    def __len__(self):
        return len(self.y_tensor)
    
    def __getitem__(self,idx):
        return self.x_tensor[idx], self.y_tensor[idx]
    
    def returnTensor_(self):
        return self.x_tensor, self.y_tensor