import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

from utils import get_obs

class ObsFunc(nn.Module):
    def __init__(self,ds,EVR=0.95):
        super(ObsFunc,self).__init__()
        self.EVR = EVR
        train_data = get_obs([ds[i] for i in range(len(ds))]).numpy()
        
        self.model = PCA(n_components=self.EVR,whiten=False).fit(train_data)
        self.output_size = self.model.n_components_
        
    def forward(self,x):
        # 前向传播函数，用于pytorch
        return torch.from_numpy(self.model.transform(x.numpy())).to(torch.float32)
    
    def info(self):
        from matplotlib import pyplot as plt
        evr = self.model.explained_variance_ratio_
        _,(ax1,ax2) = plt.subplots(2,1)
        ax1.plot(evr)
        ax2.plot(evr.cumsum())
        
        n_f = self.model.n_features_
        n_c = self.model.n_components_
        print('number of features:',n_f)
        print('number of components:',n_c)