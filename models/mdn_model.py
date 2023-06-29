#%%
# region
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split,DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
# endregion

#%%
class DimNorm(nn.Module):
    def __init__(self,num_features):
        # 沿-1维度归一化
        super(DimNorm,self).__init__()
        self.bn = nn.BatchNorm1d(num_features)  # 沿-2维度归一化的bn
        
    def forward(self,x):
        return self.bn(x.transpose(-2,-1)).transpose(-2,-1)

class LinearResBlock(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,norm_type='2d',is_shortcut=True):
        super(LinearResBlock,self).__init__()
        self.is_shortcut = is_shortcut
        if norm_type=='2d':
            norm = nn.BatchNorm1d(hidden_size)
        elif norm_type=='3d':
            norm = DimNorm(hidden_size)
        self.mainline = nn.Sequential(nn.Linear(input_size,hidden_size),norm,nn.ReLU(),
                                      nn.Linear(hidden_size,output_size),norm,nn.ReLU())
        if self.is_shortcut:
            self.shortcut = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        x = self.mainline(x)
        if self.is_shortcut:
            x +=self.shortcut(x)
        return x
    
class CoefficientLayer(nn.Module):
    def __init__(self):
        super(CoefficientLayer,self).__init__()
    def forward(self,x):
        mode = 'softmax'
        if mode=='softmax':
            return F.softmax(x,dim=-1)
        elif mode=='elu+1':
            x = F.elu(x)+1.0
            return x/x.sum(-1,keepdim=True)
        elif mode=='relu':
            x = F.relu(x)
            return x/(x.sum(-1,keepdim=True)+1e-4)

class SigmaLayer(nn.Module):
    def __init__(self):
        super(SigmaLayer,self).__init__()
    def forward(self,x):
        mode = 'exp'
        if mode=='elu+1':
            return F.elu(x)+1.0
        elif mode=='exp':
            return torch.exp(x)

class MixtureDensityNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_componets):
        super(MixtureDensityNetwork,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_componets = num_componets
        
        self.block_1 = LinearResBlock(self.input_size,self.hidden_size,self.hidden_size)
        self.block_2 = LinearResBlock(self.hidden_size,self.hidden_size,self.hidden_size)
                                        
        self.c_layer = nn.Sequential(nn.Linear(self.hidden_size,self.num_componets),CoefficientLayer())
        
        self.mu_layer = nn.Linear(self.hidden_size,self.num_componets*self.output_size)
        
        self.sigma_layer = nn.Sequential(nn.Linear(self.hidden_size,self.num_componets*self.output_size),SigmaLayer())
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        c = self.c_layer(x)
        mu = self.mu_layer(x).reshape(-1,self.num_componets,self.output_size)
        sigma = self.sigma_layer(x).reshape(-1,self.num_componets,self.output_size)
        return c,mu,sigma
    
    def loss_func(self,y_pred,y):
        c,mu,sigma = y_pred
        # region 各种分布通用但是计算缓慢的做法
        # https://pytorch.org/docs/stable/generated/torch.diag_embed.html
        # https://pytorch.org/docs/stable/generated/torch.diagonal.html#torch.diagonal
        # components = D.MultivariateNormal(mu,torch.diag_embed(sigma))
        # c_log_prob = components.log_prob(y)
        # endregion
        
        # 使用分布表达式直接计算对数概率，快得多
        c_log_prob = - (self.output_size/2)*torch.log(torch.tensor(2*np.pi))[None,None,None] - torch.log(sigma.prod(dim=-1,keepdim=True)) + (-1/2*(y[:,None,:]-mu)**2*sigma**(-2))
        nll = -torch.logsumexp(c_log_prob+torch.log(c+1e-4)[:,:,None],dim=-1).mean()
        return nll
    
    @torch.no_grad()
    def predict(self,x):
        # 不记录梯度的前向传播
        return self.forward(x)
    
# 下游计算任务
@torch.no_grad()
def aleatoric_uncertainty(y_pred):
    # 计算并输出偶然不确定性
    c,_,sigma = y_pred
    # (batch_size,otuput_size)
    return (c[:,:,None]*sigma).sum(1)   

@torch.no_grad()
def epistemic_uncertainty(y_pred):
    # 计算并输出认知不确定性
    c,mu,_ = y_pred
    # (batch_size,otuput_size)
    return (c[:,:,None]*(mu-(c[:,:,None]*mu).sum(-2,keepdim=True))**2).sum(1)

@torch.no_grad()
def mdn_mean(y_pred):
    c,mu,_ = y_pred
    # (batch_size,output_size)
    return (c[:,:,None]*mu).sum(1)

@torch.no_grad()
def mdn_variance(y_pred):
    # (batch_size,output_size)
    return aleatoric_uncertainty(y_pred)+epistemic_uncertainty(y_pred)
