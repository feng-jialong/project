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

from mdn_model import MixtureDensityNetwork,LinearResBlock
from mdn_model import mdn_mean,mdn_variance,aleatoric_uncertainty,epistemic_uncertainty

# endregion

#%% model
class DelayPredictUncertainty(nn.Module):
    def __init__(self,model_dir,batch_size,lookback,lookahead):
        super().__init__()
        self.model_dir = model_dir
        self.lookback = lookback
        self.lookahead = lookahead
        self.batch_size = batch_size
        
        hidden_channels = 16
        self.conv = nn.Sequential(nn.Conv2d(20,hidden_channels,kernel_size=(7,3),padding=(3,1)),
                                  nn.BatchNorm2d(hidden_channels),nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(4,2),stride=(4,2),padding=0),
                                  nn.Conv2d(hidden_channels,hidden_channels,kernel_size=(7,3),padding=(3,1)),
                                  nn.BatchNorm2d(hidden_channels),nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(4,2),stride=(4,2),padding=0),
                                  nn.Flatten())
        hidden_size = 16
        D = 2  # 1:单向，2:双向
        self.D = D
        l_enc_num_layers = 2

        self.l_enc_input_layer = LinearResBlock(12*hidden_channels,hidden_size,hidden_size)

        # lower encoder
        # sequential的模型只接受单输入,因此LSTM需单独写出
        self.l_enc_layer = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=l_enc_num_layers,dropout=0.0,bidirectional=bool(D-1))
        self.l_enc_output_layer = LinearResBlock(D*hidden_size,hidden_size,hidden_size)
        
        # size: phase=5 + split=8 + target_func=3*8
        tc_size = 37  # 控制方案维度，不包括可变车道切换相位
        self.tc_output_layer = LinearResBlock(tc_size,hidden_size,hidden_size)
        
        # upper encoder
        self.u_enc_input_layer = LinearResBlock(hidden_size+hidden_size,hidden_size,hidden_size,norm_type='3d')
        self.u_enc_layer = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,dropout=0.0)
        # upper context
        self.u_enc_output_layer = LinearResBlock(hidden_size,hidden_size,hidden_size,norm_type='3d')

        # decoder
        self.dec_input_layer = LinearResBlock(tc_size,hidden_size,hidden_size)
        self.dec_layer = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,dropout=0.0)
        # output delay
        self.dec_output_layer = LinearResBlock(hidden_size,hidden_size,hidden_size,norm_type='3d')
        
        num_components = 16
        # 每一步预测都使用相同的mdn
        self.mdn_layer = MixtureDensityNetwork(hidden_size,hidden_size,1,num_components)
        
        self.optimizer = torch.optim.AdamW(self.parameters(),lr=0.003)
                                           # weight_decay=1e-3)
        
        self.metrics = ['rmse','mae','mape','wape']
    
    def lower_encoding(self,obs):
        lower_context = []
        for i in range(self.lookback):
            seq_lens = []
            for s in obs[:,i]:  # obs[:,i]: (batch,frames!,***)
                seq_lens.append(s.shape[0])
            # 卷积层
            obs_conv = self.conv(torch.cat(list(obs[:,i]),dim=0))  # obs_conv: (batch&frames!,*)
            obs_conv = self.l_enc_input_layer(obs_conv)  # obs_conv: (batch&frames!,*)
            obs_conv = torch.split(obs_conv,seq_lens,dim=0)  # obs_conv: (batch,frames!,*)
            
            # lower encoder
            obs_padded = pad_sequence(obs_conv)  # obs_padded: (frames,batch,*)
            obs_packed = pack_padded_sequence(obs_padded,seq_lens,enforce_sorted=False)
            # _,(l_enc_h,_) = self.l_enc_layer(obs_packed,(None,None))   # 错误的，None会参与计算
            _,(l_enc_h,_) = self.l_enc_layer(obs_packed) # l_enc_h: (D*num_layers,batch,*)
            if self.D == 2:
                lower_context.append(torch.cat([l_enc_h[-2,:,:],l_enc_h[-1,:,:]],
                                               dim=-1))  # append: (batch,2*)
            elif self.D == 1:
                lower_context.append(l_enc_h[-1,:,:])  # append: (batch,*)

        lower_context = torch.stack(lower_context,dim=1)  # lower_context: (batch,lookback,*)
        lower_context = self.l_enc_output_layer(lower_context)  # lower_context: (batch,lookback,*)
        
        return lower_context
    
    def upper_encoding(self,tc_back,lower_context):
        tc_back = self.tc_output_layer(tc_back)  # tc_back: (batch,lookback,*)
        
        u_enc_input = torch.cat([tc_back,lower_context],dim=-1)  # u_enc_input: (batch,lookback,*)
        u_enc_input = self.u_enc_input_layer(u_enc_input).transpose(0,1)  # u_enc_input: (lookback,batch,*)
        _,(u_enc_h,_) = self.u_enc_layer(u_enc_input)  # u_enc_h: (num_layers,batch,*)
        
        upper_context = u_enc_h[[-1],:,:]  # upper_context: (1,batch,*) 保持结构，后续decoder输入需要
        upper_context = self.u_enc_output_layer(upper_context)  # upper_context: (1,batch,*)
        
        return upper_context
    
    def decoding(self,upper_context,tc_ahead):
        tc_ahead = self.dec_input_layer(tc_ahead).transpose(0,1)  # tc_ahead: (lookahead,batch,*)
        dec_output,_ = self.dec_layer(tc_ahead,(upper_context,torch.zeros_like(upper_context)))  # dec_output: (lookahead,batch,*)
        dec_output = self.dec_output_layer(dec_output.transpose(0,1))  # (batch,lookahead,*)
        
        return dec_output
    
    def uncertainty_qualify(self,dec_output):
        # dec_output: (batch,lookahead,hidden_size)
        c,mu,sigma = self.mdn_layer(dec_output)
        # c: (batch,lookahead,1)
        # mu: (batch,lookahead,num_components,1)
        # sigma: (batch,lookahead,num_components,1)
        
        # c: (batch,lookahead)
        # mu: (batch,lookahead,num_components)
        # sigma: (batch,lookahead,num_components)
        return (c[:,:,0],mu[:,:,:,0],sigma[:,:,:,0])
    
    def forward(self,x):
        obs = x['obs']  # obs: (batch,lookback,frames!,***)

        lower_context = self.lower_encoding(obs)
        tc = x['tc']  # tc: (batch,lookback+lookahead,*)
        tc_back = tc[:,:self.lookback,:]  # tc_back: (batch,lookback,*)
        tc_ahead = tc[:,self.lookback:,:]  # tc_ahead: (batch,lookahead,*)
        
        upper_context = self.upper_encoding(tc_back,lower_context)

        dec_output = self.decoding(upper_context,tc_ahead)
        y_pred = self.uncertainty_qualify(dec_output)
        
        return y_pred
        
    def loss_func(self,y_pred,y):
        # y: (batch,lookahead)
        # y_pred: (c,mu,sigma)
        # c: (batch,lookahead)
        # mu: (batch,lookahead,num_components)
        # sigma: (batch,lookahead,num_components)
        loss = self.mdn_layer.loss_func(y_pred,y)
        
        return loss
    
    def metric_func(self,y_pred,y):
        y_pred = mdn_mean(y_pred)  # y_pred: (batch,lookahead)
        
        rmse = torch.sqrt(((y-y_pred)**2).mean(dim=0))
        mae = torch.abs(y-y_pred).mean(dim=0)
        mape = torch.abs((y-y_pred)/y).mean(dim=0)
        wape = torch.abs(y-y_pred).sum(dim=0)/y.sum(dim=0)
        
        return {'rmse':rmse.detach().numpy(),
                'mae':mae.detach().numpy(),
                'mape':mape.detach().numpy(),
                'wape':wape.detach().numpy()}
    
