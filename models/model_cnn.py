import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class ObsFunc(nn.Module):
    def __init__(self,hidden_channels):
        super(ObsFunc,self).__init__()
        self.hidden_channels = hidden_channels
        # (600,4,103,4)
        self.network = nn.Sequential(nn.Conv2d(20,hidden_channels,
                                               kernel_size=(7,3),
                                               stride=1,
                                               padding=1),
                                     nn.BatchNorm2d(hidden_channels),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(4,2),
                                                  stride=2,
                                                  padding=0),
                                     nn.Conv2d(hidden_channels,hidden_channels,
                                               kernel_size=(7,3),
                                               stride=1,
                                               padding=1),
                                     nn.BatchNorm2d(hidden_channels),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=(4,2),
                                                  stride=2,
                                                  padding=0),
                                     nn.Flatten())
        
        self.output_size = 21*hidden_channels
        
    def forward(self,x):
        # 输入还原
        sample_num = x.shape[0]
        x_list = torch.chunk(x,5,dim=-1)
        y_list = []
        y_list.append(x_list[0].reshape(sample_num,4,-1,4))
        
        for i in [1,2,3,4]:
            y_list.append(x_list[i].reshape(sample_num,1,-1,4,4).transpose(1,-1).squeeze())
        # (num,4*5,grid_num,lane_num) 4*5个通道
        x_in = torch.cat(y_list,dim=1)
        y = self.network(x_in)
        return y
