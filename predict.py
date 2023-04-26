#%%
# region import 
import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader,random_split

from utils import get_dl,set_seed

# 从想要的模型中导入所有组件
# from model_pca import PCAModel
import model_cnn as obs_model
from model_fp import FlowPredictionModel,train
# endregion

exp_dir = "results/predict_and_estimate_3/"
model_dir = "results/predict_and_estimate_3/single/"

# region 设置路径
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
# endregion
        
#%% data preparation
data_name = ['multiple_cycle_first','multiple_cycle_2_first']
# data_name = ['multiple_cycle_first']

dl_train,dl_val,_,_ = get_dl(data_name,mode='multiple',batch_size=8)

#%%
# load pca
# with open(exp_dir+'estimate/'+'pca.pickle', 'rb') as f:
#     pca = pickle.load(f)
    
fp = FlowPredictionModel(obs_model.ObsFunc(8),model_dir)

set_seed(0)
train(fp,200,dl_train,dl_val)

