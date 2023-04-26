#%%
# region import 
import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader,random_split

from utils import set_seed,get_dl
from model_pca import PCAModel
import model_de as de_model

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
# endregion

exp_dir = "results/two_cycle_estimate/"
model_dir = "results/two_cycle_estimate/two/"

# region 设置路径
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)  # 只能新建一次路径,所以上层文件夹需自行新建

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
# endregion    

#%% data preparation
data_name = ['two_cycle_grid_c2']

dl_train,dl_val,_,_ = get_dl(data_name,mode='single',batch_size=8)

#%%
# region pca setting
with open(exp_dir+'one/'+'pca.pickle','rb') as f:
    pca = pickle.load(f)
# endregion

#%%
de = de_model.DelayEstimationModel(pca,model_dir)

set_seed(0)
de_model.train(de,200,dl_train,dl_val)

