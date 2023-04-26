#%%
# region import 
import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader,random_split

from utils import set_seed,get_dl
import model_cnn as obs_model
import model_de as de_model

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
# endregion

exp_dir = "results/estimate_test_3/"
model_dir = "results/estimate_test_3/more_data_1/"
# exp_dir = "results/predict_and_estimate/"
# model_dir = "results/predict_and_estimate/estimate/"
# exp_dir = "results/two_cycle_estimate/"
# model_dir = "results/two_cycle_estimate/one/"

# region 设置路径
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)  # 只能新建一次路径,所以上层文件夹需自行新建

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
# endregion    

#%% data preparation
# data_name = ['two_cycle_grid_c2_cycle1']
# data_name = ['single_cycle_grid_c2']
data_name = ['single_cycle_grid_c2','single_cycle_2_grid_c2']

dl_train,dl_val,ds_train,_ = get_dl(data_name,mode='single',batch_size=8)

#%%
# region pca setting
# if 'pca.pickle' in os.listdir(model_dir):
#     with open(model_dir+'pca.pickle','rb') as f:
#         pca = pickle.load(f)
# else:
#     pca = obs_model.ObsFunc(ds_train)
#     pca.info()
#     with open(model_dir+'pca.pickle', 'wb') as f:
#         pickle.dump(pca, f)
# endregion

#%%
de = de_model.DelayEstimationModel(obs_model.ObsFunc(8),model_dir)
# de = de_model.DelayEstimationModel(pca,model_dir)

set_seed(1)
de_model.train(de,400,dl_train,dl_val)

