#%%
# region import
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader,random_split

from utils import MultipleData,get_obs,get_sar,set_seed
from model_pca import PCAModel
import model_de as de_model
import model_fp as fp_model
# endregion

#%%
data_name = 'multiple_cycle_grid_first'

# region dataloaders
set_seed(0)  # 保证数据集相同，更好的做法是划分后将数据集保存

ds = MultipleData(data_name)
train_size = int(0.6*len(ds))
val_size = len(ds)-int(0.6*len(ds))
ds_train,ds_val = random_split(ds,[train_size,val_size])

batch_size = 8
dl_train_obs = DataLoader(ds_train,batch_size=batch_size,collate_fn=get_obs,
                          shuffle=True,drop_last=True)
dl_train_sar = DataLoader(ds_train,batch_size=batch_size,collate_fn=get_sar,
                          shuffle=True,drop_last=True)
dl_val_obs = DataLoader(ds_val,batch_size=4,collate_fn=get_obs,
                        shuffle=False,drop_last=True)
dl_val_sar = DataLoader(ds_val,batch_size=4,collate_fn=get_sar,
                        shuffle=False,drop_last=True)
# endregion

#%% 联合延误估计
exp_dir = "results/predict_and_estimate/"
predict_dir = "results/predict_and_estimate/predict/"
estimate_dir = "results/predict_and_estimate/estimate/"

# load pca
with open(exp_dir+'pca.pickle', 'rb') as f:
    pca = pickle.load(f)
    
# load de
de = de_model.DelayEstimationModel(pca,estimate_dir)
de_model.resume(de)
de.eval()

# load fp
fp = fp_model.FlowPredictionModel(pca,predict_dir)
fp_model.resume(fp)
fp.eval()

def delay_one_step():
    # 预测单个周期的延误
    pass


def delay_multi_step(obs,ts,horizon=3):
    # 输入一条观测和控制序列，预测多个周期的延误
    seq_len = obs.shape[0]
    results = np.zeros((seq_len+horizon-1,horizon))
    for c in range(seq_len):
        for s in range(horizon):
            delay = de_model(obs[c],ts[c])
            results[s,s+c] = delay
            