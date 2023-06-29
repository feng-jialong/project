#%%
# region import 
import sys,os
import numpy as np
import importlib as imp
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append("../models/")
import delay_predictor_uncertainty
try:
    imp.reload(delay_predictor_uncertainty)
except:
    pass
from delay_predictor_uncertainty import MyDataset,get_dataloader,DelayPredictUncertainty
from delay_predictor_uncertainty import train,test,resume

from utils import params_count,set_seed
# endregion

#%% 
# region 设置路径
data_dir = "../data/training_data/standard/"  # 训练数据的路径
base_dir = "../results/"  # 基本路径
exp_name = 'test'  # 实验名称
model_name = 'baseline'  # 模型名称
exp_dir = base_dir +exp_name+"/"  # 实验路径
model_dir = exp_dir + model_name +"/"  # 模型路径

if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
# endregion

# 设置实验参数
SAMPLE_SIZE = (8,4)
BATCH_SIZE = 64
LOOKBACK = 8
LOOKAHEAD = 4
SEED = 0

#%% 训练模型
if __name__ == "__main__":
    # Windows上多进程的实现问题。在Windows上，子进程会自动import启动它的文件，而在import的时候会执行这些语句，就会无限递归创建子进程报错。
    
    # 设置模型及训练随机数种子
    set_seed(SEED)
    
    dataset = MyDataset(data_dir,LOOKBACK,LOOKAHEAD,SAMPLE_SIZE,SEED)
    print(len(dataset))
    train_dl,val_dl,test_dl = get_dataloader(dataset,BATCH_SIZE,SEED)
    
    model = DelayPredictUncertainty(model_dir,BATCH_SIZE,LOOKBACK,LOOKAHEAD)
    train(model,4,train_dl,val_dl)
    
    print("finished !")