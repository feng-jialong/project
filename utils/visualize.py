#%%
# region import
import os,sys

import numpy as np
import pandas as pd
import importlib as imp
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append("../models/")
import delay_predictor
try:
    imp.reload(delay_predictor)
except:
    pass
from delay_predictor import MyDataset,train_val_test_split,DelayPredictor,test,resume
# endregion 

# 设置实验参数
SAMPLE_SIZE = (4,4)
BATCH_SIZE = 16
LOOKBACK = 2
LOOKAHEAD = 4

data_dir = "../data/training_data/big_clip/"
model_dir = "../results/test-new/standard/"

#%%
dataset = train_val_test_split(MyDataset(data_dir,LOOKBACK,LOOKAHEAD,SAMPLE_SIZE))[2]

#%% 检查数据
def get_data(dataset):

    indexes = np.random.choice(len(dataset),600)
    delay_data = np.zeros(600)
    pbar = tqdm(total=600,desc="Loading")
    for i,index in enumerate(indexes):
        delay_data[i] = dataset[index][1]['delay'][0,0]
        pbar.update(1)
        
    return delay_data

delay_data = get_data(dataset)

#%% 数据分布直方图
plt.hist(delay_data,bins=100)

#%% 检查模型预测效果
def get_data_and_pred():
    model =  DelayPredictor(model_dir,BATCH_SIZE,LOOKBACK,LOOKAHEAD)
    resume(model)
    model.eval()

    # num_samples = 600
    # indexes = np.random.choice(len(dataset),num_samples)
    # delay_data = np.zeros(num_samples)
    # delay_pred = np.zeros(num_samples)
    # pbar = tqdm(total=600,desc="Loading")
    # for i,index in enumerate(indexes):
    
    num_samples = len(dataset)
    delay_data = np.zeros((num_samples,LOOKAHEAD))
    delay_pred = np.zeros((num_samples,LOOKAHEAD))
    pbar = tqdm(total=num_samples,desc="Loading")
    for index in range(num_samples):
        data = dataset[index]
        delay_data[index,:] = data[1]['delay'][0,:LOOKAHEAD]
        delay_pred[index,:] = model.forward(data[0]).detach()[0,:LOOKAHEAD]
        pbar.update(1)
    
    return delay_data, delay_pred

delay_data,delay_pred = get_data_and_pred()

np.savez('test_data_and_pred.npz',delay_data=delay_data,delay_pred=delay_pred)
delay_data,delay_pred = tuple(np.load('val_data_and_pred.npz').values())

#%%
mpl.rcParams['font.family'] = ['Times New Roman','SimSun'] 
mpl.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
mpl.rcParams['font.size'] = 9  # 按磅数设置的
mpl.rcParams['figure.dpi'] = 300
cm = 1/2.54  # centimeters in inches
mpl.rcParams['figure.figsize'] = (12*cm,8*cm)
mpl.rcParams['savefig.dpi'] = 900
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.axis'] = 'both'
mpl.rcParams['axes.grid.which'] = 'both'
mpl.rcParams['axes.facecolor'] = 'white'

save_dir = "../results/figures/predict/"

#%%
train_result = tuple(np.load('train_data_and_pred.npz').values())
val_result = tuple(np.load('val_data_and_pred.npz').values())
test_result = tuple(np.load('test_data_and_pred.npz').values())
result = [train_result,val_result,test_result]
dataset_label = ['训练集','验证集','测试集']

#%% 数据集可视化
for i in range(3):
        plt.hist(result[i][0][:,0],
                 bins=30,range=[0,120],
                 alpha=0.2,density=True,label=dataset_label[i])
plt.legend()
plt.xlabel("车均延误/秒")
plt.xlim([0,100])
plt.ylabel("频数")

plt.savefig(save_dir+'data_distribution.jpg',bbox_inches='tight')

#%% 预测值与真实值散点图
for i in [0,1]:
    for j in range(4):
        if i!=0 or j!=0:
            continue
        plt.scatter(result[i][0][:,j],result[i][1][:,j],
                    s=2.0,alpha=0.5,label=dataset_label[i]+f' {j+1}步向前')
    
plt.plot([0,100],[0,100],c='r',label='y=x')
plt.xlabel("真实延误/秒")
plt.ylabel("预测延误/秒")
plt.xlim([0,100])
plt.ylim([0,100])
plt.legend()

#%% 预测误差分布
for i in [0,1]:
    for j in range(4):
        if j!=1:
            continue
        plt.hist(result[i][1][:,j]-result[i][0][:,j],
                 label=dataset_label[i]+f' {j+1}步向前',
                 bins=100,density=True,range=[-50,50],alpha=0.2)

plt.xlabel("预测误差/秒")
plt.ylabel("频数")
plt.legend()

#%% 真实值分布与预测值分布
for i in [0,1]:
    for j in range(4):
        if i!=0 or j!=0:
            continue
        plt.hist(result[i][1][:,j],bins=30,range=[0,120],alpha=0.5,label='预测')
        plt.hist(result[i][0][:,j],bins=30,range=[0,120],alpha=0.5,label='真实')
plt.legend()
plt.xlabel("延误(sec/veh)")
plt.ylabel("频数")

#%% 预测误差和真实值的关系
for i in [0,1]:
    for j in range(4):
        if i!=0 or j!=0:
            continue
        plt.scatter(result[i][0][:,j],result[i][1][:,j]-result[i][0][:,j],
                    s=0.5,alpha=0.2)
plt.xlabel("真实延误(sec/veh)")
plt.ylabel("预测误差(sec/veh)")
# k*delay_data + h = delay_pred - delay_data, k<0, h>0
# delay_pred = (1+k)*delay_data + h
# 误差大致呈线性？线性回归试试
plt.savefig(save_dir+'error-data.jpg',bbox_inches='tight')

#%%
def get_metrics(result):
    metrics = {}
    metrics['mse'] = ((result[0]-result[1])**2).mean(axis=0)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.abs(result[0]-result[1]).mean(axis=0)
    metrics['mape'] = np.abs((result[0]-result[1])/result[0]).mean(axis=0)
    metrics['wape'] = np.abs(result[0]-result[1]).sum(0)/np.abs(result[0]).sum(0)
    return metrics

train_metrics = get_metrics(train_result)
val_metrics = get_metrics(val_result)
test_metrics = get_metrics(test_result)

#%%  # MAPE的分布
i = 0
plt.hist(np.abs(delay_pred[:,i]-delay_data[:,i])/delay_data[:,i],
            bins=100,range=[0,1],label=str(i)+"-step-ahead",alpha=0.2)

#%% mape-delay scatter
delay_data, delay_pred = result[2]
i = 0
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
scatter = ax.scatter(delay_data[:,i],
                     100*np.abs(delay_pred[:,i]-delay_data[:,i])/delay_data[:,i],
                     alpha=0.5,s=0.5,c='g',zorder=10)
ax.set_xlabel("真实延误/秒")
ax.set_xlim([0,100])

ax.set_ylim([0,100.0])
ax.set_yticks(np.linspace(0,100,11))
ax.set_yticklabels([str(i)+'%' for i in np.linspace(0,100,11)])
ax.set_ylabel("MAPE")

ax_t = plt.twinx()
_,_,hist = ax_t.hist(delay_data[:,i],
                     bins=50,range=[0,100],alpha=0.5,
                     zorder=3)

ax_t.set_ylim([0,400])
ax_t.set_yticks(np.linspace(0,400,21))
ax_t.set_ylabel("频数")
fig.legend([scatter,hist],[str(i+1)+"步向前",'真实延误'],
           bbox_to_anchor=(0.85,0.85))

plt.savefig(save_dir+'.jpg',bbox_inches='tight')

#%% tensorboard数据可视化
tb_data_dir = "../results/tb-data/"
result = {'train':[],'val':[]}
for file in os.listdir(tb_data_dir):
    tag = file.split('_')[-1]
    tag = tag.split('-')[0]
    result[tag].append(pd.read_csv(tb_data_dir+file))

train_loss = np.stack([log.iloc[:98,2].values for log in result['train']],
                      axis=-1)
val_loss = np.stack([log.iloc[:9,2].values for log in result['val']],
                     axis=-1)
train_loss_mean = train_loss.mean(-1)
train_loss_std = train_loss.std(-1)
val_loss_mean = val_loss.mean(-1)
val_loss_std = val_loss.std(-1)

fig,ax = plt.subplots()
ax.plot(np.linspace(10,980,98),train_loss_mean,label='训练集-期望')
ax.fill_between(np.linspace(10,980,98),
                train_loss_mean-3*train_loss_std,
                train_loss_mean+3*train_loss_std,
                alpha=0.2,label='训练集-标准差')
ax.plot(np.linspace(100,900,9),val_loss_mean,label='验证集-期望')
ax.fill_between(np.linspace(100,900,9),
                val_loss_mean-3*val_loss_std,
                val_loss_mean+3*val_loss_std,
                alpha=0.2,label='验证集-标准差')
ax.set_xlim([0,1000])
ax.set_ylabel('损失函数')
ax.set_xlabel('训练步数')
ax.legend()

save_dir = "../results/figures/predict/"

fig.savefig(save_dir+'random_seed_loss_curve.jpg',bbox_inches='tight')

#%% 随机初始化的数据分析
mape = [[0.1203,0.1492,0.1399,0.1468],  # B>D>C
        [0.1170,0.1426,0.1290,0.1409],  
        [0.1193,0.1551,0.1439,0.1590],  # D>B>C
        [0.1102,0.1398,0.1344,0.1435],
        [0.1041,0.1353,0.1303,0.1416],
        [0.1140,0.1409,0.1338,0.1468]] 
# (A,B,C,D), 按理来说应该A<B<C<D,常常有A<C<B,D

mape_mean = np.array(mape).mean(0)
mape_std = np.array(mape).std(0)

mae = [[4.354,5.211,5.051,4.98],  # B>C>D
       [4.271,4.992,4.598,4.785],
       [4.157,5.178,4.953,5.179],
       [3.973,4.923,4.956,4.947],  # C>D>B
       [3.676,4.672,4.581,4.742],  
       [4.286,4.996,4.911,5.057]]  

mae_mean = np.array(mae).mean(0)
mae_std = np.array(mae).std(0)

wape = [[0.1215,0.1473,0.1411,0.1427],
        [0.1186,0.1409,0.1285,0.1372],
        [0.1158,0.1473,0.1387,0.1494],
        [0.1105,0.1390,0.1385,0.1417],
        [0.1023,0.1321,0.1280,0.1361],
        [0.1190,0.1414,0.1374,0.1447]]  #

wape_mean = np.array(wape).mean(0)
wape_std = np.array(wape).std(0)

rmse = [[5.522,6.405,6.446,6.293],  # C>B>D
        [5.377,6.272,6.043,6.109],
        [5.138,6.333,6.282,6.365],
        [4.954,6.202,6.441,6.243],
        [4.677,5.884,5.788,5.925],
        [5.620,6.354,6.486,6.419]]

rmse_mean = np.array(rmse).mean(0)
rmse_std = np.array(rmse).std(0)

mse_mean = ((np.array(rmse))**2).mean(0)
mse_std = ((np.array(rmse))**2).std(0)
