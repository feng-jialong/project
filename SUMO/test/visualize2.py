#%%
# region import
import os,sys
import numpy as np
import joblib
import torch

import matplotlib as mpl
import matplotlib as plt
import matplotlib.pyplot as plt

# mpl.rcParams['font.sans-serif']='SimHei'
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

from runner import Recorder
sys.path.append("../../models/")
from delay_predictor import DelayPredictor

save_dir = "../../results/figures/control/"
# endregion

#%%
with open("../../results/simulation_experiment/methods/bfgs_no_grad.pkl",'rb') as f:
    bfgs_no_grad_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/bfgs_no_grad_2.pkl",'rb') as f:
    bfgs_no_grad_2_res = joblib.load(f)
    
with open("../../results/simulation_experiment/methods/bfgs_with_grad.pkl",'rb') as f:
    bfgs_with_grad_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/bfgs_with_grad_2.pkl",'rb') as f:
    bfgs_with_grad_2_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/bfgs_with_grad_unlimited.pkl",'rb') as f:
    bfgs_with_grad_unlimited_res = joblib.load(f)
    
with open("../../results/simulation_experiment/methods/nelder_mead.pkl",'rb') as f:
    nelder_mead_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/nelder_mead_2.pkl",'rb') as f:
    nelder_mead_2_res = joblib.load(f)
    
# with open("../../results/simulation_experiment/methods/diff_evol.pkl",'rb') as f:
#     diff_evol_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/diff_evol_2.pkl",'rb') as f:
    diff_evol_2_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/diff_evol_3.pkl",'rb') as f:
    diff_evol_3_res = joblib.load(f)
with open("../../results/simulation_experiment/methods/diff_evol_4.pkl",'rb') as f:
    diff_evol_4_res = joblib.load(f)

#%%  控制效果对比
fig,ax = plt.subplots()
ax.plot(bfgs_no_grad_res[0].time_point,bfgs_no_grad_res[0].delay_list,
        label='BFGS-估计梯度')
# ax.plot(bfgs_no_grad_2_res[0].time_point,bfgs_no_grad_2_res[0].delay_list,
#         label='BFGS-估计梯度2')
# ax.plot(bfgs_with_grad_res[0].time_point,bfgs_with_grad_res[0].delay_list,
#         label='BFGS-代理梯度')
# ax.plot(bfgs_with_grad_2_res[0].time_point,bfgs_with_grad_2_res[0].delay_list,
#         label='BFGS-代理梯度2')
ax.plot(bfgs_with_grad_unlimited_res[0].time_point,bfgs_with_grad_unlimited_res[0].delay_list,
        label='BFGS-代理梯度-收敛')

ax.plot(diff_evol_2_res[0].time_point,diff_evol_2_res[0].delay_list,
        label='差分进化2')
# ax.plot(diff_evol_3_res[0].time_point,diff_evol_3_res[0].delay_list,
#         label='差分进化3')
# ax.plot(diff_evol_4_res[0].time_point,diff_evol_4_res[0].delay_list,
#         label='差分进化4')

ax.plot(nelder_mead_res[0].time_point,nelder_mead_res[0].delay_list,
        label='Nelder-Mead')
# ax.plot(nelder_mead_2_res[0].time_point,nelder_mead_2_res[0].delay_list,
#         label='Nelder-Mead2')

l,u = ax.get_ylim()
ax.fill_between([0.0,600.0],0.0,1000.0,color='g',alpha=0.2,label='仿真预热')
ax.set_xlabel('仿真时间/秒')
ax.set_ylabel('周期车均延误/秒')
ax.set_xlim([0.0,6000.0])
ax.set_ylim([l,u])
ax.legend(loc='center right', bbox_to_anchor=(1.5,0.5))
fig.savefig(save_dir+'methods_delay_plot.jpg',bbox_inches='tight')

#%% 控制效果对比——出图
fig,ax = plt.subplots()
ax.plot(bfgs_no_grad_res[0].time_point,bfgs_no_grad_res[0].delay_list,
        label='BFGS-估计梯度')

ax.plot(bfgs_with_grad_unlimited_res[0].time_point,bfgs_with_grad_unlimited_res[0].delay_list,
        label='BFGS-代理梯度')

ax.plot(diff_evol_2_res[0].time_point,diff_evol_2_res[0].delay_list,
        label='差分进化')

ax.plot(nelder_mead_res[0].time_point,nelder_mead_res[0].delay_list,
        label='单纯形搜索')

l,u = ax.get_ylim()
ax.fill_between([0.0,600.0],0.0,1000.0,color='g',alpha=0.2,label='仿真预热')
ax.set_xlabel('仿真时间/秒')
ax.set_ylabel('周期车均延误/秒')
ax.set_xlim([0.0,6000.0])
ax.set_ylim([l,u])
ax.legend(loc='center right', bbox_to_anchor=(1.4,0.5))
fig.savefig(save_dir+'methods_delay_plot.jpg',bbox_inches='tight')

#%% 控制效果对比——出数
print(np.array(bfgs_no_grad_res[0].delay_list).mean())
print(np.array(bfgs_with_grad_unlimited_res[0].delay_list).mean())
print(np.array(diff_evol_2_res[0].delay_list).mean())
print(np.array(nelder_mead_res[0].delay_list).mean())
