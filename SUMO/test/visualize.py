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

#%% 导入数据
with open("../../results/simulation_experiment/balance-with-queue/webster.pkl",'rb') as f:
    recorder,predict_result,control_result,upper_context,mpc_model = joblib.load(f)
    
#%% 导入数据
data_dir = "../../results/simulation_experiment/"
with open(data_dir+"dynamic-balance/webster.pkl",'rb') as f:
    webster_res = joblib.load(f)
# with open(data_dir+"dynamic-balance/smooth_webster.pkl",'rb') as f:
#     smooth_webster_res = joblib.load(f)
# with open(data_dir+"dynamic-balance/fix_webster.pkl",'rb') as f:
#     fix_webster_res = joblib.load(f)
with open(data_dir+"dynamic-balance/mpc-all.pkl",'rb') as f:
    mpc_all_res = joblib.load(f)
with open(data_dir+"dynamic-balance/mpc-all-no-grad.pkl",'rb') as f:
    mpc_all_no_grad_res = joblib.load(f)
with open(data_dir+"dynamic-balance/mpc-signal.pkl",'rb') as f:
    mpc_signal_res = joblib.load(f)

#%% 仿真过程可视化——延误
fig,ax = plt.subplots()
ax.plot(webster_res[0].time_point,webster_res[0].delay_list,
        label='Webster')
# ax.plot(smooth_webster_res[0].time_point,smooth_webster_res[0].delay_list,
#         label='Smooth-Webster')
# ax.plot(fix_webster_res[0].time_point,fix_webster_res[0].delay_list,
#         label='Fixed-Webster')
ax.plot(mpc_all_res[0].time_point,mpc_all_res[0].delay_list,
        label='MPC-启用可变车道-代理梯度')
ax.plot(mpc_all_no_grad_res[0].time_point,mpc_all_no_grad_res[0].delay_list,
        label='MPC-启用可变车道-估计梯度')
# ax.plot(mpc_signal_res[0].time_point,mpc_signal_res[0].delay_list,
#         label='MPC-禁用可变车道')
l,u = ax.get_ylim()
ax.fill_between([0.0,1000.0],0.0,1000.0,color='g',alpha=0.2,label='仿真预热')
ax.fill_between([1200.0,2400.0],0.0,1000.0,color='r',alpha=0.2,label='流量突增')
ax.set_xlabel('时间/秒')
ax.set_ylabel('车均延误/秒')
ax.set_xlim([0.0,6000.0])
ax.set_ylim([l,u])
ax.legend()
fig.savefig(save_dir+'delay_plot.jpg',bbox_inches='tight')

#%% 仿真过程可视化——排队
fig,ax = plt.subplots()
ax.plot(webster_res[0].time_point,
        np.array(webster_res[0].queue_list).max(-1),
        label='Webster')
# ax.plot(smooth_webster_res[0].time_point,
#         np.array(smooth_webster_res[0].queue_list).max(-1),
#         label='Smooth-Webster')
# ax.plot(fix_webster_res[0].time_point,
#         np.array(fix_webster_res[0].queue_list).max(-1),
#         label='Fixed-Webster')
ax.plot(mpc_all_res[0].time_point,
        np.array(mpc_all_res[0].queue_list).max(-1),
        label='MPC-启用可变车道-代理梯度')
ax.plot(mpc_all_no_grad_res[0].time_point,
        np.array(mpc_all_no_grad_res[0].queue_list).max(-1),
        label='MPC-启用可变车道-估计梯度')
# ax.plot(mpc_signal_res[0].time_point,
#         np.array(mpc_signal_res[0].queue_list).max(-1),
#         label='MPC-禁用可变车道')
l,u = ax.get_ylim()
ax.fill_between([0.0,1000.0],0.0,1000.0,color='g',alpha=0.2,label='仿真预热')
ax.fill_between([1200.0,2400.0],0.0,1000.0,color='r',alpha=0.2,label='流量突增')
ax.set_xlabel('时间/秒')
ax.set_ylabel('最大排队长度/米')
ax.set_xlim([0.0,6000.0])
ax.set_ylim([l,u])
ax.legend()
fig.savefig(save_dir+'max_queue_plot.jpg',bbox_inches='tight')

#%% 仿真过程可视化——周期
fig,ax = plt.subplots()
ax.plot(webster_res[0].time_point,
        [tc['split'][:4].sum() for tc in webster_res[0].tc_list],
        label='Webster')
# ax.plot(smooth_webster_res[0].time_point,
#         [tc['split'][:4].sum() for tc in smooth_webster_res[0].tc_list],
#         label='Smooth-Webster')
# ax.plot(fix_webster_res[0].time_point,
#         [tc['split'][:4].sum() for tc in fix_webster_res[0].tc_list],
#         label='Fixed-Webster')
ax.plot(mpc_all_res[0].time_point,
        [tc['split'][:4].sum() for tc in mpc_all_res[0].tc_list],
        label='MPC-启用可变车道-代理梯度')
ax.plot(mpc_all_no_grad_res[0].time_point,
        [tc['split'][:4].sum() for tc in mpc_all_no_grad_res[0].tc_list],
        label='MPC-启用可变车道-估计梯度')
# ax.plot(mpc_signal_res[0].time_point,
#         [tc['split'][:4].sum() for tc in mpc_signal_res[0].tc_list],
#         label='MPC-禁用可变车道')
l,u = ax.get_ylim()
ax.fill_between([0.0,1000.0],0.0,1000.0,color='g',alpha=0.2,label='仿真预热')
ax.fill_between([1200.0,2400.0],0.0,1000.0,color='r',alpha=0.2,label='流量突增')
ax.set_xlabel('时间/秒')
ax.set_ylabel('周期长度/秒')
ax.set_xlim([0.0,6000.0])
ax.set_ylim([l,u])
ax.legend()
fig.savefig(save_dir+'cycle_plot.jpg',bbox_inches='tight')

#%% 仿真过程可视化——绿灯时间
res = mpc_all_no_grad_res
fig,ax = plt.subplots()
for i in range(4):
        ax.plot(res[0].time_point,
                [tc['split'][i] for tc in res[0].tc_list],
                label='相位'+str(i+1))
l,u = ax.get_ylim()
ax.fill_between([0.0,1000.0],0.0,1000.0,color='g',alpha=0.2,label='仿真预热')
ax.fill_between([1200.0,2400.0],0.0,1000.0,color='r',alpha=0.2,label='流量突增')
ax.set_xlabel('时间/秒')
ax.set_ylabel('绿灯时间/秒')
ax.set_xlim([0.0,6000.0])
ax.set_ylim([l,u])
# ax.set_ylim([10,25])
ax.legend()

#%% 仿真过程可视化——延误及预测
res = [mpc_all_res,mpc_all_no_grad_res]
fig,axes = plt.subplots(4,2,sharex=True,figsize=(14*cm,12*cm))

for i in range(2):
        for j in range(4):
                ax = axes[j,i]
                ax.plot(res[i][0].time_point,
                        res[i][0].delay_list,
                        label='真实延误/秒')
                ax.plot(res[i][0].time_point[2+j:],
                        torch.stack(res[i][1],dim=1)[j,:-(1+j)],
                        label='预测延误/秒')
                ax.set_xlim([0.0,6000.0])
                l,u = ax.get_ylim()
                ax.fill_between([0.0,1000.0],l,u,color='g',alpha=0.2,label='仿真预热')
                # ax.fill_between([1200.0,2400.0],l,u,color='r',alpha=0.2)
                if i==0:
                        ax.set_ylabel(str(j+1)+'步向前')

axes[-1,0].legend(loc='lower center', bbox_to_anchor=(1.1,-0.9),ncol=3)
axes[0,0].set_xlabel('MPC-启用可变车道-代理梯度')
axes[0,0].xaxis.set_label_coords(0.5,1.2)
axes[0,1].set_xlabel('MPC-禁用可变车道-估计梯度')
axes[0,1].xaxis.set_label_coords(0.5,1.2)
axes[-1,0].set_xlabel('时间/秒')
axes[-1,1].set_xlabel('时间/秒')

fig.savefig(save_dir+'delay_pred_plot.jpg',bbox_inches='tight')

#%% 代理模型可视化-1D
res = mpc_all_res
fig,axes = plt.subplots(2,2)
axes = axes.reshape(-1)
tc = res[2][-1]  # 仿真过程中任取一个周期的控制
points = np.linspace(10.0,70.0,100)
f_list = np.zeros(100)
for k in range(4):
    for i,point in enumerate(points):
        tc_p = tc.copy()
        tc_p[:,0,5+k] = point
        tc_p = torch.from_numpy(tc_p)
        with torch.no_grad():
            f_list[i] = res[4].decoding(res[3],tc_p).sum().item()
    axes[k].plot(points,f_list,label='代理延误')

    axes[k].set_xticks(np.linspace(10.0,70.0,5))
    axes[k].set_xlim([10.0,70.0])
    axes[k].set_xlabel('相位'+str(k+1)+'绿灯时间/秒')
    
    l,u = axes[k].get_ylim()
    axes[k].fill_between([15.0,60.0],0.0,1000.0,color='green',alpha=0.2,label='绿灯时间范围')
    axes[k].plot([tc[0,0,5+k],tc[0,0,5+k]],[0.0,1000.0],color='red',label='最优绿灯时间')
    axes[k].set_ylim([l,u])

fig.tight_layout()  # 先后顺序有讲究
axes[2].legend(loc='lower center', bbox_to_anchor=(1.1,-0.7),ncol=3)  # 先后顺序有讲究
fig.supylabel('前向时间窗延误/秒',fontsize=9)
fig.savefig(save_dir+'surrogate_1D_plot.jpg',bbox_inches='tight')  # 保存的和看到的不一样？

#%% 代理模型可视化-2D
res = mpc_all_res
tc = res[2][-1]
points = np.linspace(10.0,70.0,100)
f_list = np.zeros((100,100))
for i,point_x in enumerate(points):
    for j,point_y in enumerate(points):
        tc_p = tc.copy()
        tc_p[:,0,7] = point_x
        tc_p[:,0,8] = point_y
        tc_p = torch.from_numpy(tc_p)
        with torch.no_grad():
            f_list[i,j] = res[4].decoding(res[3],tc_p).sum().item()

# region plot surface
# fig = plt.figure()
# ax3 = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(points, points)
# Z = f_list

# ax3.plot_surface(X,Y,Z,cmap='rainbow',zorder=2,linewidth=0.0)
# ax3.contour(points,points,Z,extend3d=False,zorder=4,
#             linewidths=1.0,levels=20,colors='black')
# endregion

# region plot contour
fig,ax = plt.subplots()
X, Y = np.meshgrid(points, points)
Z = f_list
contour = ax.contourf(points,points,Z,levels=30,cmap=plt.cm.jet,alpha=0.8)
ax.text(75/2,60.0+1.0,'可行域',horizontalalignment='center')
ax.plot([15.0,15.0],[60.0,15.0],linestyle='--',color='black')
ax.plot([60.0,15.0],[60.0,60.0],linestyle='--',color='black')
ax.plot([60.0,60.0],[15.0,60.0],linestyle='--',color='black')
ax.plot([15.0,60.0],[15.0,15.0],linestyle='--',color='black')
ax.text(tc[0,0,7]+2.0,tc[0,0,8]+2.0,'最优解',color='red')
ax.scatter(tc[0,0,7],tc[0,0,8],color='red',s=30.0,marker='+',zorder=4)

ax.set_xlabel('相位a绿灯时间/秒')
ax.set_ylabel('相位b绿灯时间/秒')
ax.set_yticks(np.linspace(10.0,70.0,13))
ax.set_xticks(np.linspace(10.0,70.0,13))
ax.grid(visible=False)
ax.set_aspect('equal')

cbar=fig.colorbar(contour, shrink=0.8, aspect=8,alpha=0.8,label="延误/秒")
# endregion

fig.savefig(save_dir+'surrogate_2D_contour.jpg',bbox_inches='tight')

#%% 仿真数值结果
webster_delay = np.array(webster_res[0].delay_list).mean()
smooth_webster_delay = np.array(smooth_webster_res[0].delay_list).mean()
fix_webster_delay = np.array(fix_webster_res[0].delay_list).mean()
mpc_all_delay = np.array(mpc_all_res[0].delay_list).mean()
mpc_signal_delay = np.array(mpc_signal_res[0].delay_list).mean()

webster_delay = np.array(webster_res[0].delay_list).std()
smooth_webster_delay = np.array(smooth_webster_res[0].delay_list).std()
fix_webster_delay = np.array(fix_webster_res[0].delay_list).std()
mpc_all_delay = np.array(mpc_all_res[0].delay_list).std()
mpc_signal_delay = np.array(mpc_signal_res[0].delay_list).std()

# webster_queue = np.array(webster_res[0].queue_list).mean()
# smooth_webster_queue = np.array(smooth_webster_res[0].queue_list).mean()
# fix_webster_queue = np.array(fix_webster_res[0].queue_list).mean()
# mpc_all_queue = np.array(mpc_all_res[0].queue_list).mean()
# mpc_signal_queue = np.array(mpc_signal_res[0].queue_list).mean()

print(webster_delay)
print(smooth_webster_delay)
print(fix_webster_delay)
print(mpc_all_delay)
print(mpc_signal_delay)
# print(webster_queue)
# print(smooth_webster_queue)
# print(fix_webster_queue)
# print(mpc_all_queue)
# print(mpc_signal_queue)
