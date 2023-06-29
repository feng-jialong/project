#%%
# region import
import sys,os
import importlib as imp
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  
import torch
from torch import nn
import torch.nn.functional as F
import joblib
from joblib import Parallel,delayed
from scipy.optimize import minimize,differential_evolution,dual_annealing

# 导入与重载自定义模块
sys.path.append("../../utils/")
sys.path.append("../../models/")
import process
import delay_predictor
try:
    imp.reload(process)
except:
    pass
try:
    imp.reload(delay_predictor)
except:
    pass
import utils
try:
    imp.reload(utils)
except:
    pass
from process import frame_process,tc_process
from delay_predictor import DelayPredictor,resume

from utils import control2predict
# endregion

#%% 
class MPCController:
    def __init__(self,model,mode):
        batch_size = 16
        self.lookahead = 4
        self.lookback = 2
        self.warmup = self.lookback
        
        self.g_min = 15.0
        self.g_max = 60.0
        self.C_min = 60.0
        self.C_max = 18.0
        
        # 流向延误的指数平滑
        self.delay_m_es = np.zeros(12)
        # 延误预测模型
        self.model = model
        
        self.scheme = None
        self.mode = mode
        
        # 记录用于预测的时窗数据
        self.obs_lookback = np.empty((1,self.lookback),dtype=object)
        self.tc_lookback = torch.zeros((1,self.lookback,37))
        
        self.par_gd = Parallel(n_jobs=8)
        
        self.predict_result = []
        self.control_result = []
        self.nit_result = []
        self.nfev_result = []

    # 整数变量直接穷举
    def brute_force(self,schemes_and_switches):
        # 枚举整数变量的方案, 包括多步lookahead
        # 毕设阶段：只给出一个方案
        discrete_controls = []
        scheme2func = [[2,2],[2,1],[2,0],[1,0]]
        for scheme,switch in schemes_and_switches:  # 遍历每个方案
            discrete_control = {}
            target_func = []
            for s in scheme:
                target_func.extend(scheme2func[s])
            discrete_control['target_func'] = np.tile(np.array(target_func,dtype=int),
                                                      reps=(self.lookahead,1))

            discrete_control['switch'] = np.zeros((self.lookahead,8),dtype=int)
            discrete_control['switch'][0] = switch
            
            discrete_control['phase'] = np.zeros((self.lookahead,5),dtype=int)
            discrete_control['phase'][:,0] = 0
            
            # 有车道切换就对向单口放行，无车道切换就对称放行
            # 南北方向单口放行: 车道切换或共用车道
            if (discrete_control['target_func'][0,[0,1,4,5]] == 1).any() or \
                discrete_control['switch'][0,[0,1,4,5]].any():
                discrete_control['phase'][:,2] = 0
                discrete_control['phase'][:,4] = 1
            # 南北向对称放行
            else:
                discrete_control['phase'][:,2] = 0
                discrete_control['phase'][:,4] = 0
                
            # 东西向单口放行: 车道切换或共用车道
            if (discrete_control['target_func'][0,[2,3,6,7]] == 1).any() or \
                discrete_control['switch'][0,[2,3,6,7]].any():
                discrete_control['phase'][:,1] = 0
                discrete_control['phase'][:,3] = 1
            # 东西向对称放行
            else:
                discrete_control['phase'][:,1] = 0
                discrete_control['phase'][:,3] = 0
            
            discrete_controls.append(discrete_control)
        
        return discrete_controls

    def wesbter_split(self):
        pass
    
    # 多次随机启动
    def multistart(self,discrete_control,scheme,lane_func,vph_m,times=8):
        # webster方案周围，多次随机启动
        # Webster绿灯时间, 使用等饱和度原则
        # 绿信比
        split = np.zeros(8)
        movement2number = [[7,4],[1,6],[3,8],[5,2]]  # 流向映射到编号
        for i in range(4):  # 假设每条车道SFR相同
            if scheme[i] == 1:  # 共用车道
                r = vph_m[i,1]/vph_m[i,0]
                split[movement2number[i][0]-1] = vph_m[i,0]/(2/(1+r)+1)
                split[movement2number[i][1]-1] = vph_m[i,1]/(2-2/(1+r))
            elif scheme[i] == 3:  # 共用车道
                r = vph_m[i,0]/vph_m[i,1]
                split[movement2number[i][0]-1] = vph_m[i,0]/(2-2/(1+r)+1)
                split[movement2number[i][1]-1] = vph_m[i,1]/(2/(1+r))
            else:  # 无共用车道
                split[movement2number[i][0]-1] = vph_m[i,0]/len(lane_func[i][0])
                split[movement2number[i][1]-1] = vph_m[i,1]/len(lane_func[i][1])
        splits = np.tile(split,reps=(times,self.lookahead,1))* \
            np.random.uniform(0.75,1.15,(times,self.lookahead,8))
        # 相位对齐
        phase_group = np.zeros((self.lookahead,2,4),dtype=int)  # 便于使用的相位方案
        for i in range(times):
            for j in range(self.lookahead):
                if discrete_control['phase'][j,3] == 1:
                    splits[i,j,0] = max(splits[i,j,0],splits[i,j,5])
                    splits[i,j,5] = splits[i,j,0]
                    splits[i,j,1] = max(splits[i,j,1],splits[i,j,4])
                    splits[i,j,4] = splits[i,j,1]
                    phase_group[j,:,0] = [0,5]
                    phase_group[j,:,1] = [1,4]
                else:
                    splits[i,j,0] = max(splits[i,j,0],splits[i,j,4])
                    splits[i,j,4] = splits[i,j,0]
                    splits[i,j,1] = max(splits[i,j,1],splits[i,j,5])
                    splits[i,j,5] = splits[i,j,1]
                    phase_group[j,:,0] = [0,4]
                    phase_group[j,:,1] = [1,5]
                
                if discrete_control['phase'][j,4] == 1:
                    splits[i,j,2] = max(splits[i,j,2],splits[i,j,7])
                    splits[i,j,7] = splits[i,j,2]
                    splits[i,j,3] = max(splits[i,j,3],splits[i,j,6])
                    splits[i,j,6] = splits[i,j,3]
                    phase_group[j,:,2] = [2,7]
                    phase_group[j,:,3] = [3,6]
                else:
                    splits[i,j,2] = max(splits[i,j,2],splits[i,j,6])
                    splits[i,j,6] = splits[i,j,2]
                    splits[i,j,3] = max(splits[i,j,3],splits[i,j,7])
                    splits[i,j,7] = splits[i,j,3]
                    phase_group[j,:,2] = [2,6]
                    phase_group[j,:,3] = [3,7]
        
        splits = splits/splits.sum(axis=-1,keepdims=True)*2
        
        splits *= np.tile(np.linspace(1.1*self.C_min,
                                      0.9*self.C_max,
                                      times)[:,None,None],
                          reps=(1,self.lookahead,8))# 限制范围内随机选取周期
        splits = splits.clip(self.g_min,self.g_max)  # 绿灯时间范围
        
        return splits,phase_group
    
    # 优化问题求解
    def optimize(self,splits,discrete_control,phase_group):
        # splits: array(batch,lookahead,8) -> (lookahead,8)
        
        # splits = torch.tensor(splits).to(torch.float32)
        
        phase = discrete_control['phase']  # (lookahead,8)
        phase = torch.from_numpy(phase)[None,:,:].to(torch.float32)
        # (1,lookahead,5)
        
        target_func = discrete_control['target_func']  # (lookahead,8)
        one_hot = OneHotEncoder(categories=8*[[0,1,2]],sparse_output=False)
        target_func = one_hot.fit_transform(target_func)
        target_func = torch.from_numpy(target_func)[None,:,:].to(torch.float32)
        # (1,lookahead,8)
        
        optimal_control_p = None
        optimal_value = 1000.0
        
        # region scipy.optimize
        def objective_func(split):
            return self._objective_func(split,phase,phase_group,target_func)[0]
        
        def objective_func_with_grad(split):
            return self._objective_func(split,phase,phase_group,target_func)
        
        for i in range(len(splits)):
            split = splits[i]
            split = split[:,:4].reshape(-1)
            res = minimize(fun=objective_func,x0=split,
                        #    method='Nelder-Mead',
                        #    method='BFGS',jac=True,
                           method='BFGS',jac=None,
                           options={'disp':True,'maxiter':1000})
            
            # res = differential_evolution(func=objective_func,
            #                              bounds=4*self.lookahead*[(15.0,60.0)],
            #                              x0=split,
            #                              popsize=4,maxiter=100)
            # i = len(splits)
            
            # res = dual_annealing(func=objective_func,
            #                      bounds=4*self.lookahead*[(15.0,60.0)],
            #                      x0=split,
            #                      maxiter=40)
            # i = len(splits)
            if res.fun < optimal_value:
                optimal_value = res.fun
                
                x = np.zeros((self.lookahead,8))
                x[:,:4] = res.x.reshape(self.lookahead,4)
                x[:,phase_group[:,1,:]] = res.x.reshape(self.lookahead,4)
                split = torch.tensor(x).to(torch.float32)
                tc = torch.cat([phase,split[None,:,:],target_func],dim=-1)
                
                optimal_control_p = tc.detach().numpy()
        # endregion
        
        # 最优控制保存：预测格式保留lookahead周期，控制格式保留当前(第一个)周期
        optimal_control_c = {}
        optimal_control_c['phase'] = discrete_control['phase'][0,:]
        optimal_control_c['target_func'] = discrete_control['target_func'][0,:]
        optimal_control_c['switch'] = discrete_control['switch'][0,:]
        optimal_control_c['split'] = optimal_control_p[0,0,5:13]
        
        return optimal_control_p,optimal_control_c,optimal_value

    def _objective_func(self,split,phase,phase_group,target_func):
        M = 1.0
        self.model.eval()
        # 还原
        x = np.zeros((self.lookahead,8))
        x[:,:4] = split.reshape(self.lookahead,4)
        x[:,phase_group[:,1,:]] = split.reshape(self.lookahead,4)
        
        split = torch.tensor(x).to(torch.float32)
        tc = torch.cat([phase,split[None,:,:],target_func],dim=-1)
        tc.requires_grad = True
        tc.retain_grad = True
        
        delay_output = self.model.decoding(self.upper_context,tc)
        gamma = 1.0
        f = delay_output[0,0] + gamma*delay_output[0,1] + \
            gamma**2*delay_output[0,2] + gamma**3*delay_output[0,3]
        # 惩罚项
        f += M*(F.softplus(15.0-tc[:,:,5:13]).sum() + \
            F.softplus(tc[:,:,5:13]-60.0).sum() +\
                F.softplus(60.0-tc[:,:,5:9].sum(dim=-1)).sum() + \
                    F.softplus(tc[:,:,5:9].sum(dim=-1)-180.0).sum())
        value = f.item()
        f.backward()
    
        # 相位组绿灯时间相等，梯度相加
        for j in range(self.lookahead):
            tc.grad[:,j,5:9] += tc.grad[:,j,phase_group[j,1,:]+5]
        # 因为罚函数的存在，进行梯度裁剪
        nn.utils.clip_grad_norm_(tc.grad[:,:,5:9],1.0)
        derivative = tc.grad[:,:,5:9].detach().numpy().reshape(-1)  # (*,*,4)
        
        return value,derivative
    
    def update(self,monitor,recorder,tc):
        # 一个周期更新一次
        # obs时窗数据从recorder拿
        # (frames:list,info) -> tenor (frames,C,H,W)
        obs = frame_process(None,recorder.obs_list[-1])
        # 索引-2，-1是初始化的空列表
        
        self.obs_lookback[0,:-1] = self.obs_lookback[0,1:]
        self.obs_lookback[0,-1] = obs
        
        # tc时窗数据从traffic_controller拿
        control = control2predict(tc.control)
        self.tc_lookback[0,:-1,:] = self.tc_lookback[0,1:,:]
        self.tc_lookback[0,[-1],:] = control
        
        if self.warmup > 0:
            self.warmup -= 1
        
        if self.warmup == 0:
            with torch.no_grad():
                self.model.eval()
                lower_context = self.model.lower_encoding(self.obs_lookback)
                self.upper_context = self.model.upper_encoding(self.tc_lookback,
                                                            lower_context)
                # delay_m = self.model.output(self.upper_context).numpy()[0]
        
        beta = 0.6  # 指数平滑中，当前值的权重
        self.delay_m_es = beta*monitor.output()[1] + (1-beta)*self.delay_m_es
    
    # 检查：流向平衡
    def flow_balance(self,scheme):
        # 根据流向平衡与否确定可变车道切换方向,返回可行scheme列表
        # 毕设阶段，只返回一个scheme
        alpha = 0.3  # 流向延误不平衡的阈值
        delay_l = self.delay_m_es.reshape(4,3)[:,0]
        delay_t = self.delay_m_es.reshape(4,3)[:,1]
        # 不均衡指数
        var = (delay_l-delay_t)/(delay_l+delay_t)
        
        # 直接确定车道功能方案
        scheme2func = [[2,2],[2,1],[2,0],[1,0]]
        
        prev_target_func = []
        for s in scheme:
            prev_target_func.extend(scheme2func[s])
        
        if self.mode=='all':
            scheme[var>alpha] += 1
            scheme[var<-alpha] -= 1
            scheme = scheme.clip(0,3)
        elif self.mode=='signal':
            scheme = np.array([0,0,0,0])
        
        target_func = []
        for s in scheme:
            target_func.extend(scheme2func[s])            
        
        switch = np.zeros(8,dtype=int)
        switch[np.array(target_func) != np.array(prev_target_func)] = 1
    
        return [(scheme,switch)]
    
    # 检查：收益充分
    def gain_sufficiency(self):
        alpha = 0.2
        self.generate_control()
    
    def generate_control(self,vph_m,scheme,lane_func):
        # scheme: 当前scheme
        schemes_and_switches = self.flow_balance(scheme)
        discrete_controls = self.brute_force(schemes_and_switches)
        
        optimal_control_c_list = []
        optimal_control_p_list = []
        optimal_value_list = []
        
        for discrete_control in discrete_controls:
            splits,phase_group = self.multistart(discrete_control,scheme,
                                                 lane_func,vph_m)
            optimal = self.optimize(splits,discrete_control,phase_group)
            
            optimal_control_p = optimal[0]
            optimal_control_c = optimal[1]
            optimal_value = optimal[2]
            
            optimal_control_c_list.append(optimal_control_c)
            optimal_control_p_list.append(optimal_control_p)
            optimal_value_list.append(optimal_value)
        
        index = np.array(optimal_value_list).argmin()
        self.scheme = schemes_and_switches[index][0]
        optimal_control_c = optimal_control_c_list[index]
        optimal_control_p = optimal_control_p_list[index]
        
        self.predict_result.append(self.predict(torch.from_numpy(optimal_control_p)))
        self.control_result.append(optimal_control_p)
        
        return optimal_control_c
    
    def predict(self,tc):
        with torch.no_grad():
            self.model.eval()
            delay_output = self.model.decoding(self.upper_context,tc)
        return delay_output[0,:self.lookahead]
    