#%% import 
from traci import vehicle as vehicle
import numpy as np
import pandas as pd

from utils import get_movement

route_list = ['EN_S','EN_W','EN_N',
            'EE_S','EE_W','EE_N',
            'ES_S','ES_W','ES_N',
            'SE_W','SE_N','SE_E',
            'SS_W','SS_N','SS_E',
            'SW_W','SW_N','SW_E',
            'WS_N','WS_E','WS_S',
            'WW_N','WW_E','WW_S',
            'WN_N','WN_E','WN_S',
            'NW_E','NW_S','NW_W',
            'NN_E','NN_S','NN_W',
            'NE_E','NE_S','NE_W']

route_info = pd.DataFrame({'route':route_list,
                           'inlet':[get_movement(route)[0] for route in route_list],
                           'turn':[get_movement(route)[1] for route in route_list]})

#%% 车辆生成器
class VehicleGenerator():
    def __init__(self,route_info=route_info,mode='static',duration=60*30):
        self.route_info = route_info
        self.route_num = self.route_info.shape[0]
        self.duration = duration  # 持续时间/秒
        self.time = 0  # 仿真进度
        self.schedule = {}
        
        self.vph_m_list = []  # 流向流量信息
        
        if mode=='static':
            self.generate_demand = self.generate_static_demand
        elif mode=='random':
            self.generate_demand = self.generate_random_demand
        elif mode=='stepfunc':
            self.generate_demand = self.generate_stepfunc_demand
        elif mode=='linear':
            self.generate_demand = self.generate_linear_demand
        elif mode=='sine':
            self.generate_demand = self.generate_sine_demand
        
    def run(self):
        if self.time%self.duration==0:
            self.vph_m = np.zeros((self.duration,4,3))
            self.generate_demand()
            self.vph_m_list.append(self.vph_m)
            
        self.generate_vehicle()
        self.time +=1
            
    def generate_vehicle(self):
        for r in route_info['route'].values:
            turn = get_movement(r)[1]
            veh_type = 'LEFT' if turn==0 else 'THROUGH' if turn==1 else 'RIGHT'
            for i in range(self.schedule[r][self.time%self.duration]):
                vehicle.add(vehID=r+'.'+str(self.time)+'.'+str(i),
                            routeID=r,typeID=veh_type,departLane='best')
    
    def generate_static_demand(self):
        # 恒定vph
        # # 进口道基础vph
        vph_level = np.array([1120.0,1120.0,1120.0,1120.0])
        # 进口道转向比
        turn_ratio = np.array([[0.25,0.5,0.25],
                               [0.25,0.5,0.25],
                               [0.25,0.5,0.25],
                               [0.25,0.5,0.25]])
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            self.vph_m[0,inlet,turn] = vph_level[inlet]*turn_ratio[inlet,turn]
            
            headway = 3600/(1/3*self.vph_m[0,inlet,turn])
            headway = np.random.exponential(headway)
            
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                # region 错误做法
                # while True: 
                #     # 假设到达车流由周围交叉口的三个进口道平均贡献
                #     headway = 3600/(1/3*vph_level[inlet]*turn_ratio[inlet,turn])
                #     headway = np.random.exponential(headway)
                #     if (second-headway)>0:
                #         second -= headway
                #         n_veh += 1
                #     else:
                #         break
                # endregion
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    headway = 3600/(1/3*self.vph_m[t,inlet,turn])
                    headway = np.random.exponential(headway)
                self.schedule[r].append(n_veh)
    
    def generate_random_demand(self):
        # 随机生成的恒定vph
        # 进口道基础vph
        vph_level = np.random.uniform(200.0,1200.0,4)
        # 进口道转向比
        turn_ratio = np.random.uniform(1.0,4.0,(4,3))
        turn_ratio = turn_ratio/turn_ratio.sum(axis=-1)[:,None]
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            self.vph_m[self.time,inlet,turn] = vph_level[inlet]*turn_ratio[inlet,turn]
            
            headway = 3600/(1/3*self.vph_m[inlet,turn])
            headway = np.random.exponential(headway)
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    headway = 3600/(1/3*self.vph_m[inlet,turn])
                    headway = np.random.exponential(headway)
                self.schedule[r].append(n_veh)
        
    def generate_stepfunc_demand(self):
        # 动态的vph: piece-wise linear, e.g. stairs
        # 进口道基础vph
        # step start location
        breakpoint_list = np.array([0,1200,2400,3600,4800,6000])
        # step volume level
        vph_list = np.array([])
        
        vph_level = np.zeros((4,self.duration))
        vph_level[:,:1200] = np.array(1200*[[640.0,640.0,640.0,640.0]]).T
        vph_level[:,1200:2400] = np.array(1200*[[960.0,960.0,960.0,960.0]]).T
        vph_level[:,2400:] = np.array((self.duration-2400)*[[640.0,640.0,640.0,640.0]]).T
        
        # 进口道转向比
        turn_ratio = [[0.25,0.5,0.25],
                      [0.25,0.5,0.25],
                      [0.25,0.5,0.25],
                      [0.25,0.5,0.25]]
        turn_ratio = np.array(turn_ratio)
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            
            headway = 3600/(1/3*vph_level[t,inlet,0]*turn_ratio[t,inlet,turn])
            headway = np.random.exponential(headway)
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    headway = 3600/(1/3*vph_level[inlet,i]*turn_ratio[inlet,turn])
                    headway = np.random.exponential(headway)
                self.schedule[r].append(n_veh)
    
    def generate_linear_demand(self):
        # 动态变化的vph: linear
        # 进口道基础vph
        vph_level_a = np.random.uniform(200.0,1200.0,4)
        vph_level_b = np.random.uniform(200.0,1200.0,4)
        vph_level = np.linspace(vph_level_a,vph_level_b,self.duration,axis=0)
        
        # 进口道转向比
        turn_ratio_a = np.random.uniform(1.0,4.0,(4,3))
        turn_ratio_b = np.random.uniform(1.0,4.0,(4,3))
        turn_ratio_a = turn_ratio_a/turn_ratio_a.sum(axis=-1)[:,None]
        turn_ratio_b = turn_ratio_b/turn_ratio_b.sum(axis=-1)[:,None]
        turn_ratio = np.linspace(turn_ratio_a,turn_ratio_b,self.duration,axis=0)
        
        for r in self.route_info['route'].values:
            inlet,turn,_ = get_movement(r)
            self.schedule[r] = []
            second = 0.0
            self.vph_m[0,inlet,turn] = vph_level[0,inlet]*turn_ratio[0,inlet,turn]
            
            headway = 3600/(1/3*self.vph_m[0,inlet,turn])
            headway = np.random.exponential(headway)
            
            for t in range(self.duration):
                second += 1.0
                n_veh = 0
                while (second-headway)>0:
                    second -= headway
                    n_veh += 1
                    
                    self.vph_m[t,inlet,turn] = vph_level[t,inlet]*turn_ratio[t,inlet,turn]
                    
                    headway = 3600/(1/3*self.vph_m[t,inlet,turn])
                    headway = np.random.exponential(headway)
                self.schedule[r].append(n_veh)
        
    def generate_sine_demand(self):
        # 动态变化的vph: sine curve
        pass
    
    def output(self):
        return np.concatenate(self.vph_m_list,axis=0)