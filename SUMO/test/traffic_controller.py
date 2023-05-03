#%% import 
import numpy as np
import pandas as pd
from itertools import cycle

import traci
from traci import trafficlight

#%% 信号控制器
class TrafficController():
    def __init__(self):
        pass
    
    def update(self):
        pass
    
    def run(self):
        pass

class BaseTrafficController(TrafficController):
    def __init__(self):
        # NEMA相位，周期级
        self.id = 'J'
        
        self.step_length = 0.2  # 仿真步长，单位：s
        self.time_interval = 1.0  # 时间间隔，单位：s
        
        # 黄灯时间与全红时间
        self.y_time = int(4.0/self.step_length)
        self.r_time = int(2.0/self.step_length)

        self.state_num = 20
        # 流向编号映射到信号灯state,流向对应编号按照双环相位示意图
        self.state_group = {'0':[0,5,10,15],
                            '1':[8,9],'2':[16,17],'3':[13,14],'4':[1,2],
                            '5':[18,19],'6':[6,7],'7':[3,4],'8':[11,12]}
        # N, E, S, W进口的可变车道状态
        self.val_state_group = [[2,3],[7,8],[12,13],[17,18]]
        
        self.all_red = ['G' if i in self.state_group['0'] else 'r' 
                        for i in range(self.state_num)]  # 右转车道不受灯控
        trafficlight.setRedYellowGreenState(self.id,''.join(self.all_red))
        
        # 当前绿灯相位
        self.green_phase_1 = None
        self.green_phase_2 = None
        # 当前相位
        self.phase_1 = None
        self.phase_2 = None
        # 当前相位时长
        self.phase_time_1 = 0  # ring1相位剩余的时间步
        self.phase_time_2 = 0  # ring2相位剩余的时间步
        
        # 周期时长
        self.cycle_time = 0
        
        # 相位循环
        self.phase_cycle_1 = None
        self.phase_cycle_2 = None
        
        # 各流向绿灯时间
        self.split = None
        
        self.target_func = None
        self.current_func = [2,2,2,2]
        self.switch = None
        
        # <优化格式>的控制方案
        self.control = None

    def set_val(self):
        for i,func in enumerate(self.current_func):
            if func == 0: # 左转
                trafficlight.setLinkState(self.id,self.val_state_group[i][1],'r')
            elif func == 1:  # 直左共用
                pass
            elif func == 2:  # 直行
                trafficlight.setLinkState(self.id,self.val_state_group[i][0],'r')
                
    def run(self):
        # 计划下一步的信号
        # 周期切换
        if self.cycle_time == 0:
            phase_cycle_1,phase_cycle_2,split,target_func,switch = self.next_cycle()
            self.phase_cycle_1 = phase_cycle_1
            self.phase_cycle_2 = phase_cycle_2
            self.split = split
            self.cycle_time = int(np.array(split[:4]).sum())+4*(self.y_time+self.r_time)
            self.target_func = target_func
            self.switch = switch
        
        # ring 1 相位切换
        if self.phase_time_1 == 0:
            # 流向对应黄灯相位结束，变换车道功能
            if self.phase_1 == 'yellow':
                for i,p in enumerate(self.switch):
                    if p == self.green_phase_1:
                        self.current_func[i] = self.target_func[i]  # 切换车道功能

            # 相位时长
            self.phase_1 = next(self.phase_cycle_1)
            if self.phase_1 == 'yellow':
                self.phase_time_1 = self.y_time
            elif self.phase_1 == 'red':
                self.phase_time_1 = self.r_time
            else:
                self.green_phase_1 = self.phase_1
                self.phase_time_1 = self.split[self.green_phase_1-1]
                
            # 相位设置
            for state in self.state_group[str(self.green_phase_1)]:
                if self.phase_1 == 'yellow':
                    trafficlight.setLinkState(self.id,state,'y')
                elif self.phase_1 == 'red':
                    trafficlight.setLinkState(self.id,state,'r')
                else:
                    trafficlight.setLinkState(self.id,state,'G')
            
            # 根据可变车道功能覆盖信号状态
            self.set_val()

        # ring 2 相位切换        
        if self.phase_time_2 == 0:
            # 流向对应黄灯相位结束，变换车道功能
            if self.phase_2 == 'yellow':
                for i,p in enumerate(self.switch):
                    if p == self.green_phase_2:
                        self.current_func[i] = self.target_func[i]  # 切换车道功能
            # 相位时长
            self.phase_2 = next(self.phase_cycle_2)
            if self.phase_2 == 'yellow':
                self.phase_time_2 = self.y_time
            elif self.phase_2 == 'red':
                self.phase_time_2 = self.r_time
            else:
                self.green_phase_2 = self.phase_2
                self.phase_time_2 = self.split[self.green_phase_2-1]
            # 相位设置
            for state in self.state_group[str(self.green_phase_2)]:
                if self.phase_2 == 'yellow':
                    trafficlight.setLinkState(self.id,state,'y')
                elif self.phase_2 == 'red':
                    trafficlight.setLinkState(self.id,state,'r')
                else:
                    trafficlight.setLinkState(self.id,state,'G')
                    
            # 根据可变车道功能覆盖信号状态
            self.set_val()
            
        self.phase_time_1 -= 1
        self.phase_time_2 -= 1
        self.cycle_time -= 1
    
    def sampling_control(self):
        # 采样策略，随机生成<优化格式>的控制方案
        # SPaT
        control = {}
        control['phase'] = np.random.choice([0,1],5)
        split_min = 15.0
        split_max = 60.0
        split = 8*[0.0]
        split[0],split[1] = np.random.uniform(split_min,split_max,2)
        split[4] = np.random.uniform(split_min,split[0]+split[1]-split_min)

        split[2],split[3] = np.random.uniform(split_min,split_max,2)
        split[6] = np.random.uniform(split_min,split[2]+split[3]-split_min)

        split[5] = split[0]+split[1]-split[4]
        split[7] = split[2]+split[3]-split[6]
        control['split'] = split
        
        # VAL
        control['target_func'] = np.random.choice(3,4)
        control['switch'] = np.random.choice(np.arange(1,9),4)
        control['switch'][self.current_func == control['target_func']] = 0   # 功能切换约束
        
        return control
    
    def optimize_control(self):
        # 优化策略，基于优化方法给出控制方案
        pass
    
    def next_cycle(self,control=None):
        # 全权负责将控制器的状态推进至下一个周期
        # 接收<优化格式>的控制方案，转换成<执行格式>
        
        # 是否输入了控制方案，无输入则使用<采样策略>生成控制方案
        if not control:
            control = self.sampling_control()
        self.control = control  
        
        # 信号相关：五个0-1变量确定环内流向顺序，六个连续变量确定相位分隔
        g_swap,r1g1_swap,r1g2_swap,r2g1_swap,r2g2_swap = control['phase']
        # 各个流向1,2,...,8的绿灯时间，取整并检查约束
        split = control['split']
        split = [int(t/self.step_length) for t in split]
        split[5] = split[0]+split[1]-split[4]
        split[7] = split[2]+split[3]-split[6]
        phase_cycle_1 = [1+1*r1g1_swap+2*g_swap,'yellow','red',
                         2-1*r1g1_swap+2*g_swap,'yellow','red',
                         3+1*r1g2_swap-2*g_swap,'yellow','red',
                         4-1*r1g2_swap-2*g_swap,'yellow','red']
        phase_cycle_2 = [5+1*r2g1_swap+2*g_swap,'yellow','red',
                         6-1*r2g1_swap+2*g_swap,'yellow','red',
                         7+1*r2g2_swap-2*g_swap,'yellow','red',
                         8-1*r2g2_swap-2*g_swap,'yellow','red']
        
        # 可变车道
        target_func = control['target_func']   # 目标车道功能
        switch = control['switch']   # 是否切换及切换相位
        
        return cycle(phase_cycle_1),cycle(phase_cycle_2),split,target_func,switch
    
    def output(self):
        # 输出<优化格式>的控制方案
        return self.control