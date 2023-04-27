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
    def __init__(self,step_length):
        # NEMA相位
        self.id = 'J'
        
        self.y_time = int(3.0/step_length)
        self.r_time = int(1.0/step_length)

        self.state_num = 16
        # 流向编号映射到信号灯state
        self.state_group = {'0':[0,4,8,12],
                            '1':[7],
                            '2':[13,14],
                            '3':[11],
                            '4':[1,2],
                            '5':[15],
                            '6':[5,6],
                            '7':[3],
                            '8':[9,10]}
        
        self.all_red = ['G' if i in self.state_group['0'] else 'r' 
                        for i in range(self.state_num)]  # 右转车道不受灯控
        trafficlight.setRedYellowGreenState(self.id,''.join(self.all_red))
        
        self.step_length = step_length
        
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
        self.green_split = None
        
    def run(self):
        # 规划下一步的信号
        # 周期切换
        if self.cycle_time == 0:
            phase_cycle_1,phase_cycle_2,split = self.next_cycle()
            self.phase_cycle_1 = phase_cycle_1
            self.phase_cycle_2 = phase_cycle_2
            self.green_split = split
            self.cycle_time = int(np.array(split[:4]).sum())+4*(self.y_time+self.r_time)
            
        if self.phase_time_1 == 0:
            self.phase_1 = next(self.phase_cycle_1)
            if self.phase_1 == 'yellow':
                self.phase_time_1 = self.y_time
            elif self.phase_1 == 'red':
                self.phase_time_1 = self.r_time
            else:
                self.green_phase_1 = self.phase_1
                self.phase_time_1 = self.green_split[self.green_phase_1-1]
                
        if self.phase_time_2 == 0:
            self.phase_2 = next(self.phase_cycle_2)
            if self.phase_2 == 'yellow':
                self.phase_time_2 = self.y_time
            elif self.phase_2 == 'red':
                self.phase_time_2 = self.r_time
            else:
                self.green_phase_2 = self.phase_2
                self.phase_time_2 = self.green_split[self.green_phase_2-1]
        
        # 每一时间步都设置状态(可能会很慢？)
        # trafficlight.setRedYellowGreenState(self.id,self.all_red)
        # all_state = self.all_red
        # ring 1
        for state in self.state_group[str(self.green_phase_1)]:
            if self.phase_1 == 'yellow':
                trafficlight.setLinkState(self.id,state,'y')
                # all_state[state] = 'y'
            elif self.phase_1 == 'red':
                trafficlight.setLinkState(self.id,state,'r')
                # all_state[state] = 'r'
            else:
                trafficlight.setLinkState(self.id,state,'G')
                # all_state[state] = 'G'
        # ring 2 
        for state in self.state_group[str(self.green_phase_2)]:
            if self.phase_2 == 'yellow':
                trafficlight.setLinkState(self.id,state,'y')
                # all_state[state] = 'y'
            elif self.phase_2 == 'red':
                trafficlight.setLinkState(self.id,state,'r')
                # all_state[state] = 'r'
            else:
                trafficlight.setLinkState(self.id,state,'G')
                # all_state[state] = 'G'

        # trafficlight.setRedYellowGreenState(self.id,''.join(all_state))
        
        self.phase_time_1 -= 1
        self.phase_time_2 -= 1
        self.cycle_time -= 1
    
    def next_cycle(self,control=None):
        # 全权负责将控制器的状态推进至下一个周期
        # 随机生成控制方案
        control = {}
        control['phase'] = np.random.choice([0,1],5)
        split_min = int(15.0/self.step_length)
        split_max = int(45.0/self.step_length)
        split = 8*[0]
        split[0],split[1] = np.random.uniform(split_min,split_max,2)
        split[4] = np.random.uniform(split_min,split[0]+split[1]-split_min)

        
        split[2],split[3] = np.random.uniform(split_min,split_max,2)
        split[6] = np.random.uniform(split_min,split[2]+split[3]-split_min)
        
        # 转变为步长
        split = [int(t/self.step_length) for t in split]  # 先取整，保证barrier约束成立
        split[5] = split[0]+split[1]-split[4]
        split[7] = split[2]+split[3]-split[6]
        control['split'] = split
        
        # 信号相关：五个0-1变量确定环内流向顺序，六个连续变量确定相位分隔
        # 可变车道相关：todo
        g_swap,r1g1_swap,r1g2_swap,r2g1_swap,r2g2_swap = control['phase']
        split = control['split']  # 各个流向的绿灯时间，满足约束
        phase_cycle_1 = [1+1*r1g1_swap+2*g_swap,'yellow','red',
                         2-1*r1g1_swap+2*g_swap,'yellow','red',
                         3+1*r1g2_swap-2*g_swap,'yellow','red',
                         4-1*r1g2_swap-2*g_swap,'yellow','red']
        phase_cycle_2 = [5+1*r2g1_swap+2*g_swap,'yellow','red',
                         6-1*r2g1_swap+2*g_swap,'yellow','red',
                         7+1*r2g2_swap-2*g_swap,'yellow','red',
                         8-1*r2g2_swap-2*g_swap,'yellow','red']
        
        self.control = control
        
        return cycle(phase_cycle_1),cycle(phase_cycle_2),split
    
    def output(self):
        return self.control
    
class WebstersTrafficController(TrafficController):
    # 周期级自适应webster信号控制器
    def __init__(self,id,r_time,y_time,g_min,c_min,c_max):
        self.id = id
        
        self.r_time = r_time  # 红灯时间步
        self.y_time = y_time  # 黄灯时间步
        self.g_min = g_min
        self.c_min = c_min
        self.c_max = c_max
        
        self.movement_state = {'WT':[13,14],'EL':[7],'ST':[9,10],'NL':[3],  # ring1 的四股车流
                               'ET':[5,6],'WL':[15],'NT':[1,2],'SL':[11],  # ring2 的四股车流
                               'R':[0,4,8,12]}   # 右转车流
        
        # hardcode, 绿灯相位以及对应state
        self.green_phases = ['GGGrGrrrGGGrGrrr',
                             'GrrGGrrrGrrGGrrr',
                             'GrrrGGGrGrrrGGGr',
                             'GrrrGrrGGrrrGrrG']
        
        self.all_red = len((self.green_phases[0]))*'r'
        
        self.phase_time = 0  # 相位剩余的时间步
        self.cycle_time = 0  # 周期剩余的时间步
        self.phase_cycle = self.next_cycle()  # 周期的相位循环
        self.green_phase_duration = {g:self.g_min for g in self.green_phases}  # 绿灯时长
    
    def run(self):
        if self.cycle_time == 0:
            # 周期切换
            self.phase_cycle = self.next_cycle()
        if self.phase_time == 0:
            # 相位切换
            self.phase = next(self.phase_cycle)
            self.conn.trafficlight.setRedYellowGreenState( self.id, self.phase)
            self.phase_time = self.next_phase_duration()

        self.phase_time -= 1
        self.cycle_time -= 1
    
    def next_phase(self):
        return next(self.phase_cycle)
    
    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_phase_duration[self.phase]
        elif 'y' in self.phase:
            return self.y_time
        else:
            return self.r_time
        
    def next_cycle(self):
        phase_cycle = []
        green_phases = self.green_phases
        next_green_phases = self.green_phases[1:] + [self.green_phases[0]]
        for g,next_g in zip(green_phases,next_green_phases):
            phase_cycle.append(g)
            phase_cycle.extend(self.get_inter_phase(self,g,next_g))
        return phase_cycle
    
    def update(self):
        pass
    
    def get_inter_phase(self,phase,next_phase):
        if phase == next_phase or phase == self.all_red:
            return []
        else:
            yellow_phase = ''.join([ p if p == 'r' else 'y' for p in phase ])
            return [yellow_phase, self.all_red]
    
    def webster(self):
        """update green phase times using lane
        vehicle counts and Websters' traffic signal
        control method
        """
        ##compute flow ratios for all lanes in all green phases
        ##find critical
        y_crit = []
        for g in self.green_phases:
            sat_flows = [(self.phase_lane_counts[g][l]/self.update_freq)/(self.sat_flow) for l in self.phase_lanes[g]]
            y_crit.append(max(sat_flows))

        #compute intersection critical lane flow rattios
        Y = sum(y_crit)
        if Y > 0.85:
            Y = 0.85
        elif Y == 0.0:
            Y = 0.01

        #limit in case too saturated
        #compute lost time
        L = len(self.green_phases)*(self.red_t + self.yellow_t)

        #compute cycle time
        C = int(((1.5*L) + 5)/(1.0-Y))
        #constrain if necessary
        if C > self.c_max:
            C = self.c_max
        elif C < self.c_min:
            C = self.c_min

        G = C - L
        #compute green times for each movement
        #based on total green times
        for g, y in zip(self.green_phases, y_crit):
            g_t = int((y/Y)*G)
            #constrain green phase time if necessary
            if g_t < self.g_min:
                g_t = self.g_min
            self.green_phase_duration[g] = g_t

    def movement_state(self):
        # hardcode
        pass

class MaxPressureTrafficController(TrafficController):
    # 相位级自适应max-pressure控制器
    def __init__(self):
        super(TrafficController,self).__init__()