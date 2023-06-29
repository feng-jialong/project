#%% import 
import numpy as np
import pandas as pd
from itertools import cycle

from traci import trafficlight

#%% 交通控制器
class BaseTrafficController():
    def __init__(self,step_length,time_interval):
        # NEMA相位，周期级
        self.id = 'J'
        
        self.step_length = step_length  # 仿真步长，单位：s
        self.time_interval = time_interval  # 时间间隔，单位：s
        
        # 黄灯时间与全红时间
        self.y = 3.0
        self.r = 2.0
        self.y_time = int(self.y/self.step_length)
        self.r_time = int(self.r/self.step_length)
        
        self.g_max = 60.0
        self.g_min = 15.0
        self.C_max = 180.0
        self.C_min = 60.0

        self.state_num = 48
        # 流向编号映射到信号灯state
        # 流向的编号按照双环相位示意图
        # 1:EL, 2:WT, 3:SL, 4:NT, 5:WL, 6:ET, 7:NL, 8:ST
        self.movement2state = [[0,12,24,36],
                               [14,17,20,23],[37,40,43,46],
                               [26,29,32,35],[1,4,7,10],
                               [38,41,44,47],[13,16,19,22],
                               [2,5,8,11],[25,28,31,34]]
        # 车道编号映射到信号灯state
        ## state除3，除数和余数即为车道编号和功能, 右直左012
        ## state除12，除数即为进口到编号
        # 车道编号从N顺时针，0,1,2,...,15
        self.lane2state = lambda l: [3*l,3*l+1,3*l+2]
        
        # (场景相关)
        # 所有可变车道，允许功能对应的link的state；当前场景，两个中间车道为可变车道
        self.val2state = [[[4,5],[7,8]],
                          [[16,17],[19,20]],
                          [[28,29],[31,21]],
                          [[40,41],[43,44]]]
        # 可变车道的车道编号
        self.val_list = [1,2,5,6,9,10,13,14]
        # 关闭的车道功能
        self.invalid_state = [1,2,3,6,9,10,
                              13,14,15,18,21,22,
                              25,26,27,30,33,34,
                              37,38,39,42,45,46]
        
        self.scheme2func = [[2,2],[2,1],[2,0],[1,0]]
        
        # 可变车道id
        self.val_id = ['N_IN_P_1','N_IN_P_2',
                       'E_IN_P_1','E_IN_P_2',
                       'S_IN_P_1','S_IN_P_2',
                       'W_IN_P_1','W_IN_P_2']
        
        # 信号灯基础状态
        # 右转车流不受灯控
        self.basic_state = ['G' if i in self.movement2state[0] else 'r' 
                           for i in range(self.state_num)]
        trafficlight.setRedYellowGreenState(self.id,''.join(self.basic_state))
        
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
        
        # 可变车道的目标功能
        self.target_func = None
        # 进口道的车道功能方案
        self.scheme = np.zeros(4,dtype=int)
        # 可变车道的当前功能
        self.current_func = np.array([2,2,2,2,2,2,2,2])
        self.switch = None
        
        # <优化格式>的控制方案
        self.control = None
        # 初始状态，使用<随机>策略热启动
        self.update(mode='random')
        
    def step(self):
        # 计划下一步的信号
        # 周期切换
        if self.cycle_time == 0:
            phase_cycle_1,phase_cycle_2,split,target_func,switch = self.next_cycle()
            self.phase_cycle_1 = phase_cycle_1
            self.phase_cycle_2 = phase_cycle_2
            self.split = split
            self.cycle_time = int(np.array(split[:4]).sum())+4*(self.y_time+self.r_time)
            self.target_func = target_func
            # 权宜之计：周期开始直接切换
            self.current_func = self.target_func
            # 车道功能切换后，更新lane_func
            self.lane_func_update()
            self.switch = switch
        
        # ring 1 相位切换
        if self.phase_time_1 == 0:
            # 流向对应黄灯相位结束，变换车道功能
            # if self.phase_1 == 'y':
            #     for i,p in enumerate(self.switch):
            #         if p == self.green_phase_1:
            #             self.current_func[i] = self.target_func[i]  # 切换车道功能

            # 相位时长
            self.phase_1 = next(self.phase_cycle_1)
            if self.phase_1 == 'y':
                self.phase_time_1 = self.y_time
            elif self.phase_1 == 'r':
                self.phase_time_1 = self.r_time
            else:
                self.green_phase_1 = self.phase_1
                self.phase_time_1 = self.split[self.green_phase_1-1]
                
            # 相位设置
            for state in self.movement2state[self.green_phase_1]:
                if self.phase_1 == 'y':
                    trafficlight.setLinkState(self.id,state,'y')
                elif self.phase_1 == 'r':
                    trafficlight.setLinkState(self.id,state,'r')
                else:
                    trafficlight.setLinkState(self.id,state,'G')

        # ring 2 相位切换        
        if self.phase_time_2 == 0:
            # 流向对应黄灯相位结束，变换车道功能
            # if self.phase_2 == 'y':
            #     for i,p in enumerate(self.switch):
            #         if p == self.green_phase_2:
            #             self.current_func[i] = self.target_func[i]  # 切换车道功能
            
            # 相位时长
            self.phase_2 = next(self.phase_cycle_2)
            if self.phase_2 == 'y':
                self.phase_time_2 = self.y_time
            elif self.phase_2 == 'r':
                self.phase_time_2 = self.r_time
            else:
                self.green_phase_2 = self.phase_2
                self.phase_time_2 = self.split[self.green_phase_2-1]
            # 相位设置
            for state in self.movement2state[self.green_phase_2]:
                if self.phase_2 == 'y':
                    trafficlight.setLinkState(self.id,state,'y')
                elif self.phase_2 == 'r':
                    trafficlight.setLinkState(self.id,state,'r')
                else:
                    trafficlight.setLinkState(self.id,state,'G')
        
        self.phase_time_1 -= 1
        self.phase_time_2 -= 1
        self.cycle_time -= 1
        
    def lane_func_update(self):
        # 更新功能可行车道集
        VAL_list = [1,2,5,6,9,10,13,14]
        self.lane_func = [[[3],[],[0]],
                          [[7],[],[4]],
                          [[11],[],[8]],
                          [[15],[],[12]]]
        
        for i,lane in enumerate(VAL_list):
            if self.current_func[i] == 0:  # 左转
                self.lane_func[int(i/2)][0].append(lane)
            elif self.current_func[i] == 1:  # 直左
                self.lane_func[int(i/2)][0].append(lane)
                self.lane_func[int(i/2)][1].append(lane)
            elif self.current_func[i] == 2:  # 直行
                self.lane_func[int(i/2)][1].append(lane)
    
    def val_func_random(self):
        if np.random.rand() > 0.5:  # 控制随机采样时可变车道的切换频率
            # 允许的方案：22,21,20,10 (可变车道从外侧编号)
            # 随机生成可变车道scheme
            for i,s in enumerate(self.scheme):
                # 可变车道的状态服从马尔可夫链
                # T,T  TL,T  L,T  L,TL
                if s==0:   
                    self.scheme[i] += np.random.choice(a=[0,1],p=[0.5,0.5])
                elif s==3:
                    self.scheme[i] += np.random.choice(a=[-1,0],p=[0.5,0.5])
                elif s==2 or s==3:
                    self.scheme[i] += np.random.choice(a=[-1,0,1],p=[0.25,0.5,0.25])

        target_func = []
        for s in self.scheme:
            target_func.extend(self.scheme2func[s])
        self.control['target_func'] = np.array(target_func)
    
    def phase_rule_based(self):
        self.control['phase'] = np.array(5*[0])
        self.control['phase'][0] = 0
        # 有车道切换就对向单口放行，无车道切换就对称放行
        # 南北方向单口放行: 车道切换或共用车道
        if self.control['switch'][[0,1,4,5]].any() or \
                (self.current_func[[0,1,4,5]] == 1).any():
            self.control['phase'][[2,4]] = [0,1]
        # 南北向对称放行
        else:
            self.control['phase'][[2,4]] = [0,0]
            
        # 东西向单口放行: 车道切换或共用车道
        if self.control['switch'][[2,3,6,7]].any() or \
                (self.current_func[[2,3,6,7]] == 1).any():
            self.control['phase'][[1,3]] = [0,1]
        # 东西向对称放行
        else:
            self.control['phase'][[1,3]] = [0,0]
    
    def phase_and_split(self):
        self.control['phase'] = np.array(5*[0])
        self.control['split'] = np.array(8*[0.0])
        self.control['phase'][0] = 0
        
        # 基于规则：有车道切换就对向单口放行，无车道切换就对称放行
        # 南北方向单口放行: 车道切换或共用车道
        if self.control['switch'][[0,1,4,5]].any() or \
                (self.current_func[[0,1,4,5]] == 1).any():
            self.control['phase'][[2,4]] = [0,1]
            self.control['split'][[2,3]] = np.random.uniform(self.g_min,self.g_max,2)
            self.control['split'][[7,6]] = self.control['split'][[2,3]]
        # 南北向对称放行
        else:   
            self.control['phase'][[2,4]] = [0,0]
            self.control['split'][[2,3]] = np.random.uniform(self.g_min,self.g_max,2)
            self.control['split'][[6,7]] = self.control['split'][[2,3]]
            
        # 东西向单口放行: 车道切换或共用车道
        if self.control['switch'][[2,3,6,7]].any() or \
                (self.current_func[[2,3,6,7]] == 1).any():
            self.control['phase'][[1,3]] = [0,1]
            self.control['split'][[0,1]] = np.random.uniform(self.g_min,self.g_max,2)
            self.control['split'][[5,4]] = self.control['split'][[0,1]]
        # 东西向对称放行
        else:   
            self.control['phase'][[1,3]] = [0,0]
            self.control['split'][[0,1]] = np.random.uniform(self.g_min,self.g_max,2)
            self.control['split'][[4,5]] = self.control['split'][[0,1]]
    
    def equisaturation(self,vph_m):
        # Webster绿灯时间和周期
        # 绿信比，假设每条车道SFR相同
        sfr = 1368  # 饱和流率veh/h, 使用文献中推荐的值
        self.control['split'] = np.array(8*[0.0])
        movement2number = [[7,4],[1,6],[3,8],[5,2]]  # 流向映射到编号
        for i in range(4):
            if self.scheme[i] == 1:  # 共用车道
                r = vph_m[i,1]/vph_m[i,0]
                self.control['split'][movement2number[i][0]-1] = vph_m[i,0]/(2/(1+r)+1)
                self.control['split'][movement2number[i][1]-1] = vph_m[i,1]/(2-2/(1+r))
            elif self.scheme[i] == 3:  # 共用车道
                r = vph_m[i,0]/vph_m[i,1]
                self.control['split'][movement2number[i][0]-1] = vph_m[i,0]/(2-2/(1+r)+1)
                self.control['split'][movement2number[i][1]-1] = vph_m[i,1]/(2/(1+r))
            else:
                self.control['split'][movement2number[i][0]-1] = vph_m[i,0]/len(self.lane_func[i][0])
                self.control['split'][movement2number[i][1]-1] = vph_m[i,1]/len(self.lane_func[i][1])
        # 对齐绿信比
        if self.control['phase'][3] == 1:
            self.control['split'][0] = self.control['split'][[0,5]].max()/sfr
            self.control['split'][5] = self.control['split'][0]
            self.control['split'][1] = self.control['split'][[1,4]].max()/sfr
            self.control['split'][4] = self.control['split'][1]
        else:
            self.control['split'][0] = self.control['split'][[0,4]].max()/sfr
            self.control['split'][4] = self.control['split'][0]
            self.control['split'][1] = self.control['split'][[1,5]].max()/sfr
            self.control['split'][5] = self.control['split'][1]
        
        if self.control['phase'][4] == 1:
            self.control['split'][2] = self.control['split'][[2,7]].max()/sfr
            self.control['split'][7] = self.control['split'][2]
            self.control['split'][3] = self.control['split'][[3,6]].max()/sfr
            self.control['split'][6] = self.control['split'][3]
        else:
            self.control['split'][2] = self.control['split'][[2,6]].max()/sfr
            self.control['split'][6] = self.control['split'][2]
            self.control['split'][3] = self.control['split'][[3,7]].max()/sfr
            self.control['split'][7] = self.control['split'][3]
    
    def generate_random(self):
        self.control = {}
        
        # target_func
        self.val_func_random()

        # switch
        self.control['switch'] = np.random.choice(np.arange(1,9),len(self.val_list))
        # 功能切换约束, 功能相同则不切换
        self.control['switch'][self.current_func == self.control['target_func']] = 0
        
        # phase and split：initialization
        self.phase_and_split()
        
        # 随机方案只作为过渡，周期长度约束不用管了
        
        # 输出绿灯时间
        print(self.control['split'][:4])
    
    def generate_sample(self,vph_m):
        self.control = {}
        
        # target_func
        self.val_func_random()
        
        # switch
        self.control['switch'] = np.random.choice(np.arange(1,9),len(self.val_list))
        # 功能切换约束, 功能相同则不切换
        self.control['switch'][self.current_func == self.control['target_func']] = 0
        
        # phase
        self.phase_rule_based()
        # split
        self.equisaturation(vph_m)

        # 随机调整周期
        self.control['split'] /= self.control['split'].sum()/2  # 注意双环是四相位
        self.control['split'] *= np.random.uniform(60.0,180.0)  # 限制范围内随机选取周期
        # 绿灯时间范围裁剪
        self.control['split'] = self.control['split'].clip(self.g_min,self.g_max)
        
        # 输出绿灯时间
        print(self.control['split'][:4])
        
    def generate_webster(self,vph_m):
        # 给出webster配时方案
        self.control = {}
        # target_func
        self.control['target_func'] = np.array([2,2,2,2,2,2,2,2])
        self.scheme = np.zeros(4,dtype=int)
        # switch
        self.control['switch'] = np.zeros(8,dtype=int)
        # phase
        self.phase_rule_based()
        # split
        self.equisaturation(vph_m)

        # 周期，假设启动损失时间3.0s
        L = 4.0*((self.y + self.r)/2 + 3.0)
        # 关键流量比
        y = self.control['split'][:4].sum()
        if y > 0.8:
            y = 0.8
        cycle = (1.5*L+5.0)/(1-y)
        cycle = np.clip(cycle,self.C_min,self.C_max)
        G = cycle - L  # 周期有效绿灯时间
        
        # 相位有效绿灯时间
        self.control['split'] /= self.control['split'].sum()/2
        self.control['split'] *= G
        self.control['split'] -= (self.y + self.r)/2
        self.control['split'] = np.clip(self.control['split'],self.g_min,self.g_max)
        
        # 输出绿灯时间
        print(self.control['split'][:4])

    def generate_mpc(self,vph_m,mpc_controller):
        self.control =  mpc_controller.generate_control(vph_m,self.scheme,self.lane_func)
        self.scheme = mpc_controller.scheme
        
        # 输出绿灯时间
        print(self.control['split'][:4])
        
    def update(self,mode,monitor=None,mpc_controller=None):
        # 生成或获取控制方案
        # 采样策略，随机生成<优化格式>的控制方案
        # <优化格式>
        
        # sample: 绿灯时间的webster采样, <Webster>
        # random: 绿灯时间的随机采样，<随机>
        # webster: 绿灯时间基于Webster公式
        # mpc: 绿灯时间的优化采样，<优化>
        assert mode in ['sample','random','mpc','webster']
        
        if mode == 'random':
            self.generate_random()
        elif mode == 'sample':
            self.generate_sample(monitor.vph_m)
        elif mode == 'mpc':
            self.generate_mpc(monitor.vph_m,mpc_controller)
        elif mode == 'webster':
            self.generate_webster(monitor.vph_m)   
    
    def next_cycle(self):
        # 将下一个周期的控制方案部署到控制器
        # 接收<优化格式>的控制方案，转换成<执行格式>
        control = self.control
        
        # 信号相关：五个0-1变量确定环内流向顺序，六个连续变量确定相位分隔
        g_swap,r1g1_swap,r1g2_swap,r2g1_swap,r2g2_swap = control['phase']
        # 各个流向1,2,...,8的绿灯时间，取整并检查约束
        split = control['split']
        split = [int(t/self.step_length) for t in split]
        split[5] = split[0]+split[1]-split[4]
        split[7] = split[2]+split[3]-split[6]
        phase_cycle_1 = [1+1*r1g1_swap+2*g_swap,'y','r',
                         2-1*r1g1_swap+2*g_swap,'y','r',
                         3+1*r1g2_swap-2*g_swap,'y','r',
                         4-1*r1g2_swap-2*g_swap,'y','r']
        phase_cycle_2 = [5+1*r2g1_swap+2*g_swap,'y','r',
                         6-1*r2g1_swap+2*g_swap,'y','r',
                         7+1*r2g2_swap-2*g_swap,'y','r',
                         8-1*r2g2_swap-2*g_swap,'y','r']
        
        # 可变车道
        target_func = control['target_func']   # 目标车道功能
        switch = control['switch']   # 是否切换及切换相位
        
        return cycle(phase_cycle_1),cycle(phase_cycle_2),split,target_func,switch
    
    def output(self):
        # 输出<优化格式>的控制方案
        return self.control
    
    def is_to_update(self):
        return self.cycle_time == 0