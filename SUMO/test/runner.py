#%%
# region import
import os,sys
import xml.etree.ElementTree as et
from xml.dom import minidom
import numpy as np
import pickle
from tqdm import tqdm
import importlib as imp

# check environment varialble 'SUMO_HOME'
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from traci import vehicle
from traci import lane
from traci import multientryexit
from traci import junction
from traci import constants

import traffic_controller
try:
    imp.reload(traffic_controller)
except:
    pass
from traffic_controller import BaseTrafficController

# endregion

#%%
# 根据车辆的路径id获取转向
def get_movement(object_id):
    # object_id为vehicle_id或route_id
    # movement为0表示该网格无车
    inlet_index = {'W':0,'N':1,'E':2,'S':3}
    o = inlet_index[object_id[0]]
    d = inlet_index[object_id[3]]
    if (d-o)%4 == 1:
        return 'L' # 左转
    elif (d-o)%4 == 2:
        return 'T' # 直行
    elif (d-o)%4 == 3:
        return 'R'  # 右转

# 随机生成时常交通需求
def generate_demand():
    routes = et.Element('routes')
    tree = et.ElementTree(routes)
    
    # 所有路径
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
    
    # 均匀分布
    # demand_data = np.random.uniform(40,400,len(route_list))
    # 泊松分布，考虑转向
    demand_ratio = {'L':0.2,'T':0.4,'R':0.2}
    demand_basic = 1000
    # 需求数据
    demand_data = [np.random.poisson(demand_ratio[get_movement(r)]*demand_basic) for r in route_list]
    demand = iter(demand_data)

    # region 设置车流
    for route_id in route_list:
        et.SubElement(routes,'flow',
                      {'id':route_id,
                       'begin':'0','end':'86400',
                       'vehsPerHour':str(next(demand)),
                       'route':route_id,
                       'departLane':'random'})
    
    tree.write('flow.rou.xml')
    pretty_xml = minidom.parse('flow.rou.xml').toprettyxml(encoding='UTF-8')
    with open('flow.rou.xml','wb') as file:
        file.write(pretty_xml)
    # endregion
    
    return demand_data

# 监测车均延误
class DelayMonitor():
    def __init__(self):
        self.prev_veh = set()
        self.cur_veh = set()

        self.time_loss = 0  # 周期总延误
        self.depart_num = 0  # 周期内离开检测器的车辆
        self.veh_dict = {}  # 交叉口内的车辆
    
    def update(self):
        self.prev_veh = self.cur_veh
        self.cur_veh = set(multientryexit.getLastStepVehicleIDs("e3"))

        # 上一时间步到达交叉口的车辆
        for veh_id in (self.cur_veh - self.prev_veh):
            self.veh_dict[veh_id] = vehicle.getTimeLoss(veh_id)
        
        # 上一时间步离开交叉口的车辆
        for veh_id in (self.prev_veh - self.cur_veh):
            self.depart_num += 1
            self.time_loss += vehicle.getTimeLoss(veh_id) - self.veh_dict[veh_id]
            self.veh_dict.pop(veh_id)
            
    def output(self):
        if self.depart_num == 0:
            mean_delay = 0.0
        else:
            mean_delay = self.time_loss/self.depart_num
        self.time_loss = 0.0
        self.depart_num = 0
        return mean_delay

class Observer():
    def __init__(self):
        self.id = 'J'   # 路段设备所在交叉口id
        self.max_obs_range = 400.0   # 路端设备观测范围  # 敏感性分析
        
        junction.subscribeContext(self.id,
                                  constants.CMD_GET_VEHICLE_VARIABLE,
                                  self.max_obs_range,
                                  [constants.VAR_SPEED,constants.VAR_ROUTE_ID])
        
        self.inlet_index = {'W':0,'N':1,'E':2,'S':3}
    
    def output(self):
        obs = []
        
        vehicle_info = junction.getContextSubscriptionResults(self.id)
        for vehicle_id in vehicle_info:
            lane_id = vehicle.getLaneID(vehicle_id)
            if lane_id[0] == ':':  # 排除越过停车线的交叉口中车辆，主要为不受信控的右转车
                continue
            if lane_id[2] == 'O':  # 排除交叉口附近正在离开的车辆
                continue
            inlet_index = self.inlet_index[lane_id[0]]
            lane_index = int(lane_id[-1])

            # 横向位置
            lat_pos = vehicle.getLateralLanePosition(vehicle_id)
            
            # 纵向位置, 从交叉口开始计算
            if lane_id[-3] == 'P':   # 在不可变道进口段
                lon_pos = (lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id)))
            else:   # 在可变道进口段
                lon_pos = lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id))
                lon_pos += lane.getLength(lane_id[:5] + 'P_' + lane_id[-1])
 
            # 速度
            speed = vehicle.getSpeed(vehicle_id)
            
            # 导向
            move = get_movement(vehicle_id)
            
            obs.append({'inlet_index':inlet_index,'lane_index':lane_index,
                        'lat_pos':lat_pos,'lon_pos':lon_pos,
                        'speed':speed,'move':move})
        return obs

#%% interaction between server and client
def run(number,step_length,mode='single',cycle_to_run=1):
    cycle_step = 0
    
    delay_monitor = DelayMonitor()
    observer = Observer()
    tc = BaseTrafficController(step_length)
    
    obs_list = []  # 记录观测数据
    delay_list = []  # 记录延误数据
    tc_list = []  # 记录信号控制数据

    warm_up = 4   # 热启动需要的周期数  # 可能需要敏感性分析
    
    if mode == 'single':
        cycle_to_run = 1
    elif mode == 'multiple':
        pass
    
    pbar = tqdm(total=cycle_to_run+warm_up,desc=f"simulation {number}")
    
    # 按周期数进行仿真
    while(cycle_step<=warm_up+cycle_to_run):
        # 规划下一步
        tc.run()
        # 仿真进一步
        traci.simulationStep()

        # 处理上一步
        delay_monitor.update()
        # 周期即将切换时进行的处理
        if tc.cycle_time == 0:
            obs_list.append(observer.output())  # 下一周期的观测
            delay_list.append(delay_monitor.output())  # 上一周期的延误
            tc_list.append(tc.output())   # 上一周期的信号控制
            cycle_step += 1
            pbar.update(1)
    
    # 热启动，以及观测和延误的对齐
    obs_list = obs_list[warm_up:-1]
    delay_list = delay_list[warm_up+1:]
    tc_list = tc_list[warm_up+1:]
    
    # 数据收集模式，单周期数据，连续多周期数据
    if mode=='single':
        obs_list = obs_list[-1]
        delay_list = delay_list[-1]
        tc_list = tc_list[-1]
    if mode=='multiple':
        pass
    
    # 结束仿真
    traci.close()
    
    return (obs_list,delay_list,tc_list)

#%% 
step_length = 0.5  # 敏感性分析
sumocfg = ["sumo-gui",
            "--route-files","test.rou.xml,flow.rou.xml",
            "--net-file","test.net.xml",
            "--additional-files","test.e3.xml",
            "--delay","0",
            "--step-length",f"{step_length}"]

data_dir = '../../data/synthetic_data/'

#%% 单周期数据集构建
def single_cycle_simulate(runs,data_name):
    obs_data = []
    delay_data = []
    tc_data = []
    demand_data = []

    for i in range(runs):  # 运行多次仿真，收集数据
        demand_data.append(generate_demand())
        
        try:
            traci.close()
        except:
            pass
        traci.start(sumocfg,port=666)
        
        obs,delay,tc = run(i,step_length,mode='single')
        
        obs_data.append(obs)
        delay_data.append(delay)
        tc_data.append(tc)

    data = {'obs':obs_data,'delay':delay_data,'tc':tc_data,'demand':demand_data}

    # data dimension description
    # obs: (list:sample, list:obs_vehicle, dict:vehicle_info)
    # delay：(list:sample)
    # tc：(list:sample, list:timing params)
    # demand：(list:sample, list:route)


    with open(data_dir+data_name+'.pickle', 'wb') as f:
        pickle.dump(data, f)
        
    with open(data_dir+data_name+'.pickle', 'rb') as f:
        data = pickle.load(f)

#%% 多周期数据集构建
def multiple_cycle_simulate(runs,cycle_to_run,data_name):
    obs_data = []
    delay_data = []
    tc_data = []
    demand_data = []

    for i in range(runs):  # 运行多次仿真，收集数据
        demand_data.append(generate_demand())
        
        try:
            traci.close()
        except:
            pass
        traci.start(sumocfg,port=666)
        
        obs,delay,tc = run(i,step_length,mode='multiple',cycle_to_run=cycle_to_run)
        
        obs_data.append(obs)
        delay_data.append(delay)
        tc_data.append(tc)

    data = {'obs':obs_data,'delay':delay_data,'tc':tc_data,'demand':demand_data}

    # data dimension description
    # obs: (list:sample, list:cycle, list:obs_vehicle, dict:vehicle_info)
    # delay：(list:sample, list:cycle)
    # tc：(list:sample, list:cycle, list:timing params)
    # demand：(list:sample, list:cycle, list:route)

    # 保存
    with open(data_dir+data_name+'.pickle', 'wb') as f:
        pickle.dump(data, f)

    # 读取
    with open(data_dir+data_name+'.pickle', 'rb') as f:
        data = pickle.load(f)

#%%
# multiple_cycle_simulate(250,20,'multiple_cycle_3')
single_cycle_simulate(1,'test')