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
from traci import route
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
# 编号映射
inlet_map = {'N':0,'E':1,'S':2,'W':3}
direction_map = {'L':0,'T':1,'R':2}
# 根据车辆的路径id获取转向
def get_movement(object_id):
    # object_id为vehicle_id或route_id
    # 示例：route_id='WS_N' 西进口北出口 
    # movement为0表示该网格无车
    o = inlet_map[object_id[0]]
    d = inlet_map[object_id[3]]
    # 返回值：[进口道编码，转向编码，出口道编码]
    return (o,(d-o)%4-1,d)

# 随机生成恒定交通需求
def generate_demand():
    routes = et.Element('routes')
    tree = et.ElementTree(routes)
    
    # 所有routes
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
    demand_ratio = [1,2,1]  # 左直右流量比，未区分进口道
    demand_basic = 1/3*300.0
    # 需求数据
    demand_data = [np.random.poisson(demand_ratio[get_movement(r)[1]]*demand_basic) for r in route_list]
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

        self.total_timeloss = 0  # 周期延误
        self.depart_num = 0  # 周期内离开检测器的车辆
        # 流向延误:北，东，南，西，左转，直行，右转
        self.total_timeloss_m = 12*[0.0]
        self.depart_num_m = 12*[0]
        self.veh_dict = {}  # 交叉口内车辆，到达时的损失时间，字典，key: vehicle_id, value: 损失时间
    
    def update(self):
        self.prev_veh = self.cur_veh
        self.cur_veh = set(multientryexit.getLastStepVehicleIDs("e3"))

        # 上一时间步到达交叉口的车辆
        for veh_id in (self.cur_veh - self.prev_veh):
            self.veh_dict[veh_id] = vehicle.getTimeLoss(veh_id)
        
        # 上一时间步离开交叉口的车辆
        for veh_id in (self.prev_veh - self.cur_veh):
            timeloss = vehicle.getTimeLoss(veh_id) - self.veh_dict[veh_id]
            self.depart_num += 1
            self.total_timeloss += timeloss
            m = get_movement(veh_id)
            self.total_timeloss_m[3*m[0]+m[1]] += timeloss
            self.depart_num_m[3*m[0]+m[1]] += 1
            self.veh_dict.pop(veh_id)
            
    def output(self):
        delay = 0.0 if self.depart_num==0 else self.total_timeloss/self.depart_num
        
        delay_m = 12*[0.0]
        for i in range(12):
            delay_m[i] = 0.0 if self.depart_num_m[i]==0 else self.total_timeloss_m[i]/self.depart_num_m[i]
        
        # reset
        self.total_timeloss = 0.0
        self.depart_num = 0
        self.total_timeloss_m = 12*[0.0]
        self.depart_num_m = 12*[0]
        return delay,delay_m

# 观测车端数据
class Observer():
    def __init__(self):
        self.id = 'J'   # 路段设备所在交叉口id
        self.max_obs_range = 400.0   # 路端设备观测范围 ,也即数据收集范围
        
        junction.subscribeContext(self.id,
                                  constants.CMD_GET_VEHICLE_VARIABLE,
                                  self.max_obs_range,
                                  [constants.VAR_SPEED,constants.VAR_ROUTE_ID])
    
    def output(self):
        obs = []
        vehicle_info = junction.getContextSubscriptionResults(self.id)
        
        for vehicle_id in vehicle_info:
            lane_id = vehicle.getLaneID(vehicle_id)
            if lane_id[0] == ':':  # 排除越过停车线的交叉口中车辆，主要为不受信控的右转车
                continue
            if lane_id[2] == 'O':  # 排除交叉口附近正在离开的车辆
                continue
            inlet_index = inlet_map[lane_id[0]]
            lane_index = int(lane_id[-1])

            # 横向位置
            lat_pos = vehicle.getLateralLanePosition(vehicle_id)
            
            # 纵向位置, 距交叉口的距离
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

class Visualizer():
    # 仿真实验的可视化
    def __init__(self):
        pass

#%% interaction between server and client
def run(number,cycle_to_run=1):
    cycle_step = 1  # 当前仿真周期的编号
    
    delay_monitor = DelayMonitor()
    observer = Observer()
    tc = BaseTrafficController()
    
    # obs_list_c = []  # 记录周期级观测数据
    # obs_list_p = []   # 记录相位级观测数据
    obs_list_s = [[]]  # 记录秒级观测数据, 二维列表
    delay_list = []  # 记录延误数据
    delay_list_m = []  # 记录流向延误数据
    tc_list = []  # 记录信号控制数据

    warm_up = 3   # 热启动需要的周期数  # 可能需要敏感性分析
    
    pbar = tqdm(total=cycle_to_run+warm_up,desc=f"simulation {number}")
    
    # 按周期数进行仿真
    count = 5
    while(cycle_step<=warm_up+cycle_to_run):
        # 安排下一步
        tc.run()
        # 仿真步进
        traci.simulationStep()
        
        # 热启动后进行的处理
        if cycle_step > warm_up:  
            # 处理：仿真步结束时
            delay_monitor.update()
            count -= 1
            
            # 处理：单位时间(秒)结束时
            if count == 0:
                count = 5
                obs_list_s[-1].append(observer.output())
                
            # 处理：相位即将切换时
            # if tc.phase_time_1 == 0 or tc.phase_time_2 == 0:
            #     obs_list_p.append(observer.output())
            
            # 处理：周期即将切换时
            if tc.cycle_time == 0:
                # obs_list_c.append(observer.output())  # 下一周期的观测
                delay,delay_m = delay_monitor.output()
                delay_list.append(delay)  # 上一周期的延误
                delay_list_m.append(delay_m)  # 上一周期的流向延误
                tc_list.append(tc.output())   # 上一周期的信号控制
                obs_list_s.append([])
        
        if tc.cycle_time == 0:
            cycle_step += 1
            pbar.update(1)
    
    # 对齐观测数据和延误数据
    obs_list_s = obs_list_s[:-1]
    delay_list = delay_list[1:]
    delay_list_m = delay_list_m[1:]
    tc_list = tc_list[1:]
    
    # 结束仿真
    traci.close()
    
    return (obs_list_s,delay_list,delay_list_m,tc_list)

#%% 
step_length = 0.2  # 敏感性分析
sumocfg = ["sumo",
            "--route-files","test.rou.xml,flow.rou.xml",
            "--net-file","test.net.xml",
            "--additional-files","test.e3.xml",
            "--delay","20","--time-to-teleport","600",
            "--step-length",f"{step_length}"]

data_dir = '../../data/synthetic_data/'

#%% 仿真实验
def get_simulation_data(runs,cycle_to_run,data_name):
    obs_data = []
    delay_data = []
    delay_m_data = []
    tc_data = []
    demand_data = []

    for i in range(runs):  # 运行多次仿真，收集数据
        demand_data.append(generate_demand())
        
        try:
            traci.close()
        except:
            pass
        traci.start(sumocfg,port=666)
        
        obs,delay,delay_m,tc = run(i,cycle_to_run=cycle_to_run)
        
        obs_data.append(obs)
        delay_data.append(delay)
        delay_m_data.append(delay_m)
        tc_data.append(tc)

    data = {'obs':obs_data,
            'delay':delay_data,'delay_m':delay_m_data,
            'tc':tc_data,'demand':demand_data}

    # data dimension description
    # obs: (list:sample_num,list:cycle_num,list:cycle_len,list:obs_vehicle, dict:vehicle_info)
    # delay：(list:sample_num,list:cycle_num)
    # delay_m：(list:sample_num,list:cycle_num,list:direction)
    # tc：(list:sample, list:cycle, list:timing params)
    # demand：(list:sample, list:cycle, list:route)

    # 保存
    with open(data_dir+data_name+'.pickle', 'wb') as f:
        pickle.dump(data, f)

#%%
get_simulation_data(2,2,'test')

#%%
# 读取
with open(data_dir+'test'+'.pickle', 'rb') as f:
    data = pickle.load(f)