#%%
# region import
import os,sys,time
import numpy as np
import joblib
from tqdm import tqdm
import importlib as imp
from multiprocessing import Pool

import traci
from traci import vehicle
from traci import lane
from traci import lanearea,multientryexit
from traci import junction
from traci import constants

# 导入与重载自定义模块
sys.path.append("../../models/")
import traffic_controller
try:
    imp.reload(traffic_controller)
except:
    pass
import vehicle_generator
try:
    imp.reload(vehicle_generator)
except:
    pass
import mpc_controller
try:
    imp.reload(mpc_controller)
except:
    pass
import delay_predictor
try:
    imp.reload(delay_predictor)
except:
    pass
import utils
try:
    imp.reload(utils)
except:
    pass
from traffic_controller import BaseTrafficController
from vehicle_generator import VehicleGenerator
from mpc_controller import MPCController
from delay_predictor import DelayPredictor,resume
from utils import get_movement,inlet_map,try_connect

# endregion

# region configuration
STEP_LENGTH = 0.5
TIME_INTERVAL = 1.0
STEP_NUM = int(TIME_INTERVAL/STEP_LENGTH)
WARM_UP = 900  # 热启动需要的时间

GRID_LENGTH = 2.0
OBS_RANGE = 200.0
LANE_NUM = 4  # hardcode
GRID_NUM = int(OBS_RANGE//GRID_LENGTH)+3

sumocfg = ["sumo-gui",
            "--route-files","test.rou.xml",
            "--net-file","test.net.xml",
            "--additional-files","test.e2.xml,test.e3.xml",
            "--gui-settings-file","gui.cfg",
            "--delay","0",
            "--time-to-teleport","600",
            "--step-length",f"{STEP_LENGTH}",
            "--no-step-log","true",
            "--quit-on-end"]

# 仿真采样数据保存路径
save_dir = '../../data/simulation_data/test/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

MODE = 'sample'
# if MODE=='experiment':
#     sumocfg += ["--seed","0"]  # 固定SUMO的随机数种子，如期望速度随机性和实际速度随机性
# endregion

#%%
class Monitor():
    def __init__(self):
        self.prev_veh = set()
        self.cur_veh = set()

        self.total_timeloss = 0  # 周期延误
        self.depart_num = 0  # 周期内离开检测器的车辆
        # 流向延误:北，东，南，西，左转，直行，右转
        self.total_timeloss_m = np.array(12*[0.0])
        self.depart_num_m = np.array(12*[0])
        # 交叉口内车辆，到达时的损失时间 
        # # 字典，key: vehicle_id, value: 损失时间
        self.veh_dict = {}
        # 流向流量
        self.update_freq = 5*60 # 流量统计的更新频率5min
        self.veh_count_m = np.ones((4,3),dtype=int)  # 流向车辆计数
        self.vph_m = np.zeros((4,3))  # 流向小时流率
        
        self.t = 0  # 时间单位(s)
        
        self.queue_length = None
        # 订阅排队长度检测器的last_step_vehice_number
        e2_id_list = lanearea.getIDList()
        for e2_id in e2_id_list:
            if e2_id[0] == 'm':
                lanearea.subscribe(e2_id,[constants.JAM_LENGTH_METERS])
                
        self.queue_length_list = []
    
    def run(self):
        # 更新流量估计，重置到达计数
        if self.t % self.update_freq == 0:
            alpha = 1.0
            self.vph_m *= (1-alpha)
            self.vph_m += alpha*self.veh_count_m/(self.update_freq/3600.0)
            self.veh_count_m = np.ones((4,3),dtype=int)
        
        e2_results = lanearea.getAllSubscriptionResults()
        alpha = 1.0
        # 获取所有排队信息
        self.queue_length = np.array(16*[0.0])
        for key,result in e2_results.items():
            if key[0] == 'm':
                current_length = result[constants.JAM_LENGTH_METERS]
                self.queue_length[int(key[4:])] *= (1.0-alpha)
                self.queue_length[int(key[4:])] += alpha*current_length
         
        self.queue_length_list.append(self.queue_length)
        self.t += 1
        
    def step(self):
        self.prev_veh = self.cur_veh
        self.cur_veh = set(multientryexit.getLastStepVehicleIDs("e3"))

        # 上一时间步到达交叉口的车辆
        for veh_id in (self.cur_veh - self.prev_veh):
            self.veh_dict[veh_id] = vehicle.getTimeLoss(veh_id)
            
            # 到达计数
            o,turn,_ = get_movement(veh_id)
            self.veh_count_m[o,turn] += 1
        
        # 上一时间步离开交叉口的车辆
        for veh_id in (self.prev_veh - self.cur_veh):
            timeloss = vehicle.getTimeLoss(veh_id) - self.veh_dict[veh_id]
            self.depart_num += 1
            self.total_timeloss += timeloss
            o,turn,d = get_movement(veh_id)
            self.total_timeloss_m[3*o+turn] += timeloss
            self.depart_num_m[3*o+turn] += 1
            self.veh_dict.pop(veh_id)
    
    def output(self):
        delay = 0.0 if self.depart_num==0 else self.total_timeloss/self.depart_num 
        
        delay_m = np.array(12*[0.0])
        for i in range(12):
            delay_m[i] = 0.0 if self.depart_num_m[i]==0 else self.total_timeloss_m[i]/self.depart_num_m[i]
            
        queue = np.array(self.queue_length_list)  # 周期内每秒的排队长度, (周期内各秒，各个流向)

        return delay, delay_m, queue, self.vph_m
    
    def reset(self):
        self.total_timeloss = 0.0
        self.depart_num = 0
        self.total_timeloss_m = 12*[0.0]
        self.depart_num_m = 12*[0]
        self.queue_length_list = []
        
class Observer():
    def __init__(self):
        self.id = 'J'   # 路段设备所在交叉口id
        self.max_obs_range = OBS_RANGE   # 路端设备观测范围 ,也即数据收集范围
        
        junction.subscribeContext(self.id,
                                  constants.CMD_GET_VEHICLE_VARIABLE,
                                  self.max_obs_range,
                                  [constants.VAR_SPEED,constants.VAR_ROUTE_ID])
        
    def output(self):
        obs = np.zeros((1+1,4,GRID_NUM,LANE_NUM),dtype=np.float32)
        obs[1] -= 1.0
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
            # lat_pos = vehicle.getLateralLanePosition(vehicle_id)
            
            # 纵向位置, 距交叉口的距离
            if lane_id[-3] == 'P':   # 在不可变道进口段
                lon_pos = (lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id)))
            else:   # 在可变道进口段
                lon_pos = lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id))
                lon_pos += lane.getLength(lane_id[:5] + 'P_' + lane_id[-1])

            grid_index = int(lon_pos//GRID_LENGTH)
            
            # 速度
            speed = vehicle.getSpeed(vehicle_id)
            # 流向
            move = get_movement(vehicle_id)[1]
            
            # 速度
            obs[0,inlet_index,grid_index,lane_index] = speed
            # 导向
            obs[1,inlet_index,grid_index,lane_index] = move
            
        return obs

class LaneSelector():
    def __init__(self):
        # 订阅车道选择点检测器的last_step_vehicle_id_list
        e2_id_list = lanearea.getIDList()
        for e2_id in e2_id_list:
            if e2_id[0] == 'e':
                lanearea.subscribe(e2_id,[constants.LAST_STEP_VEHICLE_ID_LIST])

        self.lc_cool_down = {}
    
    def update(self,tc):
        if tc.control['switch'].any():
            for key,_ in self.lc_cool_down.items():
                # 1: 冷却完毕
                self.lc_cool_down[key] = 1
        
    def run(self,tc,monitor):
        # 车辆的可变车道决策
        # for key,value in self.lc_cool_down.items():
        #     # 目标车道变更冷却
        #     if value > 0:
        #         self.lc_cool_down[key] -= 1

        # 流向可选车道No.
        now_lane = tc.lane_func
        
        # 流向最优车道index
        best_lane_index = np.zeros((4,3))
        for i in range(4):
            for j in range(3):
                best_index = monitor.queue_length[now_lane[i][j]].argmin()
                best_lane_index[i][j] = now_lane[i][j][best_index]%4
        
        e2_results = lanearea.getAllSubscriptionResults()
        # 设置车辆的目标车道
        for key,result in e2_results.items():
            if key[0] == 'e':
                # 避免车辆在两个车道之间反复横跳, 更换目标车道需要冷却
                # 速度较低时无法更换目标车道
                for veh_id in result[constants.LAST_STEP_VEHICLE_ID_LIST]:
                    o,turn,_ = get_movement(veh_id)
                    target_lane_index = best_lane_index[o][turn]
                    if veh_id not in self.lc_cool_down.keys():  # 不在冷却表中
                        self.lc_cool_down[veh_id] = 1
                    if vehicle.getLaneIndex(veh_id) != target_lane_index:  # 需要换道
                        if self.lc_cool_down[veh_id] == 1:  # 是否冷却完毕
                            if vehicle.getSpeed(veh_id) > 3.0:  # 是否仍在行进
                                vehicle.changeLane(veh_id,target_lane_index,3600.0)
                                # 0: 进入冷却
                                self.lc_cool_down[veh_id] = 0
                            
class Recorder():
    def __init__(self,mode):
        self.obs_list = []  # 记录秒级观测数据
        self.obs_c = []
        self.delay_list = []  # 记录延误数据
        self.delay_list_m = []  # 记录流向延误数据
        self.tc_list = []  # 记录信号控制数据
        self.time_point = []
        self.queue_list = []
        self.vph_m_list = []
        self.mode = mode
    
    def run(self,observer):
        self.obs_c.append(observer.output())  # 获取每秒的观测数据
    
    def update(self,monitor,tc):
        delay,delay_m,queue,vph_m = monitor.output()
        self.queue_list.append(queue)
        self.vph_m_list.append(vph_m)
        self.delay_list.append(delay)  # 上一周期的延误
        self.delay_list_m.append(delay_m)  # 上一周期的流向延误
        self.tc_list.append(tc.output())   # 上一周期的信号控制
        
        self.obs_c = np.stack(self.obs_c,axis=0)
        self.obs_list.append(self.obs_c)
        self.obs_c = []
        
        self.time_point.append(monitor.t)  # 记录的时间点
        
    def save(self,save_name,index):
        # 保存<仿真数据>
        if self.mode=="sample":
            data = (self.obs_list,np.array(self.delay_list),
                    np.stack(self.delay_list_m,axis=0),self.tc_list)
            with open(save_dir+save_name+'_'+str(index)+'.pkl', 'wb') as f:
                joblib.dump(data, f)
        elif self.mode=="experiment":
            pass

class Clock():
    def __init__(self,step_num,warm_up,cycle_to_run,time_to_run):
        self.cycle_step = -1  # 已完成的周期数
        self.time = 0  # 已经过的时间
        
        self.step_num = step_num
        self.count = step_num
        self.warm_up = warm_up
        
        self.cycle_to_run = cycle_to_run
        self.time_to_run = time_to_run
    
    def step(self):
        self.count -= 1
        
    def run(self):
        self.count = self.step_num
        self.time += 1
    
    def update(self):
        self.cycle_step += 1
        
    def is_warm(self):
        return self.time >= self.warm_up
    
    def is_clear(self):
        return self.cycle_step >= 0
    
    def is_end(self,mode):
        if mode=='sample':
            return self.cycle_step>=self.cycle_to_run
        elif mode=='experiment':
            return self.time > self.time_to_run
    
    def is_to_run(self):
        return self.count == 0

def run_experiment(index,cycle_to_run,time_to_run):
    mode = 'experiment'
    np.random.seed(0)
    try_connect(8,sumocfg)

    clock = Clock(STEP_NUM,WARM_UP,cycle_to_run,time_to_run)
    monitor = Monitor()
    observer = Observer()
    
    # 神经网络代理模型的超参数
    batch_size = 16
    lookback = 2
    lookahead = 4
    model_dir = '../../results/test-new/random_seed/standard/'
    mpc_model = DelayPredictor(model_dir,batch_size,lookback,lookahead)
    resume(mpc_model)
    mpc_controller = MPCController(mpc_model,'all')  # 还要选择模式
    
    tc = BaseTrafficController(STEP_LENGTH,TIME_INTERVAL)
    veh_gen = VehicleGenerator(mode='static')
    
    selector = LaneSelector()
    recorder = Recorder(mode)
    
    pbar = tqdm(total=cycle_to_run,desc=f"simulation {index}")
    
    # 热启动完毕且当前周期结束时正式开始记录数据
    # 按周期数进行仿真
    while(not clock.is_end(mode)):
        # 规划: 仿真步开始前
        tc.step()
        # 仿真步进
        traci.simulationStep()
        
        # 处理：仿真步结束时
        if clock.is_clear() and clock.is_warm():  # warm up结束后
            pass
        monitor.step()
        clock.step()
        
        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            if clock.is_clear() and clock.is_warm():  # warm up结束且清空周期后
                recorder.run(observer)
            veh_gen.run()  # 车辆生成器根据schedule生成车辆
            monitor.run()
            selector.run(tc,monitor)
            clock.run()
                    
        # 处理：周期即将切换时
        if tc.is_to_update():
            if clock.is_clear() and clock.is_warm():  # warm up结束且清空周期后
                recorder.update(monitor,tc)
                mpc_controller.update(monitor,recorder,tc)
                pbar.update(1)
                
            if clock.is_warm():  # warm up结束后
                clock.update()
            
            # 控制方案更新
            if mpc_controller.warmup == 0 and True:  # mpc
                # 使用MPC
                tc.update('mpc',monitor,mpc_controller)
            elif clock.time > monitor.update_freq:  # 更新webster
                # 使用webster配时进行控制
                # 普通Webster
                tc.update('webster',monitor)
                # 上帝Webster
                # monitor.vph_m[:] = veh_gen.vph_m[:]
                # tc.update('webster',monitor)
            else:   # 仿真初期使用随机控制方案进行预热
                # 使用随机方案
                tc.update('random')
                    
            selector.update(tc)
            monitor.reset()  # 重置性能指标监测器
    
    # 结束仿真
    traci.close()
    
    if mode=='experiment':
        return recorder,mpc_controller,mpc_model

def run_sample(index,cycle_to_run,time_to_run,save_name):
    mode = 'sample'
    np.random.seed()  # 多进程跑数据，随机设置种子
        
    try_connect(8,sumocfg)

    clock = Clock(STEP_NUM,WARM_UP,cycle_to_run,time_to_run)
    monitor = Monitor()
    observer = Observer()
    
    tc = BaseTrafficController(STEP_LENGTH,TIME_INTERVAL)
    veh_gen = VehicleGenerator(mode='linear')
        
    selector = LaneSelector()
    recorder = Recorder(mode)
    
    pbar = tqdm(total=cycle_to_run,desc=f"simulation {index}")
    
    # 热启动完毕且当前周期结束时正式开始记录数据
    # 按周期数进行仿真
    while(not clock.is_end(mode)):
        # 规划: 仿真步开始前
        tc.step()
        # 仿真步进
        traci.simulationStep()
        
        # 处理：仿真步结束时
        if clock.is_clear() and clock.is_warm():  # warm up结束后
            pass
        monitor.step()
        clock.step()
        
        # 处理：单位时间(秒)结束时
        if clock.is_to_run():
            if clock.is_clear() and clock.is_warm():  # warm up结束且清空周期后
                recorder.run(observer)
            veh_gen.run()  # 车辆生成器根据schedule生成车辆
            monitor.run()
            selector.run(tc,monitor)
            clock.run()
                    
        # 处理：周期即将切换时
        if tc.is_to_update():
            if clock.is_clear() and clock.is_warm():  # warm up结束且清空周期后
                recorder.update(monitor,tc)
                pbar.update(1)
                
            if clock.is_warm():  # warm up结束后
                clock.update()
            
            # 控制方案更新
            if clock.time > monitor.update_freq:
                # 使用webster配时进行采样
                tc.update('sample',monitor)
            else:
                # 使用随机方案
                tc.update('random')
            selector.update(tc)
            monitor.reset()  # 重置性能指标监测器
            # 排队长度大于200m，严重拥堵，终止仿真
            if (monitor.queue_length>200.0).any():
                break
    
    # 结束仿真
    traci.close()
    
    recorder.save(save_name,index)

#%% 多进程仿真，获取数据
if MODE=='sample':
    RUN_NUM = 1
    CYCLE_NUM = 100
    TIME_TO_RUN = 12000.0
elif MODE=='experiment':
    CYCLE_NUM = 666
    TIME_TO_RUN = 6000.0

def wrapper(index):
    run_sample(index,CYCLE_NUM,TIME_TO_RUN,'simulation_data')

def main():
    with Pool(8) as pool:
        pool.map(wrapper,[i for i in range(RUN_NUM)])
        pool.close()
        pool.join()

if __name__ == "__main__":
    if MODE=='sample':
        # 多进程会各复制一份本文件，并且完全执行，需要把其他脚本部分注释掉
        main()
    elif MODE=='experiment':
        try:
            traci.close()
        except:
            pass
        recorder,mpc_controller,mpc_model = run_experiment(0,CYCLE_NUM,TIME_TO_RUN)

        # 保存画图需要的东西
        # with open(save_dir+"test.pkl",'wb') as f:
        #     joblib.dump((recorder,
        #                  mpc_controller.predict_result,
        #                  mpc_controller.control_result,
        #                  mpc_controller.upper_context,
        #                  mpc_model), f)

#%% 仿真数据保存格式探索
with open("../../data/simulation_data/test/simulation_data_0.pkl",'rb') as f:
    data = joblib.load(f)