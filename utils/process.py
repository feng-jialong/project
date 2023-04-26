#%%
# region import 
import numpy as np
import pickle
# endregion

#%% 单周期数据
# processing——网格划分
data_dir = 'data/synthetic_data/'
data_name = 'single_cycle_2'

with open(data_dir+data_name+'.pickle', 'rb') as f:
    data = pickle.load(f)

def single_cycle_process(data,grid_length,obs_range,name):
    # hparams
    grid_length = grid_length
    obs_range = obs_range
    grid_num = int(obs_range//grid_length)+3
    sample_num = len(data['obs'])
    lane_num = 4  # hardcode

    # obs
    obs_speed = np.zeros((sample_num,4,grid_num,lane_num),
                        dtype=np.float32)
    obs_move = np.zeros((sample_num,4,grid_num,lane_num),
                        dtype='<U1')
    for i,sample in enumerate(data['obs']):
        for vehicle in sample:
            if vehicle['lon_pos'] <= obs_range:
                grid_index = int(vehicle['lon_pos']//grid_length)
                inlet_index = vehicle['inlet_index']
                lane_index = vehicle['lane_index']
                
                obs_speed[i,inlet_index,grid_index,lane_num-lane_index-1] = vehicle['speed']
                obs_move[i,inlet_index,grid_index,lane_num-lane_index-1] = vehicle['move']
    # delay
    delay = np.array(data['delay'])
    # ts
    ts = np.array(data['ts'])
    # demand
    demand = np.array(data['demand'])
    
    # 输出ndarray的格式
    np.savez(data_dir+name+'.npz',
            obs_speed=obs_speed,obs_move=obs_move,delay=delay,ts=ts,demand=demand)

single_cycle_process(data,2,200.0,data_name+'_grid_c2')

#%% 多周期数据
def multiple_cycle_process(data,grid_length,obs_range,name):
    # processing——网格
    # hparams
    grid_length = grid_length
    obs_range = obs_range
    grid_num = int(obs_range//grid_length)+3
    sample_num = len(data['obs'])
    cycle_num = len(data['obs'][0])
    lane_num = 4  # hardcode

    # obs
    obs_speed = np.zeros((sample_num,cycle_num,4,grid_num,lane_num),
                        dtype=np.float32)
    obs_move = np.zeros((sample_num,cycle_num,4,grid_num,lane_num),
                        dtype='<U1')
    for i,sequence in enumerate(data['obs']):
        for j,sample in enumerate(sequence):
            for vehicle in sample:
                if vehicle['lon_pos'] <= obs_range:
                    grid_index = int(vehicle['lon_pos']//grid_length)
                    inlet_index = vehicle['inlet_index']
                    lane_index = vehicle['lane_index']
                    
                    obs_speed[i,j,inlet_index,grid_index,lane_num-lane_index-1] = vehicle['speed']
                    obs_move[i,j,inlet_index,grid_index,lane_num-lane_index-1] = vehicle['move']
    # delay
    delay = np.array(data['delay'])
    # ts
    ts = np.array(data['ts'])
    # demand
    demand = np.array(data['demand'])
    
    # 输出ndarray的格式
    np.savez(data_dir+name+'.npz',
            obs_speed=obs_speed,obs_move=obs_move,delay=delay,ts=ts,demand=demand)

data_dir = 'data/synthetic_data/'
data_name = 'multiple_cycle_2'

with open(data_dir+data_name+'.pickle', 'rb') as f:
    data = pickle.load(f)

multiple_cycle_process(data,2.0,200.0,data_name+'_first')
# region data dimension description
# obs: (list:sample, list:cycle, grid_num, lane_num)
# delay：(list:sample, list:cycle)
# ts：(list:sample, list:cycle, list:timing params)
# demand：(list:sample, list:cycle, list:route)
# endregion

#%%
load_name = 'single_cycle_grid_c2'
data = np.load('data/synthetic_data/'+load_name+'.npz')

# %%
# single cycle data from two cycle data
data_new  = {key:value[:,0] for key,value in data.items()}
np.savez('data/synthetic_data/'+'two_cycle_grid_c2_cycle1'+'.npz',**data_new)
