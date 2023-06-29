#%%
# region import 
import sys,os
import numpy as np
import torch
from tqdm import tqdm
import joblib
from joblib import Parallel,delayed
from sklearn.preprocessing import OneHotEncoder

GRID_LENGTH = 2.0
OBS_RANGE = 200.0
LANE_NUM = 4  # hardcode
GRID_NUM = int(OBS_RANGE//GRID_LENGTH)+3

LOOKBACK = 4
LOOKAHEAD = 4
# endregion

#%% 仿真数据整合与处理
def agg_sim_data(data_dir,output_dir):
    agg_freq = 1  # 一个仿真文件处理一次
    data = {'obs':[],'delay':[],'delay_m':[],'tc':[]}
    pbar = tqdm(total=len(os.listdir(data_dir)),desc=f"loading ")
    chunk_index = 0
    sample2chunk = []
    for i,file in enumerate(os.listdir(data_dir)):
        with open(data_dir+file,'rb') as f:
            try:
                data_p = joblib.load(f)
            except EOFError:
                pbar.update(1)
                print(file)
                continue
        pbar.update(1)
        # 舍弃周期数过少的仿真序列
        if len(data_p[0]) < 10:
            print(file)
            continue
        # 裁剪
        if len(data_p[0]) > 60:
            data_p = list(data_p)
            data_p[0] = data_p[0][:60]
            data_p[1] = data_p[1][:60]
            data_p[2] = data_p[2][:60]
            data_p[3] = data_p[3][:60]
        data['obs'].append(data_p[0])
        data['delay'].append(data_p[1])
        data['delay_m'].append(data_p[2])
        data['tc'].append(data_p[3])
        if len(data['obs']) == agg_freq:   # 流式处理
            sample_num = pre_sim_data(chunk_index,data,output_dir)
            temp = [sample_num*[chunk_index],list(range(sample_num))]
            temp = list(zip(*temp))
            sample2chunk.extend(temp)
            chunk_index += 1
            data = {'obs':[],'delay':[],'delay_m':[],'tc':[]}
    # 末尾数据
    if len(data['obs']) != 0:
        sample_num = pre_sim_data(chunk_index,data,output_dir)
        temp = [sample_num*[chunk_index],list(range(sample_num))]
        temp = list(zip(*temp))
        sample2chunk.extend(temp)

    with open(output_dir+'sample2chunk.pkl','wb') as f:
        joblib.dump(sample2chunk, f)

def pre_sim_data(chunk_index,data,output_dir):    
    o = obs_process(data['obs'])

    d,d_m = delay_process(data['delay'],data['delay_m'])

    tc = tc_process(data['tc'])
    
    # 时窗数据样本数
    sample_num = len(o)
    torch.save({'obs':o,'delay':d,'delay_m':d_m,'tc':tc},
                output_dir+'chunk_'+str(chunk_index)+'.pth')
    
    return sample_num

def obs_process(obs_data):
    # input: (sample:list,cycle:list,frames:array)
    # output: (sample:2d-array,cycle:2d-array,frames:tensor)
    sample_num = len(obs_data)

    # 多进程，进程中的变量会丢失
    # 所以要么放到硬盘(写入文件),要么返回值(结果列表)
    if sample_num>1:
        obs = Parallel(n_jobs=8)(delayed(par_obs_process)(i,sample) for (i,sample) in enumerate(obs_data))
    else:
        obs = [par_obs_process(0,obs_data[0])]
    
    # lookback
    o = []
    for i in range(sample_num):
        obs_sample = obs[i]
        cycle_num = len(obs_sample)
        for j in range(LOOKBACK,cycle_num-LOOKAHEAD+1):
            o.append(obs_sample[j-LOOKBACK:j])
    # object array: unequal size in some dims
    o_temp = o
    o = np.empty((len(o_temp),LOOKBACK),dtype=object)
    # o: (sample,lookback,frames!,*)
    for i,sample in enumerate(o_temp):
        for j,cycle in enumerate(sample):
            o[i,j] = cycle
            
    return o

def par_obs_process(i,sample):
    # input: (cycle:list,frames:array)
    # output: (cycle:list,frames:array)
    sample = sample[:-1]  # 去除末尾空位
    obs_i = []
    one_hot = OneHotEncoder(categories=(4*GRID_NUM*LANE_NUM)*[[-1.0,0.0,1.0,2.0]],
                            sparse_output=False)
    for _,cycle in enumerate(sample):
        obs_c = frame_process(one_hot,cycle)
        # 观测数据tensor的二维列表
        obs_i.append(obs_c)
    
    return obs_i

def frame_process(one_hot,cycle):
    # input: frame list
    # output: tensor
    # 车端数据的网格划分与one-hot编码
    if one_hot is None:
        one_hot = OneHotEncoder(categories=(4*GRID_NUM*LANE_NUM)*[[-1.,0.,1.,2.]],
                                sparse_output=False)
    frame_num = len(cycle)  # 周期的帧数量
    obs_speed = cycle[:,0]
    obs_move = cycle[:,1]
    
    obs_move = one_hot.fit_transform(obs_move.reshape(frame_num,-1))
    obs_move = obs_move.reshape(frame_num,4,GRID_NUM,LANE_NUM,4)
    obs_speed = obs_speed.reshape(frame_num,4,GRID_NUM,LANE_NUM,1)
    
    obs_c = np.concatenate([obs_move,obs_speed],axis=-1)
    obs_c = np.moveaxis(obs_c,-1,1).reshape(frame_num,20,GRID_NUM,LANE_NUM)
    obs_c = torch.from_numpy(obs_c).to(torch.float32)

    return obs_c  # Tensor: (frame_num,20,GRID_NUM,LANE_NUM)

def delay_process(delay_data,delay_m_data):
    # intput d: (sample:list,cycle:list,1), d_m: (sample:list,cycle:list,12)
    # output d: tensor(sample,lookahead), d_m: tensor(sample,12)
    d = []
    d_m = []
    sample_num = len(delay_data)
    for i in range(sample_num):
        delay_sample = delay_data[i]
        delay_m_sample = delay_m_data[i]
        cycle_num = len(delay_sample)
        # lookahead
        for j in range(LOOKBACK,cycle_num-LOOKAHEAD+1):
            d.append(delay_sample[j:j+LOOKAHEAD])
            d_m.append(delay_m_sample[j])
    
    d = np.array(d).reshape(-1,LOOKAHEAD)
    d = torch.from_numpy(d).to(torch.float32)
    d_m = np.array(d_m).reshape(-1,12)
    d_m = torch.from_numpy(d_m).to(torch.float32)
    
    return d,d_m

def tc_process(tc_data):
    # input: (sample:list,cycle:list,control:dict)
    # output: tensor(sample,lookback+lookahead,37)
    tc_p = []
    tc_g = []
    tc_t = []
    # 2023.5.9: 注意，目前功能在周期开始时完全切换，故switch变量无影响
    # tc_s = []
    sample_num = len(tc_data)
    for i in range(sample_num):
        sample = tc_data[i]
        cycle_num = len(sample)
        # LOOKBACK
        # tc: (sample_num,cycle_num,dict,ndarray)
        for j in range(LOOKBACK,cycle_num-LOOKAHEAD+1):
            tc_p.append([sample[k]['phase'] for k in range(j-LOOKBACK,j+LOOKAHEAD)])
            tc_g.append([sample[k]['split'] for k in range(j-LOOKBACK,j+LOOKAHEAD)])
            tc_t.append([sample[k]['target_func'] for k in range(j-LOOKBACK,j+LOOKAHEAD)])
            # tc_s.append([sample[k]['switch'] for k in range(j-LOOKBACK,j+LOOKAHEAD)])
    
    tc_p = np.array(tc_p).reshape(-1,LOOKBACK+LOOKAHEAD,5)
    tc_g = np.array(tc_g).reshape(-1,LOOKBACK+LOOKAHEAD,8)
    
    tc_t = np.array(tc_t).reshape(-1,8)
    # target_func的one-hot编码器
    one_hot = OneHotEncoder(categories=8*[[0,1,2]],sparse_output=False)
    tc_t = one_hot.fit_transform(tc_t).reshape(-1,LOOKBACK+LOOKAHEAD,8*3)
    
    # tc_s = np.array(tc_s).reshape(-1,8)
    # switch的one-hot编码器
    # one_hot = OneHotEncoder(categories=8*[list(range(9))],sparse_output=False)
    # tc_s = one_hot.fit_transform(tc_s).reshape(-1,LOOKBACK+LOOKAHEAD,8*9)
    
    tc = np.concatenate([tc_p,tc_g,tc_t],axis=-1)
    tc = torch.from_numpy(tc).to(torch.float32)
    
    return tc

#%%
if __name__ == "__main__":
    data_dir = '../data/simulation_data/first/'
    output_dir = '../data/training_data/first/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    agg_sim_data(data_dir,output_dir)