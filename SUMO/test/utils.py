#
# import xml.etree.ElementTree as et
# from xml.dom import minidom
import numpy as np
import traci
import sys,os,time
import torch
from sklearn.preprocessing import OneHotEncoder

# 映射到编号
inlet_map = {'N':0,'E':1,'S':2,'W':3}
direction_map = {'L':0,'T':1,'R':2}

# 根据车辆的路径id获取信息
def get_movement(object_id):
    # object_id为vehicle_id或route_id
    # 示例：route_id='WS_N' 西进口北出口 
    # movement为0表示该网格无车
    o = inlet_map[object_id[0]]
    d = inlet_map[object_id[3]]
    # 返回值：[进口道编号，转向编号，出口道编号]
    return (o,(d-o)%4-1,d)

def try_connect(num,sumocfg):
    for _ in range(num):
        try:
            traci.start(sumocfg)
            break
        except:
            time.sleep(0.5)
            
def control2predict(control):
    # input: <控制格式>或<紧凑控制格式>的控制方案
    # output: <预测格式>的控制方案
    tc_p = torch.from_numpy(control['phase']).to(torch.float32)
    if len(tc_p.shape)==1:
         tc_p = tc_p.reshape(1,-1)
    tc_g = torch.from_numpy(control['split']).to(torch.float32)
    if len(tc_g.shape)==1:
         tc_g = tc_g.reshape(1,-1)
         
    tc_t = control['target_func']
    if len(tc_t.shape)==1:
         tc_t = tc_t.reshape(1,-1)
    one_hot = OneHotEncoder(categories=8*[[0,1,2]],sparse_output=False)
    tc_t = torch.from_numpy(one_hot.fit_transform(tc_t)).to(torch.float32)
    
    return torch.cat([tc_p,tc_g,tc_t],dim=-1)