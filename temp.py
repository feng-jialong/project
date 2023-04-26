def set_control(control):
    # 设置信号配时方案
    # 处理控制输入
    
    # region NEMA相位的处理
    signal = control['signal']
    phase_info = {}
    if signal['y'][0]:
        phase_info['WT_G'] = 0.0
        phase_info['WT_Y'] = signal['x'][0]
        phase_info['WT_R'] = phase_info['WT_Y'] + signal['x_y']
        phase_info['EL_G'] = phase_info['WT_R'] + signal['x_r']
        phase_info['EL_Y'] = phase_info['EL_G'] + signal['x'][1]
        phase_info['EL_R'] = phase_info['EL_Y'] + signal['x_y']
        phase_info['EL_R'] = phase_info['EL_Y'] + signal['x_y']
    # endregion
    
    # 根据控制的决策变量，生成tlsLogic, 即Logic对象
    # 普通相位方案的导入
    phases = ()
    phases += (trafficlight.Phase(duration=control['x'][0],
                                  state='GrrrGGGrGrrrGGGr'),
               trafficlight.Phase(duration=control['x_y'],
                                  state='GrrrGyyrGrrrGyyr'),
               trafficlight.Phase(duration=control['x_r'],
                                  state='GrrrGrrrGrrrGrrr'))
    
    phases += (trafficlight.Phase(duration=control['x'][1],
                                  state='GrrrGrrGGrrrGrrG'),
               trafficlight.Phase(duration=control['x_y'],
                                  state='GrrrGrryGrrrGrry'),
               trafficlight.Phase(duration=control['x_r'],
                                  state='GrrrGrrrGrrrGrrr'))
    
    phases += (trafficlight.Phase(duration=control['x'][2],
                                  state='GGGrGrrrGGGrGrrr'),
               trafficlight.Phase(duration=control['x_y'],
                                  state='GyyrGrryGyyrGrry'),
               trafficlight.Phase(duration=control['x_r'],
                                  state='GrrrGrrrGrrrGrrr'))

    phases += (trafficlight.Phase(duration=control['x'][3],
                                  state='GrrGGrrGGrrGGrrG'),
               trafficlight.Phase(duration=control['x_y'],
                                  state='GrryGrryGrryGrrr'),
               trafficlight.Phase(duration=control['x_r'],
                                  state='GrrrGrrrGrrrGrrr'))
    
    logic = trafficlight.Logic(programID='0',type=0,currentIndex=0,phases=phases)

    # tlsLogic导入信号灯
    trafficlight.setProgramLogic('J',logic)

# 随机生成控制方案
def generate_control():
    pass
    additional = et.Element("additional")
    tree = et.ElementTree(additional)
    tl_logic = et.SubElement(additional,"tlLogic",
                          {'id':'J','programID':'1','type':'static'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'30','state':'GGGrGrrrGGGrGrrr','name':'G_NS_T'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'3','state': 'GyyrGrrrGyyrGrrr','name':'Y_NS_T'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'1','state':'rrrrrrrrrrrrrrrr','name':'R_NS_T'})
    
    et.SubElement(tl_logic,"phase",
                  {'duration':'30','state':'GrrGGrrrGrrGGrrr','name':'G_NS_L'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'3','state': 'GrryGrrrGrryGrrr','name':'Y_NS_L'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'1','state':'rrrrrrrrrrrrrrrr','name':'R_NS_L'})
    
    et.SubElement(tl_logic,"phase",
                  {'duration':'30','state':'GrrrGGGrGrrrGGGr','name':'G_WE_T'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'3','state': 'GrrrGyyrGrrrGyyr','name':'Y_WE_T'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'1','state':'rrrrrrrrrrrrrrrr','name':'R_WE_T'})
    
    et.SubElement(tl_logic,"phase",
                  {'duration':'30','state':'GrrrGrrGGrrrGrrG','name':'G_WE_L'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'3','state': 'GrrrGrryGrrrGrry','name':'Y_WE_L'})
    et.SubElement(tl_logic,"phase",
                  {'duration':'1','state':'rrrrrrrrrrrrrrrr','name':'R_WE_L'})

    tree.write('test.add.xml')
    pretty_xml = minidom.parse('test.add.xml').toprettyxml(encoding='UTF-8')
    with open('test.add.xml','wb') as file:
        file.write(pretty_xml)

# 管理交通需求
class DemandManager():
    def __init__(self,id_list):
        self.calibrator_list = id_list
        self.demand = {'L':300,'T':600,'R':300}
        
        self.flow_length = 30  # 流量持续的分钟
        
        self.set_demand()
    
    def set_demand(self):
        for calib in self.calibrator_list:
            movement = calib[-1]
            calibrator.setParameter(calib,'jamThreshold','0.0')
            calibrator.setFlow(calib,begin=0.0,end=1200.0,
                            vehsPerHour=self.demand[movement],
                            speed=50.0/3.6,
                            typeID='DEFAULT_VEHTYPE',
                            routeID=calib[:-2],
                            departLane='free',departSpeed='max')

def get_observation():
    grid_length = 4.0  # 道路划分方格的尺寸
    e2_length = 100.0
    lane_div = 1
    lane_num = 4
    fetures = ['vel','movement']
    obs = np.zeros((4,int(e2_length//grid_length)+1,lane_div*lane_num,len(fetures)))
    
    for i,inlet in enumerate(['N','S','W','E']):
        for j,lane_index in enumerate(['0','1','2','3']):
            e2_id = 'e2_'+inlet+'_'+lane_index
            for vehicle_id in lanearea.getLastStepVehicleIDs(e2_id):
                lane_id = vehicle.getLaneID(vehicle_id)
                if lane_id[-3] == 'P':
                    grid_index = (lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id)))//grid_length
                else:
                    grid_index = lane.getLength(lane_id) - (vehicle.getLanePosition(vehicle_id) - vehicle.getLength(vehicle_id))//grid_length
                grid_index = (e2_end - (vehicle.getLanePosition(vehicle_id) - 0.5*vehicle.getLength(vehicle_id)))//grid_length
                grid_index = int(grid_index)
                print(grid_index)
                obs[i,grid_index,j,:] = [vehicle.getSpeed(vehicle_id),get_movement(vehicle_id)]
    return obs

# region 仿真状态存取
# 读取交通状态
def save_state(num):
    save_path = "P:\学校相关\第八学期\毕业设计\project\SUMO\test"
    # 保存仿真状态
    traci.simulation.saveState(save_path+'\\'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())+f'{num}'+'.xml')
    pass

# 载入交通状态
def load_state(file_path):
    traci.simulation.loadState(file_path)
    pass
# endregion

# 监测周期的切换
class CycleMonitor():
    def __init__(self):
        self.cycle_step = 0  # 周期序号
        self.prev_cycle = 0.0  # 上一步，在周期中的位置
        self.cur_cycle = 0.0  # 这一步，在周期中的位置
        self.is_switching = False
    
    def update(self):
        self.prev_cycle = self.cur_cycle  # 本周期剩下的时间：上一步
        self.cur_cycle = float(trafficlight.getParameter('J','cycleSecond'))  # 本周期剩下的时间：这一步
        
        # 周期是否切换
        if self.cur_cycle < self.prev_cycle:
            self.cycle_step += 1
            self.is_switching = True
        else:
            self.is_switching = False
    
    def output(self):
        return self.is_switching