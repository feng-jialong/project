#%%
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

data_path = "data\\Peachtree-Street-Atlanta-GA\\NGSIM_Peachtree_Vehicle_Trajectories.csv"

data = pd.read_csv(data_path)

#%%
def feet2meters(feet):
    return feet/3.2808399

#%%
veh_68 = data.query('Vehicle_ID==68').query('Int_ID==1.0')
plt.plot('Global_X','Global_Y',data=veh_68)
plt.plot('Local_X','Local_Y',data=veh_68)
plt.plot('Lane_ID','Local_Y',data=veh_68)

#%%
data_veh_grouped = data.query('Int_ID==1.0 and Direction==1.0').groupby('Vehicle_ID')
fig, ax = plt.subplots()
for _,veh in data_veh_grouped:
    ax.plot("Global_X","Global_Y",data=veh.sort_values('Frame_ID'))
    
#%%
veh_2 = data.query('Vehicle_ID==2')
plt.plot('Global_X','Global_Y',data=veh_2)
plt.plot('Local_X','Local_Y',data=veh_2)
plt.plot('Lane_ID','Local_Y',data=veh_2)