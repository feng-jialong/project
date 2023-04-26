#%% 
# region import
import os,sys
import numpy as np

# check environment varialble 'SUMO_HOME'
if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'],'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")
# endregion