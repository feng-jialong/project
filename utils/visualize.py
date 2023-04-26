#%%
import pickle

import numpy as np
import pandas as pd

import sklearn

import matplotlib as mpl
import matplotlib.pyplot as plt

#%% 
with open('data/synthetic_data/single_cycle.pickle','rb') as f:
    data = pickle.load(f)

plt.hist(np.array(data['delay']).flatten(),density=True)
#%% 

