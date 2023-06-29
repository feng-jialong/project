#%%
# region import
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
# endregion

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()
