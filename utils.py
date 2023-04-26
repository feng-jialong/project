#%%
# region import
import numpy as np
import torch
from torch.utils.data import Dataset,random_split,DataLoader,ConcatDataset
from sklearn.preprocessing import OneHotEncoder
# endregion

class SingleData(Dataset):
    def __init__(self,data_name):
        
        data_dir = 'data/synthetic_data/'
        data = np.load(data_dir+data_name+'.npz')
        # 延误
        self.delay = torch.from_numpy(data['delay']).reshape(-1,1).to(torch.float32)
        self.sample_num = self.delay.shape[0]
        # 速度
        self.speed = torch.from_numpy(data['obs_speed']).reshape(self.sample_num,-1).to(torch.float32)

        # 导向,one-hot
        move_data = data['obs_move'].reshape(self.sample_num,-1)
        one_hot_data = OneHotEncoder(categories=move_data.shape[1]*[['','L','T','R']],sparse=False).fit_transform(move_data)
        self.move = torch.from_numpy(one_hot_data).to(torch.float32)

        # 配时
        self.ts = torch.from_numpy(data['ts']).reshape(self.sample_num,4).to(torch.float32)
    
    def __getitem__(self,i):
        return {'delay':self.delay[[i]],'obs_speed':self.speed[[i]],
                'obs_move':self.move[[i]],'ts':self.ts[[i]]}  # 保持维度结构
    
    def __len__(self):
        return self.sample_num

class MultipleData(Dataset):
    def __init__(self,data_name):
        
        data_dir = 'data/synthetic_data/'
        data = np.load(data_dir+data_name+'.npz')
        
        # seq_dim, sample_dim, feature_dim
        
        # 获取数据维度
        sample_num = data['delay'].shape[0]
        cycle_num = data['delay'].shape[1]
        self.sample_num = sample_num
        self.cycle_num = cycle_num
        
        # 延误
        self.delay = torch.from_numpy(data['delay']).reshape(sample_num,cycle_num,1).to(torch.float32).transpose(0,1)
        
        # 速度
        self.speed = torch.from_numpy(data['obs_speed']).reshape(sample_num,cycle_num,-1).to(torch.float32).transpose(0,1)

        # 导向,one-hot
        move_data = data['obs_move'].reshape(sample_num*cycle_num,-1)
        one_hot_data = OneHotEncoder(categories=move_data.shape[1]*[['','L','T','R']],sparse=False).fit_transform(move_data)
        self.move = torch.from_numpy(one_hot_data).reshape(sample_num,cycle_num,-1).to(torch.float32).transpose(0,1)

        # 配时
        self.ts = torch.from_numpy(data['ts']).reshape(sample_num,cycle_num,-1).to(torch.float32).transpose(0,1)
    
    def __getitem__(self,i):
        # 保持维度结构
        return {'delay':self.delay[:,[i]],'obs_speed':self.speed[:,[i]],
                'obs_move':self.move[:,[i]],'ts':self.ts[:,[i]]}  
        
    def __len__(self):
        return self.sample_num
    
def get_obs(samples):
    # 观测数据的dataloader
    batch = [torch.cat([sample['obs_speed'],sample['obs_move']],dim=-1) for sample in samples]
    batch = torch.cat(batch,dim=-2)
    
    return batch

def get_sar(samples):
    batch_obs = get_obs(samples)
    batch_ts = torch.cat([sample['ts'] for sample in samples],dim=-2)
    batch_delay = torch.cat([sample['delay'] for sample in samples],dim=-2)
    
    return batch_obs,batch_ts,batch_delay

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

def get_dl(name_list,mode,batch_size=8,train_split=0.6):
    set_seed(0)  # 保证数据集相同，更好的做法是划分后将数据集保存
    ds_list = []
    if mode=='single':
        for name in name_list:
            ds_list.append(SingleData(name))
    elif mode=='multiple':
        for name in name_list:
            ds_list.append(MultipleData(name))
    
    ds = ConcatDataset(ds_list)
    
    train_size = int(train_split*len(ds))
    val_size = len(ds)-int(train_split*len(ds))
    
    ds_train,ds_val = random_split(ds,[train_size,val_size])

    dl_train = DataLoader(ds_train,batch_size=batch_size,collate_fn=get_sar,
                          shuffle=True,drop_last=True)
    dl_val = DataLoader(ds_val,batch_size=4,collate_fn=get_sar,
                        shuffle=False,drop_last=True)
    return dl_train,dl_val,ds_train,ds_val