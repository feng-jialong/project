#%%
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split,DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

#%% process
def train_step(model,x,y):
    model.train()
    
    model.optimizer.zero_grad()
    y_pred = model(x)
    loss = model.loss_func(y_pred,y)
    metric = model.metric_func(y_pred,y)
    loss.backward()
    model.optimizer.step()
    
    return loss.item(),metric

@torch.no_grad()
def validate_step(model,x,y):
    model.eval()
    
    y_pred = model(x)
    loss = model.loss_func(y_pred,y)
    metric = model.metric_func(y_pred,y)
    
    return loss.item(),metric

def resume(model):
    # 断点续训：寻找并导入checkpoint
    saved_step = 0
    saved_file = None
    current_step = 0
    
    global_epoch = 1
    global_step = 1
    global_loss = 0.0
    
    model_dir = model.model_dir
    if os.path.isdir(model_dir+'checkpoints/'):
        for file in os.listdir(model_dir+'checkpoints/'):
            if file.startswith('checkpoint'):
                tokens = file.split('.')[0].split('-')
                if len(tokens) != 2:
                    continue
                current_step = int(tokens[1])
                if current_step > saved_step:
                    saved_file = file
                    saved_step = current_step
        checkpoint_path = model_dir + 'checkpoints/' + saved_file
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model'])
        global_epoch = checkpoint['epoch']+1  # 设置开始的epoch
        global_step = checkpoint['step']+1  # 设置开始的step
        global_loss = checkpoint['loss']
    else:
        print("No exisiting model !")

    return global_epoch,global_step,global_loss

def check(model,epoch,step,loss):
    # 断点续训，保存checkpoint
    model_dir = model.model_dir
    checkpoint = {"model":model.state_dict(),"epoch":epoch,"step":step,"loss":loss}
    if not os.path.isdir(model_dir+'checkpoints/'):
        os.mkdir(model_dir+'checkpoints/')
    torch.save(checkpoint,model_dir+'checkpoints/'+'checkpoint-%s.pth' % (str(step)))

def train(model,epochs,train_dl,val_dl,device='cpu'):
    model_dir = model.model_dir
    # 断点续训：寻找并导入checkpoint，正在处理的epoch和step，以及对应loss
    global_epoch,global_step,global_loss = resume(model)
    model.to(device)
    
    train_log_step_freq = 10
    val_step_freq = 100
    check_step_freq = 100
    
    # tensorboard:初始化，若路径不存在会创建路径
    writer_step = SummaryWriter(log_dir=model_dir+'tb-step-logs',purge_step=global_step)
    # writer_epoch = SummaryWriter(log_dir=model_dir+'tb-epoch-logs',purge_step=global_epoch)
    
    # train loop
    loss_sum = global_loss
    for _ in range(epochs):
        pbar = tqdm(total=len(train_dl),desc=f"training epoch {global_epoch}")
        # train
        for _,(x,y) in enumerate(train_dl):
            loss,_ = train_step(model,x,y)
            loss_sum += loss
            pbar.update(1)
            
            if global_step % train_log_step_freq == 0:
                writer_step.add_scalars('loss',{'train':loss_sum/train_log_step_freq},
                                        global_step=global_step)
                loss_sum = 0.0
            
            
            if global_step % val_step_freq == 0:
                # validate
                loss_sum_val = 0.0
                metric_sum_val = {metric:0.0 for metric in model.metrics}
                
                for step_val,(x,y) in enumerate(val_dl,1):
                    loss_val,metric_val = validate_step(model,x,y)
                    loss_sum_val += loss_val
                    for metric in model.metrics:
                        metric_sum_val[metric] += metric_val[metric]
               
                for metric in model.metrics:
                    output_list = [str(i)+'-step-ahead' for i in range(1,model.lookahead+1)]
                    output_list += [str(i) for i in range(12)]
                    writer_step.add_scalars(metric,
                                            {key:value/step_val \
                                                for (key,value) \
                                                    in zip(output_list,metric_sum_val[metric])},
                                            global_step=global_step)
                writer_step.add_scalars('loss',{'val':loss_sum_val/step_val},
                                        global_step=global_step)

            # 断点续训，epochs保存checkpoint
            if global_step % check_step_freq == 0:
                check(model,global_epoch,global_step,loss_sum)
                
            global_step += 1
        
        pbar.close()
        # 断点续训，epochs保存checkpoint
        if global_epoch % 1 == 0:
            check(model,global_epoch,global_step-1,loss_sum)
        global_epoch += 1

def test(model,test_dl):
    # use trained model
    # loss_sum = 0.0
    metric_sum = {metric:0.0 for metric in model.metrics}
    
    for step,(x,y) in enumerate(test_dl,1):
        _,metric = validate_step(model,x,y)
    
        # loss_sum += loss
        for metric in model.metrics:
            metric_sum[metric] += metric[metric]

    result = np.stack([metric_sum[metric]/step for metric in model.metrics])  # (metric_num,feature_num)
    
    result = pd.DataFrame(result,
                          columns=model.metrics)
    
    return result

#%% data
class MyDataset(Dataset):
    def __init__(self,data_dir,lookback,lookahead,sample_size):
        self.data_dir = data_dir
        self.lookback = lookback
        self.lookahead = lookahead
        self.sample_size = sample_size
        assert self.lookback <= self.sample_size[0], "lookback out of range"
        assert self.lookahead <= self.sample_size[1], "lookahead out of range"
        with open(data_dir+'sample2chunk.pkl','rb') as f:
            self.sample2chunk = joblib.load(f)
        
    def __getitem__(self,i):
        chunk_index,sample_index = self.sample2chunk[i]
        
        for file in os.listdir(self.data_dir):
            if file.split('.')[-1] == 'pkl':
                continue
            if int(file.split('.')[0].split('_')[-1]) != chunk_index:
                continue
            data = torch.load(self.data_dir+file)
            x = {'obs':data['obs'][[sample_index],(self.sample_size[0]-self.lookback):self.sample_size[0]],  # lookback
                 'tc':data['tc'][[sample_index],(self.sample_size[0]-self.lookback):(self.sample_size[0]+self.lookahead)]}  # lookback + lookahead
            y = data['delay'][[sample_index],:self.lookahead]  # lookahead, 数据可以更长但是只取lookahead那么长
            del data
        return (x,y)
        
    def __len__(self):
        return len(self.sample2chunk)
    
def train_val_test_split(dataset,splits=[0.8,0.1,0.1],seed=0):
    train_size = int(splits[0]*len(dataset))
    val_size = int(splits[1]*len(dataset))
    test_size = len(dataset)-train_size-val_size

    # 随机数种子需要设置
    train_ds,val_ds,test_ds = random_split(dataset,[train_size,val_size,test_size],
                                           generator=torch.Generator().manual_seed(seed))

    return train_ds,val_ds,test_ds

def my_collate_fn(samples):    
    x = {}
    x['obs'] = np.concatenate([sample[0]['obs'] for sample in samples],axis=0)
    x['tc'] = torch.cat([sample[0]['tc'] for sample in samples],dim=0)
    
    y = torch.cat([sample[1] for sample in samples],dim=0)
    
    return (x,y)

def get_dataloader(dataset,batch_size,seed):
    train_ds,val_ds,test_ds = train_val_test_split(dataset)
    num_workers = 4
    
    # dataloader使用固定的随机数种子
    train_dl = DataLoader(train_ds,batch_size=batch_size,collate_fn=my_collate_fn,
                          shuffle=True,drop_last=True,
                          # generator=torch.Generator().manual_seed(seed),
                          num_workers=num_workers)
    val_dl = DataLoader(val_ds,batch_size=16,collate_fn=my_collate_fn,
                        shuffle=False,drop_last=True,
                        # generator=torch.Generator().manual_seed(seed),
                        num_workers=num_workers)
    test_dl = DataLoader(test_ds,batch_size=16,collate_fn=my_collate_fn,
                         shuffle=False,drop_last=True,
                         # generator=torch.Generator().manual_seed(seed),
                         num_workers=num_workers)
    # 为严格保证batch_size，设置drop_last=True
    return train_dl,val_dl,test_dl