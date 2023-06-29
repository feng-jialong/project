#%%
# region
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
# endregion

#%% model
class DelayPredictor(nn.Module):
    def __init__(self,model_dir,batch_size,lookback,lookahead):
        super().__init__()
        self.model_dir = model_dir
        self.lookback = lookback
        self.lookahead = lookahead
        self.batch_size = batch_size
        
        hidden_channels = 16
        self.conv = nn.Sequential(nn.Conv2d(20,hidden_channels,
                                            kernel_size=(7,3),padding=1),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(4,2),stride=2,padding=0),
                                  nn.Conv2d(hidden_channels,hidden_channels,
                                            kernel_size=(7,3),padding=1),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(4,2),stride=2,padding=0),
                                  nn.Flatten())
        l_enc_input_size = 16
        l_enc_hidden_size = 16
        l_enc_output_size = 16
        D = 2 # 1: 单向，2: 双向
        self.D = D
        l_enc_num_layers = 2
        self.l_enc_input_layer = nn.Sequential(nn.Linear(21*hidden_channels,
                                                         l_enc_input_size),
                                               nn.BatchNorm1d(l_enc_input_size),
                                               nn.ReLU(),
                                               nn.Linear(l_enc_input_size,
                                                         l_enc_input_size),
                                               nn.BatchNorm1d(l_enc_input_size),
                                               nn.ReLU())
        # sequential的模型只接受单输入
        self.l_enc_layer = nn.LSTM(input_size=l_enc_input_size,
                                   hidden_size=l_enc_hidden_size,
                                   num_layers=l_enc_num_layers,dropout=0.0,
                                   bidirectional=bool(D-1))
        # lower context
        self.l_enc_output_layer = nn.Sequential(nn.Linear(D*l_enc_hidden_size,
                                                          l_enc_output_size),
                                                TheNorm(l_enc_output_size),
                                                nn.ReLU(),
                                                nn.Linear(l_enc_output_size,
                                                          l_enc_output_size),
                                                TheNorm(l_enc_output_size),
                                                nn.ReLU())
        lower_context_size = l_enc_output_size
        
        # size: phase=5 + split=8 + target_func=3*8
        tc_size = 37
        tc_output_size = 16
        self.tc_output_layer = nn.Sequential(nn.Linear(tc_size,
                                                       tc_output_size),
                                             TheNorm(tc_output_size),
                                             nn.ReLU(),
                                             nn.Linear(tc_output_size,
                                                       tc_output_size),
                                             TheNorm(tc_output_size),
                                             nn.ReLU())
        
        u_enc_input_size = 16
        u_enc_hidden_size = 16
        u_enc_output_size = 16
        self.u_enc_input_layer = nn.Sequential(nn.Linear(tc_output_size+lower_context_size,
                                                         u_enc_input_size),
                                               TheNorm(u_enc_input_size),
                                               nn.ReLU(),
                                               nn.Linear(u_enc_input_size,
                                                         u_enc_input_size),
                                               TheNorm(u_enc_input_size),
                                               nn.ReLU())
        self.u_enc_layer = nn.LSTM(input_size=u_enc_input_size,
                                   hidden_size=u_enc_hidden_size,
                                   num_layers=1,dropout=0.0)
        # upper context
        self.u_enc_output_layer = nn.Sequential(nn.Linear(u_enc_hidden_size,
                                                          u_enc_output_size),
                                                TheNorm(u_enc_output_size),
                                                nn.ReLU(),
                                                nn.Linear(u_enc_output_size,
                                                          u_enc_output_size),
                                                TheNorm(u_enc_output_size),
                                                nn.ReLU())
        upper_context_size = u_enc_output_size

        dec_input_size = 16
        dec_hidden_size = upper_context_size
        self.dec_input_layer = nn.Sequential(nn.Linear(tc_size,dec_input_size),
                                             TheNorm(dec_input_size),
                                             nn.ReLU(),
                                             nn.Linear(dec_input_size,dec_input_size),
                                             TheNorm(dec_input_size),
                                             nn.ReLU())
        self.dec_layer = nn.LSTM(input_size=dec_input_size,
                                 hidden_size=dec_hidden_size,
                                 num_layers=1,dropout=0.0)
        # output delay
        self.dec_output_layer = nn.Sequential(nn.Linear(dec_hidden_size,dec_hidden_size),
                                              TheNorm(dec_hidden_size),
                                              nn.ReLU(),
                                              nn.Linear(dec_hidden_size,1))
        # output delay_m
        self.delay_m_output_layer = nn.Sequential(nn.Linear(upper_context_size,
                                                            upper_context_size),
                                                  nn.BatchNorm1d(upper_context_size),
                                                  nn.ReLU(),
                                                  nn.Linear(upper_context_size,12))
        
        self.optimizer = torch.optim.AdamW(self.parameters(),lr=0.002)
                                           # weight_decay=1e-3)
        
        self.metrics = ['rmse','mae','mape','wape']
    
    def lower_encoding(self,obs):
        lower_context = []
        for i in range(self.lookback):
            seq_lens = []
            for s in obs[:,i]:  # (batch,frames!,***)
                seq_lens.append(s.shape[0])
            # 卷积层
            obs_conv = self.conv(torch.cat(list(obs[:,i]),dim=0))  # (batch&frames!,*)
            obs_conv = self.l_enc_input_layer(obs_conv)  # (batch&frames!,*)
            obs_conv = torch.split(obs_conv,seq_lens,dim=0)  # (batch,frames!,*)
            
            # lower encoder
            obs_padded = pad_sequence(obs_conv)  # (frames,batch,*)
            obs_packed = pack_padded_sequence(obs_padded,seq_lens,enforce_sorted=False)
            # _,(l_enc_h,_) = self.l_enc_layer(obs_packed,(None,None))   # 错误的，None会参与计算
            _,(l_enc_h,_) = self.l_enc_layer(obs_packed)
            # (D*num_layers,batch,H)
            if self.D == 2:
                lower_context.append(torch.cat([l_enc_h[-2,:,:],l_enc_h[-1,:,:]],
                                               dim=-1))  # (batch,2*)
            elif self.D == 1:
                lower_context.append(l_enc_h[-1,:,:])  # (batch,*)

        lower_context = torch.stack(lower_context,dim=1)  # (batch,lookback,*)
        lower_context = self.l_enc_output_layer(lower_context)
        
        return lower_context
    
    def upper_encoding(self,tc_back,lower_context):
        tc_back = self.tc_output_layer(tc_back.transpose(0,1)).transpose(0,1)  # (batch,lookback,*)
        
        u_enc_input = torch.cat([tc_back,lower_context],dim=-1)  # (batch,lookback,*)
        u_enc_input = self.u_enc_input_layer(u_enc_input).transpose(0,1)  # (lookback,batch,*)
        _,(u_enc_h,_) = self.u_enc_layer(u_enc_input)
        # (num_layers,batch,H)
        upper_context = u_enc_h[[-1],:,:]  # (1,batch,*) 保持结构，后续decoder输入需要 
        upper_context = self.u_enc_output_layer(upper_context)
        
        return upper_context
    
    def decoding(self,upper_context,tc_ahead):
        # decoding
        tc_ahead = self.dec_input_layer(tc_ahead).transpose(0,1)  # (lookahead,batch,*)
        dec_output,_ = self.dec_layer(tc_ahead,(upper_context,torch.zeros_like(upper_context)))
        # (lookahead,batch,H)
        delay_output = self.dec_output_layer(dec_output.transpose(0,1)) # (batch,lookahead,1)
        delay_output = delay_output[:,:,0]  # (batch,lookahead)
        
        return delay_output
    
    def output(self,upper_context):
        # delay_m
        delay_m_output = self.delay_m_output_layer(upper_context[0])  # (batch,12)
        
        return delay_m_output
    
    def forward(self,x):
        obs = x['obs']  # (batch,lookback,frames!,*)

        lower_context = self.lower_encoding(obs)
        
        tc = x['tc']  # (batch,lookback+lookahead,*)
        tc_back = tc[:,:self.lookback,:]  # (batch,lookback,*)
        tc_ahead = tc[:,self.lookback:,:]  # (batch,lookahead,*)
        
        upper_context = self.upper_encoding(tc_back,lower_context)
        
        delay_m_output = self.output(upper_context)

        delay_output = self.decoding(upper_context,tc_ahead)
        
        y = torch.cat([delay_output,delay_m_output],dim=-1)
        
        return y
        
    def loss_func(self,y_pred,y):
        y = torch.cat([y['delay'],y['delay_m']],dim=-1)  # (batch,lookahead+12)
        
        k = 0.0
        
        mse_loss = ((y[:,:self.lookahead]-y_pred[:,:self.lookahead])**2).mean()
        mse_loss += k*((y[:,self.lookahead:]-y_pred[:,self.lookahead:])**2).mean()
        
        return mse_loss
    
    def metric_func(self,y_pred,y):
        y = torch.cat([y['delay'],y['delay_m']],dim=-1)
        
        rmse = torch.sqrt(((y-y_pred)**2).mean(dim=0))
        mae = torch.abs(y-y_pred).mean(dim=0)
        mape = torch.abs((y-y_pred)/y).mean(dim=0)
        wape = torch.abs(y-y_pred).sum(dim=0)/y.sum(dim=0)
        
        return {'rmse':rmse.detach().numpy(),
                'mae':mae.detach().numpy(),
                'mape':mape.detach().numpy(),
                'wape':wape.detach().numpy()}

class TheNorm(nn.Module):
    def __init__(self,num_features):
        # 沿-1维度归一化
        super(TheNorm,self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        
    def forward(self,x):
        return self.bn(x.transpose(-2,-1)).transpose(-2,-1)
    
#%% utils
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

def train(model,epochs,train_dl,val_dl):
    model_dir = model.model_dir
    # 断点续训：寻找并导入checkpoint，正在处理的epoch和step，以及对应loss
    global_epoch,global_step,global_loss = resume(model)
    
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

@torch.no_grad()   
def test(model,test_dl):
    # using trained model
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
        
#%% data utils
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
            x = {'obs':data['obs'][[sample_index],
                                   (self.sample_size[0]-self.lookback):self.sample_size[0]],  # lookback
                 'tc':data['tc'][[sample_index],
                                 (self.sample_size[0]-self.lookback):(self.sample_size[0]+self.lookahead)]}  # lookback + lookahead
            y = {'delay':data['delay'][[sample_index],
                                       :self.lookahead],  # lookahead
                 'delay_m':data['delay_m'][[sample_index]]}
            del data
        return (x,y)
        
    def __len__(self):
        return len(self.sample2chunk)
    
def train_val_test_split(dataset,splits=[0.8,0.1,0.1]):
    train_size = int(splits[0]*len(dataset))
    val_size = int(splits[1]*len(dataset))
    test_size = len(dataset)-train_size-val_size

    # 固定随机数种子
    train_ds,val_ds,test_ds = random_split(dataset,[train_size,val_size,test_size],
                                           generator=torch.Generator().manual_seed(0))

    return train_ds,val_ds,test_ds

def my_collate_fn(samples):    
    x = {}
    x['obs'] = np.concatenate([sample[0]['obs'] for sample in samples],axis=0)
    x['tc'] = torch.cat([sample[0]['tc'] for sample in samples],dim=0)
    
    y = {}
    y['delay'] = torch.cat([sample[1]['delay'] for sample in samples],dim=0)
    y['delay_m'] = torch.cat([sample[1]['delay_m'] for sample in samples],dim=0)
    
    return (x,y)

def get_dataloader(dataset,batch_size):
    train_ds,val_ds,test_ds = train_val_test_split(dataset)
    
    num_workers = 4
    
    # dataloader使用固定的随机数种子
    train_dl = DataLoader(train_ds,batch_size=batch_size,collate_fn=my_collate_fn,
                          shuffle=True,drop_last=True,
                          generator=torch.Generator().manual_seed(0),
                          num_workers=num_workers)
    val_dl = DataLoader(val_ds,batch_size=16,collate_fn=my_collate_fn,
                        shuffle=False,drop_last=True,
                        generator=torch.Generator().manual_seed(0),
                        num_workers=num_workers)
    test_dl = DataLoader(test_ds,batch_size=16,collate_fn=my_collate_fn,
                         shuffle=False,drop_last=True,
                         generator=torch.Generator().manual_seed(0),
                         num_workers=num_workers)
    # 严格保证batch_size，所以drop_last=True
    return train_dl, val_dl, test_dl