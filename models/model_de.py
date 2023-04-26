import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 延误估计模型
class DelayEstimationModel(nn.Module):
    def __init__(self,obs_func,model_dir):
        super(DelayEstimationModel,self).__init__()
        self.obs_func = obs_func
        ts_size = 4
        input_size = self.obs_func.output_size + ts_size
        
        hidden_size = 4
        
        self.network = nn.Sequential(nn.Linear(input_size,hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size,hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size,1))
        
        self.ts_bn = nn.BatchNorm1d(ts_size)  # ts的维度, hardcode

        self.metrics = ['MAE','MAPE']
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.01)
        
        self.model_dir = model_dir

    def forward(self,obs,ts):
        obs_p = self.obs_func(obs)
        delay = self.network(torch.cat([obs_p,self.ts_bn(ts)],dim=-1))
        return delay
    
    def loss_func(self,delay_p,delay):
        return F.mse_loss(delay_p,delay)
    
    def metric_func(self,delay_p,delay):
        MAE = torch.abs(delay_p-delay).mean()
        MAPE = torch.abs((delay_p-delay)/delay).mean()
        return {'MAE':MAE.item(),'MAPE':MAPE.item()}

def train_step(model,obs,ts,delay):
    model.train()
    
    model.optimizer.zero_grad()
    delay_p = model(obs,ts)
    loss = model.loss_func(delay_p,delay)
    metric = model.metric_func(delay_p,delay)
    loss.backward()
    model.optimizer.step()
    
    return loss.item(),metric

@torch.no_grad()
def validate_step(model,obs,ts,delay):
    model.eval()
    
    delay_p = model(obs,ts)
    loss = model.loss_func(delay_p,delay)
    metric = model.metric_func(delay_p,delay)
    
    return loss.item(),metric

def resume(model):
    # 断点续训：寻找并导入checkpoint
    saved_epoch = 0
    saved_file = None
    current_epoch = 0
    start_epoch = 1
    model_dir = model.model_dir
    if os.path.isdir(model_dir+'checkpoints/'):
        for file in os.listdir(model_dir+'checkpoints/'):
            if file.startswith('checkpoint'):
                tokens = file.split('.')[0].split('-')
                if len(tokens) != 2:
                    continue
                current_epoch = int(tokens[1])
                if current_epoch > saved_epoch:
                    saved_file = file
                    saved_epoch = current_epoch
        checkpoint_path = model_dir + 'checkpoints/' + saved_file
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model'])
        # model.network.load_state_dict(checkpoint['model'][0])
        # model.ts_bn.load_state_dict(checkpoint['model'][1])
        # model.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1  # 设置开始的epoch
    return start_epoch

def check(model,epoch):
    # 断点续训，保存checkpoint
    model_dir = model.model_dir
    checkpoint = {
        "model": model.state_dict(),
        # "model": [model.network.state_dict(),model.ts_bn.state_dict()],
        # 'optimizer': model.optimizer.state_dict(),
        "epoch": epoch
    }
    if not os.path.isdir(model_dir+'checkpoints/'):
        os.mkdir(model_dir+'checkpoints/')
    torch.save(checkpoint,model_dir+'checkpoints/'+'checkpoint-%s.pth' % (str(epoch)))

def train(model,epochs,dl_train,dl_val):
    model_dir = model.model_dir
    # 断点续训：寻找并导入checkpoint
    start_epoch = resume(model)
    
    history = pd.DataFrame(columns=['epoch','loss','loss_val']+model.metrics)
    
    # tensorboard:初始化，若路径不存在会创建路径
    writer = SummaryWriter(log_dir=model_dir+'tb-logs',purge_step=start_epoch)
    
    # train loop
    for epoch in range(start_epoch,start_epoch+epochs):
        # train
        loss_sum = 0.0
        for step,(obs,ts,delay) in enumerate(dl_train,1):
            loss,_ = train_step(model,obs,ts,delay)
            loss_sum += loss
        
        # validate
        loss_sum_val = 0.0
        metric_sum_val = {metric:0.0 for metric in model.metrics}
        for step_val,(obs,ts,delay) in enumerate(dl_val,1):
            loss_val,metric_val = validate_step(model,obs,ts,delay)
            loss_sum_val += loss_val
            for metric in model.metrics:
                metric_sum_val[metric] += metric_val[metric]
    
        info = {'epoch':epoch,
                'loss':loss_sum/step,
                'loss_val':loss_sum_val/step_val}
        info.update({metric:metric_sum_val[metric]/step_val for metric in model.metrics})
        
        # tensorboard logging
        writer.add_scalars('loss',
                        {'train':info['loss'],'val':info['loss_val']},
                        global_step=epoch)
        for name,param in model.named_parameters():
            writer.add_histogram(name+'_grad',param.grad,epoch)
            writer.add_histogram(name+'_param',param,epoch)

        for metric in model.metrics:
            writer.add_scalar(metric,info[metric],
                            global_step=epoch)
    
        history.loc[epoch-1] = info
        
        # 通过print的方式告知进度，使用tqdm库创建进度条好一些
        print(f"epoch {info['epoch']}, loss:{info['loss']}, val_loss:{info['loss_val']}")  
        # 注意用双引号套住单引号
        
        # 断点续训，保存checkpoint
        if epoch % 100 == 0:
            check(model,epoch)
        
    return history

def evaluate(model,dl_test):
    # use trained model
    loss_sum = 0.0
    metric_sum = {metric:0.0 for metric in model.metrics}
    
    for step,(obs,ts,delay) in enumerate(dl_test,1):
        loss,metric_test = validate_step(model,obs,ts,delay)
    
        loss_sum += loss
        for metric in model.metrics:
            metric_sum[metric] += metric_test[metric]
    
    print(f"loss: {loss_sum/step}")
    for metric in model.metrics:
        print(metric+f":{metric_sum[metric]/step}")