import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#%% 车流预测模型
class FlowPredictionModel(nn.Module):
    def __init__(self,obs_func,model_dir):
        super(FlowPredictionModel,self).__init__()
        self.model_dir = model_dir
        self.obs_func = obs_func  
        self.obs_size = self.obs_func.output_size
        ts_size = 4
        input_size = self.obs_size + ts_size
        
        self.hidden_size = 4
        
        self.input_layer = nn.Sequential(nn.Linear(input_size,self.hidden_size),
                                         nn.BatchNorm1d(self.hidden_size),
                                         nn.ReLU())
        
        self.lstm_1 = nn.LSTMCell(input_size=self.hidden_size,
                                  hidden_size=self.hidden_size)
        # self.lstm_2 = nn.LSTMCell(input_size=self.hidden_size,
        #                           hidden_size=self.hidden_size)
        
        self.lstm_list = [self.lstm_1]
        
        if True:  # 权重矩阵的正交初始化
            for lstm in self.lstm_list:
                for w in torch.split(lstm.weight_ih,4):
                    nn.init.orthogonal_(w)
                for w in torch.split(lstm.weight_hh,4):
                    nn.init.orthogonal_(w)

        
        self.output_layer = nn.Sequential(nn.BatchNorm1d(self.hidden_size),
                                          nn.Linear(self.hidden_size,self.obs_size))
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.01)
        
        self.loss_func = nn.MSELoss()
        self.metrics = ['MAE','MAPE']
        
        # hidden state和cell state初始化
        self.reset()
    
    def forward(self,obs,ts):
        # 接收序列数据，更新状态，返回输出
        # (cycle_num, batch_size, features_num)
        seq_len = obs.shape[0]
        batch_size = obs.shape[1]
        
        if batch_size > 1:  # 处于训练模式，前向传播前重置网络状态
            self.reset(batch_size)
            
        output = []

        obs = self.obs_func(obs.reshape(seq_len*batch_size,-1)).reshape(seq_len,batch_size,-1)
        
        x = torch.cat([obs,ts],dim=2)

        for i in range(seq_len):
            input_x = self.input_layer(x[i])
            for i,lstm in enumerate(self.lstm_list):
                self.h_list[i],self.C_list[i] = lstm(input_x,(self.h_list[i],self.C_list[i]))
                input_x = self.h_list[i]
            y = self.output_layer(input_x)
            output.append(torch.unsqueeze(y,dim=0))
            
        if batch_size > 1:  # 处于训练模式，前向传播前重置网络状态
            self.reset(batch_size)
            
        return torch.cat(output,dim=0)
    
    def loss_func(self,x_p,x):
        # (cycle_num,batch_size,obs_size)
        # dimensionality reduction
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x = self.obs_func(x.reshape(seq_len*batch_size,-1)).reshape(seq_len,batch_size,-1)
        
        return F.mse_loss(x_p,x,reduction='mean')
        
    def metric_func(self,x_p,x):
        # dimensionality reduction
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x = self.obs_func(x.reshape(seq_len*batch_size,-1)).reshape(seq_len,batch_size,-1)
        
        MAE = torch.abs(x_p-x).mean()
        MAPE = torch.abs((x_p-x)/x).mean()
        return {'MAE':MAE.item(),'MAPE':MAPE.item()}
    
    def reset(self,batch_size=1):
        # 重置网络状态
        self.h_list = [torch.zeros(batch_size,lstm.hidden_size)
                       for lstm in self.lstm_list]
        self.C_list = [torch.zeros(batch_size,lstm.hidden_size)
                       for lstm in self.lstm_list]
    
    def predict(self,obs,ts,step=1):
        # 接收单步输入，递推进行单步预测，不更新网络状态
        # obs: (1,1,feature_num)  # 保持维度结构
        # ts: (step,1,feature_num)
        pred = torch.zeros_like(obs)  # 单步预测结果缓存
        output = []
        
        # 多步预测的state
        h_list = [h.clone() for h in self.h_list]
        C_list = [C.clone() for C in self.C_list]
        
        for _ in range(step):
            obs = torch.unsqueeze(self.obs_func(obs[0]),dim=0)
            x = torch.cat([obs,ts],dim=-1)
            input_x = self.input_layer(x)
            for i,lstm in self.lstm_list:
                h_list[i],C_list[i] = lstm(input_x,(h_list[i],C_list[i]))
                input_x = h_list[i]
            pred = self.ouput_layer(input_x)
            output.append(pred)
        
        return output

def train_step(model,obs,ts,next_obs):
    model.train()
    
    model.optimizer.zero_grad()
    next_obs_p = model(obs,ts)
    loss = model.loss_func(next_obs_p,next_obs)
    metric = model.metric_func(next_obs_p,next_obs)
    loss.backward()
    model.optimizer.step()
    
    return loss.item(),metric

@torch.no_grad()
def validate_step(model,obs,ts,next_obs):
    model.eval()
    
    next_obs_p = model(obs,ts)
    loss = model.loss_func(next_obs_p,next_obs)
    metric = model.metric_func(next_obs_p,next_obs)
    
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
        start_epoch = checkpoint['epoch']+1  # 设置开始的epoch
    return start_epoch

def check(model,epoch):
    # 断点续训，保存checkpoint
    model_dir = model.model_dir
    checkpoint = {
        "model": model.state_dict(),
        'epoch': epoch
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
        for step,(obs,ts,_) in enumerate(dl_train,1):
            # (cycle_num,sample_num,feature_num)
            # 因为要预测下一步，最后一个周期不作训练
            loss,_ = train_step(model,obs[:-1,:,:],ts[:-1,:,:],obs[1:,:,:])
            loss_sum += loss
        
        # validate
        loss_sum_val = 0.0
        metric_sum_val = {metric:0.0 for metric in model.metrics}
        for step_val,(obs,ts,_) in enumerate(dl_val,1):
            loss_val,metric_val = validate_step(model,obs[:-1,:,:],ts[:-1,:,:],obs[1:,:,:])
            loss_sum_val += loss_val
            for metric in model.metrics:
                metric_sum_val[metric] += metric_val[metric]
    
        info = {'epoch':epoch,
                'loss':loss_sum/step,
                'loss_val':loss_sum_val/step_val}
        info.update({metric:metric_sum_val[metric]/step_val for metric in model.metrics})
        
        # tensorboard logging
        # loss
        writer.add_scalars('loss',
                        {'train':info['loss'],'val':info['loss_val']},
                        global_step=epoch)
        # metrics
        for metric in model.metrics:
            writer.add_scalar(metric,info[metric],
                            global_step=epoch)
        # params and grads
        for name,param in model.named_parameters():
            if name.startswith('lstm'):  # lstm param
                param_list = torch.split(param,4)
                grad_list = torch.split(param.grad,4)  
                # 在计算图中求出梯度再split, split操作后的张量没有梯度
                # split后的结果为原tensor的view
                # input gate
                writer.add_histogram(name+'_i_grad',grad_list[0],epoch)
                writer.add_histogram(name+'_i_param',param_list[0],epoch)
                # forget gate
                writer.add_histogram(name+'_f_grad',grad_list[1],epoch)
                writer.add_histogram(name+'_f_param',param_list[1],epoch)
                # cell gate
                writer.add_histogram(name+'_g_grad',grad_list[2],epoch)
                writer.add_histogram(name+'_g_param',param_list[2],epoch)
                # output gate
                writer.add_histogram(name+'_o_grad',grad_list[3],epoch)
                writer.add_histogram(name+'_o_param',param_list[3],epoch)
            else:
                writer.add_histogram(name+'_grad',param.grad,epoch)
                writer.add_histogram(name+'_param',param,epoch)

        history.loc[epoch-1] = info
        
        # 通过print的方式告知进度，使用tqdm库创建进度条好一些
        print(f"epoch {info['epoch']}, loss:{info['loss']}, val_loss:{info['loss_val']}")  
        # 注意用双引号套住单引号
        
        # 断点续训，保存checkpoint
        if epoch % 100 == 0:
            check(model,epoch)
        
    return history
