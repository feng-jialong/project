# import 
import pandas as pd
import torch
import torch.nn as nn

# region autoencoder
class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1,1),nn.Tanh(),
            nn.Linear(1,1),nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1,1),nn.Tanh(),
            nn.Linear(1,1),nn.Tanh()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.005)
        self.loss_func = nn.MSELoss()
    
    def forward(self,x):
        x_p = self.decoder(self.encoder(x))
        return x_p
    
    def encode(self,x):
        return self.encoder(x)

def train_step(model,x):
    model.train()
    
    model.optimizer.zero_grad()
    x_p,s = model(x)
    loss = model.loss_func(x_p,x)
    loss.backward()
    model.optimizer.step()
    
    return loss.item()

@torch.no_grad()
def validate_step(model,x):
    model.eval()
    
    x_p,s = model(x)
    loss = model.loss_func(x_p,x)
    
    return loss.item()

# 不完善，参考其他模型改写
def train(model,epochs,dl_train,dl_val):
    history = pd.DataFrame(columns=['epoch','loss','loss_val'])
    for epoch in range(1,epochs+1):
        loss_sum = 0.0
        for step,x in enumerate(dl_train,1):
            loss = train_step(model,x)
            loss_sum += loss
        
        loss_sum_val = 0.0
        for step_val,x in enumerate(dl_val,1):
            loss_val = validate_step(model,x)
            loss_sum_val += loss_val
    
        info = [epoch,loss_sum/step,loss_sum_val/step_val]
        history.loc[epoch-1] = info
        
    return history
# endregion