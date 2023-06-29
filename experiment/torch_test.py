#%% import
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

#%%
x = torch.tensor(0.5,requires_grad=True)

iter_num = 100
alpha = 0.01

for i in range(iter_num):
    
    y = 1.0/x
    f = x**2 + y**2
    
    f.backward()
    
    with torch.no_grad():
        x = x - alpha * x.grad
    x.grad = None
    x.requires_grad = True
    
    if i%10 == 0:
        print(x.item())
    
#%%
x = torch.tensor(6.6,requires_grad=True)
y = torch.tensor(-3.2,requires_grad=True)

iter_num = 100
alpha = 0.1

for i in range(iter_num):
    
    z = x
    f = x + y**2 - z
    
    f.backward()
    
    with torch.no_grad():
        x = x - alpha * x.grad
        y = y - alpha * y.grad
    x.grad,y.grad = None,None
    x.requires_grad,y.requires_grad = True,True
    
    if i%10 == 0:
        print(x.item(),y.item(),z.item())

#%% BatchNorm1d: (N,C,L)
t = torch.tensor([[[1,2],[3,-2],[-1,-2]],
              [[-1,-2],[-3,2],[1,2]]],dtype=torch.float)
print(t)
print(torch.nn.BatchNorm1d(4)(t))

#%% 索引浅拷贝
a = torch.tensor([[1.0,2.0],[4.0,5.0]])
print(a.requires_grad)
b = a[0,0]
b *= 2
b.requires_grad = True
print(a.requires_grad)

#%%
a = torch.tensor(5.0,requires_grad=True)
b = a**2
a.item()