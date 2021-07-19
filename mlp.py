import torch
import torch.nn as nn
import torch.nn.functional as F
class mlp(nn.Module):
    def __init__(self,w,h,kinds):
        super().__init__()
        self.w=w
        self.h=h
        self.kinds=kinds
        self.lay1=nn.Linear(self.w*self.h*3,2048,bias=True)
        self.lay2=nn.Linear(2048,2048)
        self.lay3=nn.Linear(2048,self.kinds)
    def forward(self,x):
        x=F.relu(self.lay1(x))
        x=F.relu(self.lay2(x))
        x=F.relu(self.lay3(x))
        x=F.softmax(x,dim=0)
        return x