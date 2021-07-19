import torch
import torch.nn as nn
from layernorm import LayerNormalization
from mlp import mlp
class Net(nn.Module):
    def __init__(self,w,h,batchsize):
        super().__init__()
        self.w=w
        self.h=h
        self.mlp=mlp(self.w,self.h,10)
        self.batchsize=batchsize
        self.norm=LayerNormalization(self.w*self.h*3)
        self.capus=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(self.w,self.h),padding=0,groups=3)
        self.transformerlayer=nn.TransformerEncoderLayer(d_model=3,nhead=3,dim_feedforward=2048,dropout=0.1,activation="gelu")
        self.transformer=nn.TransformerEncoder(self.transformerlayer,6,norm=None)
    def forward(self,x):
        for i in range(self.batchsize):
            ori=x[i:i+1]
            ori=ori.resize(self.w,self.h,3)
            original=torch.unsqueeze(ori.resize(3,32,32),dim=0)
            #print(torch.unsqueeze(ori.resize(3,32,32),dim=0).size())
            ori=self.capus(torch.unsqueeze(ori.resize(3,32,32),dim=0))
            ori=torch.add(original,ori)
            #print(ori.size())
            ori=ori.resize(32,32,3)
            ori=self.transformer(ori)
            ori=ori.view(1,-1)
            ori=self.mlp(ori)
            if i==0:
                sum=ori
            else:
                sum=torch.cat([sum,ori],dim=0)
        return sum