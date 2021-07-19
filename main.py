import torch.nn as nn
import torch.optim as optim
import math
from Net import Net #网络类
from test_data_loader import testdata #测试集 需要是nn.DataLoader类实例化的对象
from train_data_loader import traindata#训练集 同上
from train import trainer
from plotpicture import plot #画图函数
Batchsize=64
testdata=testdata(Batchsize)
traindata=traindata(Batchsize)
optimizer = optim.Adam #需要是一个迭代器类，不可是对象
criterion = nn.CrossEntropyLoss #损失函数，同样需要是类
lambda1 = lambda epoch: 0.001* (7.955-math.exp(epoch/30)) if epoch<5  else 0.0001 if 0.006*((2.98-1.5*math.exp(epoch/50)))<0 else 0.006*((2.98-1.5*math.exp(epoch/50)))
#lambada表达式，来确定可变学习率的值，默认是采用optim.lr_scheduler.LambdaLR的改变方法，默认为None，即固定学习率
trainer=trainer(Net,optimizer,criterion,traindata,100,Batchsize,testdata,lambda1)#原型：（model,optim,criterion,dataset,epoch,batchsize,testdata,schedulerlambda=None,boolload=False,loadermodle=None)
testacc,trainacc,testloss,trainloss=trainer.train(1e-4,10,"mymodelname.pkl")#（学习率（不传入schedulerlambda有效），测试间隔（每隔多少个batch进行测试），保存的模型名称（.pt一样可以））
plot(testacc,trainacc,testloss,trainloss)#绘图，可重写plot函数




