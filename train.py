import torch
import torch.nn
from test import tester as ts
class trainer():
    def __init__(self,model,optim,criterion,dataset,epoch,batchsize,testdata,schedulerlambda=None,boolload=False,loadermodle=None):
        self.modle=model
        self.optim=optim
        self.criterion=criterion()
        self.dataset=dataset
        self.epoch=epoch
        self.sche=schedulerlambda
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.boolload=boolload
        self.loadmodel=loadermodle
        self.bsize=batchsize
        self.testdata=testdata
    def train(self,lr,testnum,modelname):
        if self.boolload==False:#是否加载已有模型参数，默认仅加载网络参数，需传入网络结构
            net=self.modle(w=32,h=32,batchsize=self.bsize)#在这里实例化
            net=net.to(self.device)
        else:
            net = self.modle(w=32, h=32, batchsize=self.bsize)
            net = net.to(self.device)
            net.load_state_dict(torch.load(self.loadmodel))#加载现有模型参数
        opt=self.optim(net.parameters(),lr)#实例化迭代器
        if self.sche!=None:
            sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=self.sche)#实例化scheduler 即可变学习率调整器
        accmax=0
        for e in range(self.epoch):
            lossnow=0
            trainnumber=0
            trainloss = []
            testloss = []
            trainacc = []
            testacc = []
            trainac=0
            for i, data in enumerate(self.dataset):
                net.train()
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                opt.zero_grad()#清空上一个batch的梯度
                outputs = net(inputs)
                _, trainpredicted = torch.max(outputs.data, 1)#分类问题，返回概率最大的类别（概率，类别）
                outlist = trainpredicted.tolist()
                labelslist = labels.tolist()
                for ii in range(0,self.bsize):#统计训练集准确率
                    a = outlist[ii]
                    b = labelslist[ii]
                    if a == b:
                        trainac = trainac + 1
                loss = self.criterion(outputs, labels)#求loss
                loss.backward()#反向传播，求各个weight的导数
                opt.step()#更新weight
                trainnumber=trainnumber+1
                lossnow = lossnow + loss.item()
                if trainnumber%testnum==0:
                    ave_train_loss=lossnow/testnum
                    ave_train_acc = trainac / (self.bsize * testnum)
                    torch.cuda.empty_cache()
                    tester=ts(net,self.testdata,self.criterion,self.bsize)
                    ave_test_loss, ave_test_acc =tester.test()
                    trainloss.append(ave_train_loss)
                    trainacc.append(ave_train_acc)
                    testloss.append(ave_test_loss)
                    testacc.append(ave_test_acc)
                    if ave_test_acc>accmax:
                        torch.save(net.state_dict(), modelname)
                        accmax = ave_test_acc
                    lossnow=0
                    trainnumber=0
                    trainac=0
                    #print(testacc, trainacc, testloss, trainloss)
            if self.sche!=None:
                sch.step()

        return testacc,trainacc,testloss,trainloss









