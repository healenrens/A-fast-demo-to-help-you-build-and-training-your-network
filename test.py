import torch
import torch.nn as nn
class tester():
    def __init__(self,model,testdata,criterion,batchsize):
        self.modle=model
        self.set=testdata
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cri=criterion
        self.batchsize=batchsize
        self.avetestloss=0
        self.avetestacc=0
    def test(self):
        testlosss=0
        correct=0
        total=0
        for j, dataa in enumerate(self.set):
            self.modle.eval()#将模型设置为测试，在含有dropout层等可变层时应该添加
            inputst, labelst = dataa
            inputst, labelst = inputst.to(self.device), labelst.to(self.device)
            outputst = self.modle(inputst)
            testlosss = testlosss + self.cri(outputst, labelst).item()
            _, predictedd = torch.max(outputst.data, 1)
            outlistt = predictedd.tolist()
            labelslistt = labelst.tolist()
            for q in range(0, self.batchsize):
                aq = outlistt[q]
                bq = labelslistt[q]
                if aq == bq:
                    correct = correct + 1
            total = total + 1
            #if total==2:
            #    break
        self.avetestloss=testlosss/total
        self.avetestacc = correct / (total * self.batchsize)
        return self.avetestloss,self.avetestacc