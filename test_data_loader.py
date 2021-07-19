from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import os
def testdata(batchsize):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = datasets.CIFAR10(root=os.getcwd(),train=False,transform=transform_test, target_transform=None, download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True, num_workers=0)
    return test_loader


# if you wang to use your own dataset
# you need to rewriter this class
#     import torch.utils.data as data
#     class myDataset(data.Dataset):#继承data.Dataset
#         def __init__(self):
#             # TODO
#             # 1. Initialize file path or list of file names.将数据传入，具体使用什么样的数据结构取决于你的训练数据
            
#         def __getitem__(self, index):
#             # TODO
#             # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#             # 2. Preprocess the data (e.g. torchvision.Transform).
#             # 3. Return a data pair (e.g. image and label).
#             #即对应每一个index 返回一个data 和 label，注意需要为tensor类型
#         def __len__(self):
#             # You should change 0 to the total size of your dataset.
#             #返回数据集的长度，即index+1最大值
#             return 0
#       test_data=myDataset()
#       test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True, num_workers=0)#num——workers为并行加载数
