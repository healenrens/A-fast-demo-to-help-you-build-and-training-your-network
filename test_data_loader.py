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