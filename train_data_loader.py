from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import os
def traindata(batchsize):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(root=os.getcwd(),train=True,transform=transform_train, target_transform=None, download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True, num_workers=0)
    return train_loader