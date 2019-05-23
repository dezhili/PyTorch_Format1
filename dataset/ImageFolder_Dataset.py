import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


'''
Data storage format: class independent

../data/train/dog/xxx.jpg
../data/train/cat/xxx.jpg
../data/val/dog/xxx.jpg

'''



# define image transforms to do data augumentation
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val':
    transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# define data folder using ImageFolder to get images and classes from folder
root = '../data/'
data_folder = {
    'train':
    ImageFolder(
        os.path.join(root, 'train'), transform=data_transforms['train']),
    'val':
    ImageFolder(
        os.path.join(root, 'val'), transform=data_transforms['val'])
}

# define dataloader to load images
batch_size = 32
dataloader = {
    'train':
    DataLoader(
        data_folder['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4),
    'val':
    DataLoader(data_folder['val'], batch_size=batch_size, num_workers=4)
}

# get train data size and validation data size
data_size = {
    'train': len(dataloader['train'].dataset),
    'val': len(dataloader['val'].dataset)
}

# get numbers of classes
img_classes = len(dataloader['train'].dataset.classes)


