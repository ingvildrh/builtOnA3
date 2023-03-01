from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
#from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale

import os
import utils
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision
from torch import optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


COLOR = 1
if COLOR: transform = Compose([Resize((256, 256)), ToTensor(), transforms.Normalize([0.,], [1.])])
else: transform = Compose([Grayscale(1), Resize((256, 256)), ToTensor(), transforms.Normalize([0.,], [1.])])

def get_data(LOCAL, transform):
    if LOCAL:
        dataset = ImageFolder('C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/full_fishdata', transform=transform)
    else: 
        dataset = ImageFolder('/home/ingvilrh/Documents/full_fishdata-20230227T075012Z-001/full_fishdata', transform=transform)
    print('Loaded', len(dataset), 'images')
    return dataset


def create_dataloaders(dataset, batch_size):
    #Splitting the dataset into random subsets for training, validation and testing
    train_size = int(0.8 * len(dataset))
    val_size = test_size = (len(dataset) - train_size) // 2
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    
    indices = list(range(len(train_data)))
    split_idx = int(np.floor(0.8 * len(train_data)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(train_data,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=1,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(train_data,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=1)

    dataloader_test = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=1)

    return dataloader_train, dataloader_val, dataloader_test

def generate_class_dict(data):
    class_lst = data.classes
    keys = range(0, len(class_lst))
    class_dict = {key: None for key in keys}
    for i in range(len(class_lst)):
        class_dict[i] = class_lst[i]
    return class_dict

def count_classes(dataset):
    print("Your classes are: ", dataset.classes)
    return len(dataset.classes)