import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from load_my_custom import *


def get_class_distribution(dataloader, num_classes):
    labels = []
    with torch.no_grad():
        for X, y in dataloader:
            labels.append(y)
    
    keys = range(0, num_classes)
    class_distribution = {key: None for key in keys}
    for key in class_distribution:
        class_distribution[key] = 0

    for i in range(len(labels)):
        class_distribution[labels[i]] += 1
    
    return class_distribution

data = get_data(1,transform)

dataloader_train, dataloader_val, dataloader_test = create_dataloaders(data, 32)

dict = get_class_distribution(dataloader_train, 9)

print(dict)