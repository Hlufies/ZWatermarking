import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

class IPDataset(Dataset):
    def __init__(self, dataset_path):
        assert os.path.exists(dataset_path) and dataset_path.split('.')[-1] == 'pth'
        dataset = torch.load(dataset_path).to(torch.device('cpu'))
        self.latents = dataset
    def __getitem__(self, index):
        return self.latents[index], 0
    def __len__(self):
        return len(self.latents)
    
class SourceTargetDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
    def __getitem__(self, index):
        image_1, label_1 = self.source.__getitem__(index)
        image_2, label_2 = self.target.__getitem__(index)
        assert label_1 == label_2
        return image_1, image_2
    def __len__(self):
        l1 = self.source.__len__()
        l2 = self.target.__len__()
        assert l1 == l2
        return l1

def get_dataset(name='IP', root='data',opt=None):
    train_target = IPDataset(opt.dataset_path)
    train_source = IPDataset(opt.dataset_path)
    train_source_target = SourceTargetDataset(train_source, train_target)
    return 0, train_source_target, train_source, train_target
        
 