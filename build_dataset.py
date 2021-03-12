from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset


def process(data, option=None):
    if option == 'load_path':
        return torch.from_numpy(np.array(Image.open(data)))
        
    elif option == 'load_direct':
        None
    else:
        print("please select option")

class my_transform (object):
    def __init__(self, process):
        self.process = process
    
    def __call__(self, data):        
        return self.process(data)

class SimpleDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.transform(self.data_list[idx])

        return sample