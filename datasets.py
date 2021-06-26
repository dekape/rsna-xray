import torch
from torch.utils.data import Dataset 
from PIL import Image
from imageio import imread
import numpy as np

    
class XRAYTensorDataset(Dataset):
    def __init__(self, img_paths, targets=None, transform=None):
        """
        Args:
            data (list): a list containing the path of the images
            targets (list): A list containing all the labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert len(img_paths) == len(targets)
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label = self.img_paths[idx], self.targets[idx]
        try:
            sample = torch.from_numpy(imread(img_path))
        except Exception as e:
            print("!!!!!!!!!!! Couldn't read ", img_path, type(img_path), len(img_path))
            raise(e)
        if len(sample.shape) == 3: sample = torch.mean(sample.float(), 2) # take mean of 3-channel images along the channel axis, expecting b&w
        sample = sample.unsqueeze(0).float()/255.                         # unsqueeze to add channel dimension
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(label).long()
  
    
class XRAY3C_TensorDataset(Dataset):
    def __init__(self, img_paths, targets=None, transform=None):
        """
        This will read the xray b&w image, but expand into a 3channel image so that networks like ResNet can be used for feature extraction
        Args:
            data (list): a list containing the path of the images
            targets (list): A list containing all the labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert len(img_paths) == len(targets)
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, label = self.img_paths[idx], self.targets[idx]
        try:
            sample = torch.from_numpy(imread(img_path))
        except Exception as e:
            print("!!!!!!!!!!! Couldn't read ", img_path, type(img_path), len(img_path))
            raise(e)
        if len(sample.shape) == 3: sample = torch.mean(sample.float(), 2) # take mean of 3-channel images along the channel axis, expecting b&w
        sample = sample.unsqueeze(0).float()/255.                         # unsqueeze to add channel dimension
        sample = sample.expand(3, -1, -1)
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(label).long()


    