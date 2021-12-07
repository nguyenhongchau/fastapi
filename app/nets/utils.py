import torch
from torch.utils.data import Dataset
from mlxtend.data import loadlocal_mnist

from typing import Tuple, Any
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
    
class CustomedMNISTDataset(Dataset):
    """Build MNIST Dataset from data_path
    Image and their label files should be named as "name_images" and "name_lables" correspondingly
    """
    
    def __init__(self, data_path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        datafiles = [(join(data_path,f), join(data_path,f.replace("images", "labels"))) \
                     for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith("images")]

        self.data = np.zeros((0, 784), dtype=np.uint8)
        self.targets = np.zeros((0), dtype=np.uint8)
        for img_file, label_file in datafiles:
            if isfile(img_file) and isfile(label_file):
                data, targets = loadlocal_mnist(images_path=img_file, labels_path=label_file)
                self.data = np.concatenate((self.data, data))
                self.targets = np.concatenate((self.targets, targets))
            
        self.data = torch.from_numpy(self.data.reshape(len(self.data), 28, 28))
        self.targets = torch.from_numpy(self.targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:    
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target