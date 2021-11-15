import os
import numpy as np
from PIL import Image
import h5py
from torch.utils.data import Dataset

class DryRotDataset(Dataset):
    """[summary]

    Args:
        set: bit representing train (0), val (1), or test (2) dataset
        path: path to h5 file containing the data
    """
    def __init__(self, set=0, path='/work/dryngler/dry_rot/datasets/segmentation_dataset.h5', transform=None):
        self.set = set
        self.path = path
        self.transform = transform
    
    def __len__(self):
        with h5py.File(self.path) as f:
            if self.set == 0:
                return len(f.get("X_train"))
            elif self.set == 1:
                return len(f.get("X_val"))
            elif self.set == 2:
                return len(f.get("X_test"))
    
    def __getitem__(self, idx):
        with h5py.File(self.path) as f:
            if self.set == 0:
                image = np.array(f["X_train"][idx])
                mask = np.array(f["Y_train"][idx], dtype=float32)
            elif self.set == 1:
                image = np.array(f["X_val"][idx])
                mask = np.array(f["Y_val"][idx], dtype=float32)
            elif self.set == 2:
                image = np.array(f["X_test"][idx])
                mask = np.array(f["Y_test"][idx], dtype=float32)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
        