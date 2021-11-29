import os
import numpy as np
from PIL import Image
import h5py
from torch.utils.data import Dataset

class DryRotDataset(Dataset):
    """[summary]

    Args:
        dset: bit representing train (0), val (1), or test (2) dataset
        path: path to h5 file containing the data
    """
    def __init__(self, dset=0, path='/work/dryngler/dry_rot/datasets/small_segmentation_dataset.h5', transform=None):
        self.dset = dset
        self.path = path
        self.transform = transform
    
    def __len__(self):
        with h5py.File(self.path) as f:
            if self.dset == 0:
                return len(f.get("small_X_train"))
            elif self.dset == 1:
                return len(f.get("small_X_val"))
            elif self.dset == 2:
                return len(f.get("small_X_test"))
    
    def __getitem__(self, idx):
        with h5py.File(self.path) as f:
            if self.dset == 0:
                image = np.array(f["small_X_train"][idx])
                mask = np.array(f["small_Y_train"][idx], dtype='f')
            elif self.dset == 1:
                image = np.array(f["small_X_val"][idx])
                mask = np.array(f["small_Y_val"][idx], dtype='f')
            elif self.dset == 2:
                image = np.array(f["small_X_test"][idx])
                mask = np.array(f["small_Y_test"][idx], dtype='f')
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
        
