import sys
import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
sys.path.append('../../training/')
from dataset.segmentation_dataset import DryRotDataset
sys.path.remove('../../training/')

#from inference import ClassificationModel
#model = SegmentationModel(checkpoints_path="/work/dryngler/dry_rot/Dry-Rot/inference/")

model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
load_checkpoint(torch.load(MODEL_CHECKPOINT), model)

DEVICE = "cuda:0"
THRES = .7

transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  
            ),
            ToTensorV2()
        ]
    )

dataset = DryRotDataset(
            dset=1,
            transform=transforms
        )

loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )

miss_count = 0
for idx, (data, targets) in enumerate(loader):
    data = data.to(DEVICE)
    targets = targets.permute(0,3,1,2).to(DEVICE)
    dice_score = (2 * (predictions * targets).sum()) / (
                (predictions + targets).sum() + 1e-8
            )
    if dice_score < THRES:
        # save to folder
        torchvision.utils.save_image(data, f"/space/dryngler/dry_rot/experiments/check_dice_threshold_Feb20,22/bad_imgs/{idx}.jpeg")
        miss_count += 1

print(f"Total misses: {miss_count}")





