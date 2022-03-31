import sys
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
sys.path.append('../../training/')
from dataset.segmentation_dataset import DryRotDataset
sys.path.remove('../../training/')
sys.path.append('../../')
from models.unet import UNET
sys.path.remove('../../')
sys.path.append('../../training/')
from utils.segmentation_utils import load_checkpoint, check_metrics
sys.path.remove('../../training/')
from matplotlib import pyplot as plt



DEVICE = "cuda:1"
MODEL_CHECKPOINT = "/space/dryngler/dry_rot/experiments/4/checkpoints/DICE_epoch_96_unet_checkpoint.pth.tar"
# MODEL_CHECKPOINT = "/space/dryngler/dry_rot/experiments/clean_build_Feb27,22/checkpoints/DICE_epoch_37_unet_checkpoint.pth.tar"
THRES = .1
TARG_THRES = .05

model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
load_checkpoint(torch.load(MODEL_CHECKPOINT), model)
model.eval()

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

print("Running")

miss_count = 0
scores = []
tdc = 0
skip = 0
with torch.no_grad():
    for idx, (data, targets) in enumerate(loader):
        data = data.to(DEVICE)

        target_precent = np.sum(targets.numpy(), where=[1]) / (256 * 256)
        if target_precent < TARG_THRES:
            skip += 1
            continue
        targets = targets.permute(0,3,1,2).to(DEVICE)
        predictions = torch.sigmoid(model(data))
        predictions = (predictions > 0.5).float()
        dice_score = (2 * (predictions * targets).sum()) / (
                (predictions + targets).sum() + 1e-8
            )
        scores.append(dice_score.cpu())
        tdc += dice_score 
        if dice_score <= THRES: 
            # save to folder 
            # print(data.size())
            # print(targets.size())
            # print(predictions.size())
            torchvision.utils.save_image(data, f"/space/dryngler/dry_rot/experiments/check_dice_threshold_Feb20,22/bad_imgs_filtered/image{idx}.jpeg") 
            torchvision.utils.save_image(predictions, f"/space/dryngler/dry_rot/experiments/check_dice_threshold_Feb20,22/bad_imgs_filtered/prediction{idx}.jpeg")
            torchvision.utils.save_image(targets, f"/space/dryngler/dry_rot/experiments/check_dice_threshold_Feb20,22/bad_imgs_filtered/target{idx}.jpeg")
            miss_count += 1 

        if miss_count == 20:
            break

scores = np.array(scores)
fig, ax = plt.subplots(figsize = (10, 7))
# ax.hist(scores, bins = [0, .25, .50, .75, 1])
ax.hist(scores, bins = [0, .1, .3, .5, .7, .9, 1])
plt.savefig('hist_filtered.png')

print(f"Len of loader: {len(loader)}")
print(f"Total dice score: {tdc / (len(loader) - skip)}")
print(f"Total misses: {miss_count}")





