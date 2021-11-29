import torch
import torchvision
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from torchmetrics import IoU
from dataset import DryRotDataset

def save_checkpoint(state, epoch, loss, filename="unet_checkpoint.pth.tar", folder="model_checkpoints"):
    """[summary]

    Args:
        state ([type]): [description]
        filename (str, optional): [description]. Defaults to "unet_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    filename = f"{folder}/{loss}/{loss}_epoch_{epoch}_{filename}"
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """[summary]

    Args:
        checkpoint ([type]): [description]
        model ([type]): [description]
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    batch_size,
    train_transforms,
    val_transforms,
    num_workers=1,
    pin_memory=True,
    path=None,
):
    """[summary]

    Args:
        batch_size ([type]): [description]
        train_transforms ([type]): [description]
        val_transforms ([type]): [description]
        num_workers (int, optional): [description]. Defaults to 2.
        pin_memory (bool, optional): [description]. Defaults to True.
        path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if path is not None:
        train_dataset = DryRotDataset(
            dset=0,
            path=path,
            transform=train_transforms
        )
        
        val_dataset = DryRotDataset(
            dset=1,
            path=path,
            transform=val_transforms
        )
    else:
        train_dataset = DryRotDataset(
            dset=0,
            transform=train_transforms
        )
        
        val_dataset = DryRotDataset(
            dset=1,
            transform=val_transforms
        )
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    """[summary]
    dice_score: better measure of accuracy.

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        device (str, optional): [description]. Defaults to "cuda".
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    ioU = IoU(num_classes=1)
    ioU_avg = 0
    model.eval()
    
    with torch.no_grad():
        for _, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.permute(0,3,1,2).to(device)
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == targets).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * targets).sum()) / (
                (predictions + targets).sum() + 1e-8
            )
            #ioU_avg += ioU(predictions, targets)
            
    print(
        f"=> {num_correct}/{num_pixels} correct. Accuracy: {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"IoU: {ioU_avg/len(loader)}")
    model.train()

def plot_losses(train_losses, eval_losses):
    plt.plot(train_losses, '-o')
    plt.plot(eval_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.show()

    
def save_predictions_to_folder(loader, model, epoch, max=None, folder="model_predictions", loss="DICE", device="cuda"):
    """[summary]

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        epoch: curr epoch,
        max: number of images to save
        folder (str, optional): [description]. Defaults to "model_predictions/".
        device (str, optional): [description]. Defaults to "cuda".
    """
    model.eval()
    for idx, (data, targets) in enumerate(loader):
        if idx == max:
            return
        with torch.no_grad():
            data = data.to(device=device)
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()

        #image_target_combined = overlay(data.detach().clone().cpu().numpy(), targets.detach().clone().cpu().numpy(), color=(0,0,0), alpha=1)
        #image_mask_combined = overlay(image_target_combined, predictions.detach().clone().cpu().numpy())
        #torchvision.utils.save_image(torch.from_numpy(image_mask_combined), f"{folder}/{loss}/{epoch}/target_overlay_{idx}.png")

        image_mask_combined = overlay(data.detach().clone().cpu().numpy(), predictions.detach().clone().cpu().numpy(), color=(255,0,0), alpha=.5)
        torchvision.utils.save_image(torch.from_numpy(image_mask_combined), f"{folder}/{loss}/{epoch}/overlay_{idx}.png")

        torchvision.utils.save_image(data, f"{folder}/{loss}/{epoch}/img_{idx}.png")
        torchvision.utils.save_image(predictions, f"{folder}/{loss}/{epoch}/pred_{idx}.png")
        torchvision.utils.save_image(targets.permute(0,3,1,2), f"{folder}/{loss}/{epoch}/target_{idx}.png")

        #grid = torchvision.utils.make_grid([data.cpu(), torch.from_numpy(image_mask_combined)], nrow=2)
        #torchvision.utils.save_image(grid, f"{folder}/{epoch}/grid_{idx}.png")

    model.train()


def overlay(
    image,
    mask,
    color=(255, 0, 0),
    alpha=0.5, 
    resize=None
):
    """Combines image and its segmentation mask into a single image.

    https://www.kaggle.com/purplejester/showing-samples-with-segmentation-mask-overlay
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
        
    """
    color = np.asarray(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    if resize is not None:
        image = cv.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv.resize(image_overlay.transpose(1, 2, 0), resize)
    
    image_combined = cv.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    return image_combined
        
