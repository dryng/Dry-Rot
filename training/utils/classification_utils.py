import sys
import torch
import torchvision
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from torchmetrics import IoU
sys.path.append('../')
from dataset.classification_dataset import DryRotDataset
sys.path.remove('../')

def save_checkpoint(state, epoch, filename="checkpoint.pth.tar", folder="model_checkpoints"):
    """[summary]

    Args:
        state ([type]): [description]
        filename (str, optional): [description]. Defaults to "unet_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    filename = f"{folder}/epoch_{epoch}_{filename}"
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

def check_metrics(loader, model, batchsize, device="cuda"):
    """[summary]
    Binary Classification Accuracy
    Binary Classification Precision
    Binary Classification Recall
    Binary Classification F1-Score (harmonic mean)

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        device (str, optional): [description]. Defaults to "cuda".
    """
    model.eval()
    length = len(loader) * batchsize

    true_positive = 0 
    false_positive = 0 
    true_negative = 0 
    false_negative = 0 

    num_correct = 0

    print("=> Checking metrics")
        
    with torch.no_grad():
        for _, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.unsqueeze(1).to(device)
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()

            true_positive += (torch.logical_and(predictions == 1, targets == 1)).sum()
            false_positive += (torch.logical_and(predictions == 1, targets == 0)).sum() 
            true_negative += (torch.logical_and(predictions == 0, targets == 0)).sum()
            false_negative += (torch.logical_and(predictions == 0, targets == 1)).sum()

            num_correct += (predictions == targets).sum()
    
    precision = true_positive / (true_positive + false_positive)  
    recall = true_positive / (true_positive + false_negative) 
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy = num_correct/length*100

    print(f"=> Accuracy: {num_correct}/{length} => {accuracy:.3f}")
    print(f"=> Precision: {true_positive}/{true_positive + false_positive} => {precision:.3f}")
    print(f"=> Recall: {true_positive}/{(true_positive + false_negative)} => {recall:.3f}")
    print(f"=> F1-Score (harmonic mean): {f1_score:.3f}")
    print()
    
    model.train()

    return accuracy, precision, recall, f1_score

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

        image_mask_combined = overlay(data.detach().clone().cpu().numpy(), predictions.detach().clone().cpu().numpy(), color=(255,0,0), alpha=.5)
        torchvision.utils.save_image(torch.from_numpy(image_mask_combined), f"{folder}/{loss}/{epoch}/overlay_{idx}.png")

        torchvision.utils.save_image(data, f"{folder}/{loss}/{epoch}/img_{idx}.png")
        torchvision.utils.save_image(predictions, f"{folder}/{loss}/{epoch}/pred_{idx}.png")
        torchvision.utils.save_image(targets.permute(0,3,1,2), f"{folder}/{loss}/{epoch}/target_{idx}.png")

    model.train()

        
