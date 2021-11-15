import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics import IoU
from dataset import DryRotDataset

def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    """[summary]

    Args:
        state ([type]): [description]
        filename (str, optional): [description]. Defaults to "unet_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
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
    num_workers=2,
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
            0,
            path,
            train_transforms
        )
        
        val_dataset = DryRotDataset(
            1,
            path,
            val_transforms
        )
    else:
        train_dataset = DryRotDataset(
            0,
            train_transforms
        )
        
        val_dataset = DryRotDataset(
            1,
            val_transforms
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
        for data, targets in loader:
            data = data.to(device)
            targets = targets.unsqueeze(1).to(device)
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == targets).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * targets).sum()) / (
                (predictions + targets).sum() + 1e-8
            )
            ioU_avg += ioU(predictions, targets)
            
    print(
        f"=> {num_correct}/{num_pixels} correct. Accuracy: {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"IoU: {ioU_avg/len(loader)}")
    model.train()
    
def save_predictions_to_folder(loader, model, epoch, max=None, folder="model_predictions/", device="cuda"):
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
    for idx, (data, targets) in enumerate(loader[:max]):
        with torch.no_grad():
            data = data.to(device=device)
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()
        torchvision.utils.save_image(predictions, f"{folder}/{epoch}/pred_{idx}.png")
        torchvision.utils.save_image(targets.unsqueeze(1), f"{folder}/{epoch}/target_{idx}.png")
        