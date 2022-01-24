import json
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET,TGIUNET, DiceLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_metrics,
    save_predictions_to_folder
)
from torch.utils.tensorboard import SummaryWriter

LEARNING_RATE = 1e-4 # 3e-5 from 35 (34 from 0) on
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2 # 32  # change back to train
NUM_EPOCHS = 1  # change back to 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
DICE_LOSS = True

writer = SummaryWriter(f'runs/u-net_{LEARNING_RATE}/')

def train(loader, model, optimizer, loss_fn, scaler):
    """[summary]

    Args:
        loader ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        loss_fn ([type]): [description]
        scaler ([type]): [description]
    """
    loop = tqdm(loader)
    length = len(loader)
    running_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
         data = data.to(device=DEVICE)
         targets = targets.permute(0,3,1,2).to(device=DEVICE)
         
         # float16 to speed up training
         with torch.cuda.amp.autocast():
             predictions = model(data)
             loss = loss_fn(predictions, targets)
        
         optimizer.zero_grad()
         scaler.scale(loss).backward()
         scaler.step(optimizer)
         scaler.update()
        
         loop.set_postfix(loss=loss.item())
         running_loss += loss.item()
    
    return running_loss / len(loader)


def eval(loader, model, loss_fn, epoch):
    """[summary]
        Calculate the validation loss over 1 epoch
    Args:
        loader ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        loss_fn ([type]): [description]
        scaler ([type]): for mixed precision
        epoch ([type])): current epoch to track total steps for tensorboard
    """
    loop = tqdm(loader)
    length = len(loader)
    running_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
         data = data.to(device=DEVICE)
         targets = targets.permute(0,3,1,2).to(device=DEVICE)
         
         # float16 to speed up training
         with torch.cuda.amp.autocast():
             predictions = model(data)
             loss = loss_fn(predictions, targets)
        
         loop.set_postfix(loss=loss.item())
         running_loss += loss.item()

    return running_loss / len(loader)


def main():
    train_transform = A.Compose(
        [
            # data agumentation? -> already have 200K samples
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  
            ),
            ToTensorV2()
        ]
    )
    
    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  
            ),
            ToTensorV2()
        ]
    )
    
    print(f"Device: {DEVICE}. Device count: {torch.cuda.device_count()}")
    # model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    model = TGIUNET(in_channels=3, out_channels=1).to(device=DEVICE)

    if DICE_LOSS:
        loss_fn = DiceLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("model_checkpoints_small/labels_v2/1e4/DICE/DICE_epoch_19_unet_checkpoint.pth.tar"), model)
        # Remove for training
        print(f"=> saving predictin images to folder")
        save_predictions_to_folder(val_loader, model, epoch=20, folder="model_predictions/labels_v2", loss="DICE", max=50, device=DEVICE)
        return
    scaler = torch.cuda.amp.GradScaler() # mixed percision for faster training 

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)
        writer.add_scalar('Training Loss', train_loss, global_step=epoch)

        train_accuracy, dice_score = check_metrics(train_loader, model, device=DEVICE)
        train_accuracies.append(train_accuracy.cpu().numpy().tolist())
        writer.add_scalar('Training Accuracy', train_accuracy, global_step=epoch)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        
        if DICE_LOSS:
            save_checkpoint(checkpoint, epoch=epoch, loss="DICE", folder="model_checkpoints_small")
        else:
            save_checkpoint(checkpoint, epoch=epoch, loss="BCE", folder="model_checkpoints_small/labels_v2")
        
        val_loss = eval(val_loader, model, loss_fn, epoch)
        val_losses.append(val_loss)
        writer.add_scalar('Validation Loss', val_loss, global_step=epoch)

        val_accuracy, dice_score = check_metrics(val_loader, model, device=DEVICE)
        val_accuracies.append(val_accuracy.cpu().numpy().tolist())
        writer.add_scalar('Validation Accuracy', val_accuracy, global_step=epoch)
        #save_predictions_to_folder(val_loader, model, epoch, max=25, device=DEVICE)
        
    # dump results to json file
    metrics = {
        "train_losses" : train_losses,
        "train_accuracies" : train_accuracies,
        "val_losses" : val_losses,
        "val_accuracies" : val_accuracies
    }

    with open("../../metrics/segmentation/u-net/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

if __name__ == "__main__":
    main()
