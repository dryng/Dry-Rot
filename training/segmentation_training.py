import os
import sys
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils.segmentation_utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_metrics,
    save_predictions_to_folder
)
from torch.utils.tensorboard import SummaryWriter
from utils.earlyStopping import EarlyStopping

sys.path.append('../')
from models.unet import UNET, TGIUNET, DiceLoss
sys.path.remove('../')

with open(sys.argv[1]) as json_data:
    model_config = json.load(json_data)

with open(sys.argv[2]) as json_data:
    results_config = json.load(json_data)

MODEL = model_config["MODEL"]
LEARNING_RATE = model_config["LEARNING_RATE"]
DEVICE = model_config["DEVICE"]
BATCH_SIZE = model_config["BATCH_SIZE"]
NUM_EPOCHS = model_config["NUM_EPOCHS"]
PATIENCE  = model_config["PATIENCE"]
NUM_WORKERS = model_config["NUM_WORKERS"]
IMAGE_HEIGHT = model_config["IMAGE_HEIGHT"]
IMAGE_WIDTH = model_config["IMAGE_WIDTH"]
PIN_MEMORY = model_config["PIN_MEMORY"] in ("True")
LOAD_MODEL = model_config["LOAD_MODEL"] in ("True")
MODEL_CHECKPOINT = model_config["MODEL_CHECKPOINT"]
DICE_LOSS = model_config["DICE_LOSS"] in ("True")
DATASET_PATH = ""
if "DATASET_PATH" in model_config:
    DATASET_PATH = model_config["DATASET_PATH"]

SAVE_CHECKPOINT = results_config["SAVE_CHECKPOINT"]
SAVE_METRICS = results_config["SAVE_METRICS"]
SAVE_TB_LOGS = results_config["SAVE_TB_LOGS"]

earlyStopping = EarlyStopping(PATIENCE)

if not os.path.exists(SAVE_TB_LOGS):
    os.makedirs(SAVE_TB_LOGS)
if not os.path.exists(SAVE_CHECKPOINT):
    os.makedirs(SAVE_CHECKPOINT)
if not os.path.exists(SAVE_METRICS):
    os.makedirs(SAVE_METRICS)

writer = SummaryWriter(f"{SAVE_TB_LOGS}")

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
    model.eval()
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

    model.train()
    return running_loss / len(loader)


def main():
    train_transform = A.Compose(
        [
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
    if MODEL == "UNET":
        print(f"Using UNET Model")
        model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    elif MODEL == "TGIUNET":
        print(f"Using TGIUNET Model")
        model = TGIUNET(in_channels=3, out_channels=1).to(device=DEVICE)
    else:
        print("NO VALID MODEL TYPE PASSED")
        return

    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if DATASET_PATH != "":
        train_loader, val_loader = get_loaders(
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
            DATASET_PATH
        )
    else:
        train_loader, val_loader = get_loaders(
            BATCH_SIZE,
            train_transform,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY
        )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_CHECKPOINT), model)

    scaler = torch.cuda.amp.GradScaler() # mixed percision for faster training 

    train_losses = []
    train_accuracies = []
    train_dice = []
    val_losses = []
    val_accuracies = []
    val_dice = []

    min_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)
        writer.add_scalar('Training Loss', train_loss, global_step=epoch)

        train_accuracy, dice_score = check_metrics(train_loader, model, device=DEVICE)
        train_accuracies.append(train_accuracy.cpu().numpy().tolist())
        writer.add_scalar('Training Accuracy', train_accuracy, global_step=epoch)
        writer.add_scalar('Training Dice Score', dice_score, global_step=epoch)
        train_dice.append(dice_score.cpu().numpy().tolist())

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        
        val_loss = eval(val_loader, model, loss_fn, epoch)
        val_losses.append(val_loss)
        writer.add_scalar('Validation Loss', val_loss, global_step=epoch)

        val_accuracy, dice_score = check_metrics(val_loader, model, device=DEVICE)
        val_accuracies.append(val_accuracy.cpu().numpy().tolist())
        writer.add_scalar('Validation Accuracy', val_accuracy, global_step=epoch)
        writer.add_scalar('Validation Dice Score', dice_score, global_step=epoch)
        val_dice.append(dice_score.cpu().numpy().tolist())

        if val_loss < min_val_loss:
            save_checkpoint(checkpoint, epoch=epoch, loss="DICE", folder=SAVE_CHECKPOINT)
            min_val_loss = val_loss

        if earlyStopping.training_completeV2(val_loss):
            break
        
    
    # dump results to json file
    metrics = {
        "minimum_loss" : earlyStopping.minimum_loss,
        "best_epoch" : earlyStopping.best_epoch,
        "train_losses" : train_losses,
        "train_accuracies" : train_accuracies,
        "train_dice" : train_dice,
        "val_losses" : val_losses,
        "val_accuracies" : val_accuracies,
        "val_dice" : val_dice,
    }

    with open(f"{SAVE_METRICS}/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)


if __name__ == "__main__":
    main()
