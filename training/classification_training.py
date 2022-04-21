import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import albumentations as A
from PIL import Image
from torchsummary import summary    
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils.classification_utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_metrics,
    save_predictions_to_folder
)

from torch.utils.tensorboard import SummaryWriter
from utils.earlyStopping import EarlyStopping

sys.path.append('../models')
from custom_mobilenet_v3_small import customMNSmall
sys.path.remove('../models')

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

'''
GPU_NUM = sys.argv[1]
LEARNING_RATE = 1e-3
DEVICE = f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu"  #cuda:1
BATCH_SIZE = 32
START_EPOCH = 0 
NUM_EPOCHS = 100
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
MODEL = sys.argv[2]
'''

def train(loader, model, optimizer, loss_fn, scaler, epoch):
    """[summary]

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
         targets = torch.unsqueeze(targets, 1).to(device=DEVICE)
         
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

    return running_loss / length


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
         targets = torch.unsqueeze(targets, 1).to(device=DEVICE)
         
         # float16 to speed up training
         with torch.cuda.amp.autocast():
             predictions = model(data)
             loss = loss_fn(predictions, targets)
        
         loop.set_postfix(loss=loss.item())
         running_loss += loss.item()

         # writer.add_scalar('Validation Loss', loss.item(), global_step=((epoch * length) + batch_idx))
    model.train()
    return running_loss / length


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
    if MODEL == "resnet_18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif MODEL == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 1)
    elif MODEL == "custom_mobilenet_v3_small":
        # 576 -> 288 -> 64 -> 1
        pretrained = models.mobilenet_v3_small(pretrained=True)
        model = customMNSmall(pretrained)
    elif MODEL == "custom_mobilenet_v3_small_untrained":
        untrained = models.mobilenet_v3_small(pretrained=False)
        model = customMNSmall(untrained)
    elif MODEL == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 1)
    elif MODEL == "efficient_net_b3":
        model = models.efficientnet_b3(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
    elif MODEL == "efficient_net_b4":
        model = models.efficientnet_b4(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
    #print(f"MODEL: {model}")
    
    model = model.to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
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
        print(f"Loading Model From Checkpoint")
        load_checkpoint(torch.load(MODEL_CHECKPOINT), model)
       
    scaler = torch.cuda.amp.GradScaler() # mixed percision for faster training 

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    '''
    best_epoch = None
    global_val_loss = float('inf')
    earlyStopping = EarlyStopping(patience=10)

    images, targets = iter(train_loader).next()
    images = images.to(device=DEVICE)
    writer.add_graph(model, images)
    writer.close()
    '''

    min_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(train_loader, model, optimizer, loss_fn, scaler, epoch)
        train_losses.append(train_loss)
        writer.add_scalar('Training Loss', train_loss, global_step=epoch)

        train_acc, _, _, _ = check_metrics(train_loader, model, BATCH_SIZE, device=DEVICE)
        train_accuracies.append(train_acc.cpu().numpy().tolist())
        writer.add_scalar('Training Accuracy', train_acc, global_step=epoch)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        val_loss = eval(val_loader, model, loss_fn, epoch)
        val_losses.append(val_loss)
        writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
        
        val_accuracy, _, _, _ = check_metrics(val_loader, model, BATCH_SIZE, device=DEVICE)
        val_accuracies.append(val_accuracy.cpu().numpy().tolist())
        writer.add_scalar('Validation Accuracy', val_accuracy, global_step=epoch)

        if val_loss < min_val_loss:
            save_checkpoint(checkpoint, epoch, folder=SAVE_CHECKPOINT)
            min_val_loss = val_loss

        if earlyStopping.training_completeV2(val_loss):
            break

    _, precision, recall, f1_score = check_metrics(val_loader, model, BATCH_SIZE, device=DEVICE)

    metrics = {
        "minimum_loss" : earlyStopping.minimum_loss,
        "best_epoch" : earlyStopping.best_epoch,
        "train_losses" : train_losses,
        "train_accuracies" : train_accuracies,
        "val_losses" : val_losses,
        "val_accuracies" : val_accuracies,
        "precision" : precision.cpu().numpy().tolist(),
        "recall" : recall.cpu().numpy().tolist(),
        "f1_score" : f1_score.cpu().numpy().tolist()
    }

    with open(f"{SAVE_METRICS}/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

if __name__ == "__main__":
    main()
