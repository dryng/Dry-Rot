import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET, DiceLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_to_folder
)

LEARNING_RATE = 1e-4 # 3e-5 from 35 (34 from 0) on
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # 32  # change back to train
NUM_EPOCHS = 20  # change back to 20
NUM_WORKERS = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = True
DICE_LOSS = True

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
    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
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
    
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        if DICE_LOSS:
            save_checkpoint(checkpoint, epoch=epoch, loss="DICE", folder="model_checkpoints_small/labels_v2/1e4")
        else:
            save_checkpoint(checkpoint, epoch=epoch, loss="BCE", folder="model_checkpoints_small/labels_v2")
        
        check_accuracy(val_loader, model, device=DEVICE)
        
        #save_predictions_to_folder(val_loader, model, epoch, max=25, device=DEVICE)

if __name__ == "__main__":
    main()
