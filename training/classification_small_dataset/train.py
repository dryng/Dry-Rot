import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from custom_mobilenet_v3_small import customMNSmall
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_metrics,
    save_predictions_to_folder
)
from torch.utils.tensorboard import SummaryWriter

GPU_NUM = sys.argv[1]
LEARNING_RATE = 1e-3
DEVICE = f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu"  #cuda:1
BATCH_SIZE = 32
START_EPOCH = 0 # 0
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
MODEL = sys.argv[2]

writer = SummaryWriter(f'runs/{MODEL}_{LEARNING_RATE}/')

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

         # writer.add_scalar('Training Loss', loss.item(), global_step=((epoch * length) + batch_idx))
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
        # collapsing neurons to fast i think -> add more layers at end 
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
    
    # model = nn.DataParallel(model)
    model = model.to(device=DEVICE)
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
        print(f"Model: {MODEL}")
        load_checkpoint(torch.load(f"model_checkpoints/{MODEL}/epoch_{START_EPOCH - 1}_checkpoint.pth.tar"), model)
        check_metrics(val_loader, model, BATCH_SIZE, device=DEVICE)
        #print(f"=> saving predictin images to folder")
        #save_predictions_to_folder(val_loader, model, epoch=20, folder="model_predictions/labels_v2", loss="DICE", max=50, device=DEVICE)
        return
    scaler = torch.cuda.amp.GradScaler() # mixed percision for faster training 

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
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
        
        save_checkpoint(checkpoint, epoch=epoch, model=MODEL, folder="model_checkpoints_small")

        val_loss = eval(val_loader, model, loss_fn, epoch)
        val_losses.append(val_loss)
        writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
        
        val_accuracy, _, _, _ = check_metrics(val_loader, model, BATCH_SIZE, device=DEVICE)
        val_accuracies.append(val_accuracy.cpu().numpy().tolist())
        writer.add_scalar('Validation Accuracy', val_accuracy, global_step=epoch)
        
        #save_predictions_to_folder(val_loader, model, epoch, max=25, device=DEVICE)
    
    _, precision, recall, f1_score = check_metrics(val_loader, model, BATCH_SIZE, device=DEVICE)

    metrics = {
        "train_losses" : train_losses,
        "train_accuracies" : train_accuracies,
        "val_losses" : val_losses,
        "val_accuracies" : val_accuracies,
        "precision" : precision.cpu().numpy().tolist(),
        "recall" : recall.cpu().numpy().tolist(),
        "f1_score" : f1_score.cpu().numpy().tolist()
    }

    with open(f"../../metrics/classification/{MODEL}/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

if __name__ == "__main__":
    main()
