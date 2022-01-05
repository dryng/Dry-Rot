import torch
import torch.nn as nn
import torchvision.models as models
from custom_models import custom_mobilenet_v3_small, unet
from utils import load_checkpoint, numpy_to_torch


class SegmentationModel:
    """[summary]
        Run segmentation model (u-net by default) on given image and return prediction mask
    Args:
        model_name : name of model to use. currently onlt u-net is available
    Methods:
        prediction() : pass image to model to generate mask for
    """
    def __init__(self, model_name="u-net"):
        if model_name != "u-net":
            raise ValueError("Currently only u-net is supported for segmentation")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = unet.UNET(in_channels=3, out_channels=1).to(self.device)
        load_checkpoint(torch.load("checkpoints/unet.pth.tar"), self.model)
    
    def predict(self, image):
        """[summary]
            Run segmentation model (u-net by default) on given image and return prediction mask
            Takes in a numpy array and returns a numpy array
        Args:
            image (numpy array): image to generate mask for
        """
        data = numpy_to_torch(image)

        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            data = data.unsqueeze(0).to(self.device)
            prediction = torch.sigmoid(self.model(data))
            prediction = (prediction > 0.5).float()
            prediction = prediction.squeeze(0).cpu().numpy()
        
        return prediction
    

class ClassificationModel:
    """[summary]
        Run segmentation model (u-net by default) on given image and return prediction mask
        Takes in a numpy array and retuns an integer
    Args:
        model_name : currently available models: 
                        resnet_18
                        mobilenet_v3_small
                        custom_mobilenet_v3_small
                        mobilenet_v3_small
                        efficient_net_b3
                        efficient_net_b4
    Methods:
        prediction() : pass image to model to predict
    """
    def __init__(self, model_name):
        if model_name == "resnet_18":
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
            load_checkpoint(torch.load("checkpoints/resnet_18.pth.tar"), model)
        elif model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=True)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, 1)
            load_checkpoint(torch.load("checkpoints/mobilenet_v3_small.pth.tar"), model)
        elif model_name == "custom_mobilenet_v3_small":
            pretrained = models.mobilenet_v3_small(pretrained=True)
            model = customMNSmall(pretrained)
            load_checkpoint(torch.load("checkpoints/custom_mobilenet_v3_small.pth.tar"), model)
        elif model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_large(pretrained=True)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, 1)
            load_checkpoint(torch.load("checkpoints/mobilenet_v3_small.pth.tar"), model)
        elif model_name == "efficient_net_b3":
            model = models.efficientnet_b3(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 1)
            load_checkpoint(torch.load("checkpoints/efficient_net_b3.pth.tar"), model)
        elif model_name == "efficient_net_b4":
            model = models.efficientnet_b4(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 1)
            load_checkpoint(torch.load("checkpoints/efficient_net_b4.pth.tar"), model)
        else:
            raise ValueError(f"Unsupported Model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

    def predict(self, image):
        """[summary]
            Run classification model on given image and return prediction. 1 for True and 0 for False
        Args:
            image (numpy array): image to generate prediction for
        """
        data = numpy_to_torch(image)

        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            data = data.unsqueeze(0).to(self.device)
            prediction = torch.sigmoid(self.model(data))
            prediction = (prediction > 0.5).float()
            prediction = int(prediction.cpu().item())

        return prediction

