import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.autograd import Variable

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

class DoubleConv(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # perserve spatial dimenstion and don't need bias cause batch norm cancels it
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """[summary]

    Args:
        in_channels: default to RGB
        out_channels: default binary segmentation
        features: default channels through network
    """
    def __init__(
        self, 
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512]
        ):
        super(UNET, self).__init__()
        self.encode = nn.ModuleList() # allow to do eval etc - look into it
        self.decode = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder 
        for feature in features:
            self.encode.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Decoder
        for feature in reversed(features):
            self.decode.append(
                nn.ConvTranspose2d(
                    feature*2, # *2 becuase adding the skip connection back in and concat across channels
                    feature,
                    kernel_size=2, stride=2 # double size
                )
            )
            self.decode.append(DoubleConv(feature*2, feature)) # after going up with transpose, now we go across
            
        self.bottle_neck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        
    def forward(self, x):
        skip_connections = [] # to store connections
        
        for e in self.encode:
            x = e(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decode), 2):
            x = self.decode[idx](x) # transpose
            skip_connection = skip_connections[idx//2]
            
            # if input image isn't divisible by 16 then ..
            # pooling will take the floor of / 2 so the (h,w) of x  
            # will be slightly different than the skip from decoding
            # and we won't be able to concatenate
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # dim = 1 for channels. (b, c, h, w)
            x = self.decode[idx+1](concat_skip)
            
        return self.final_conv(x)
        
    
class ThresholdLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.rand(1), requires_grad=True)
        
    def forward(self, x):
        #print(x)
        x = x[:,1,:,:]-(0.39*x[:,0,:,:])-(0.61*x[:,2,:,:])
        # x = torch.add(x[:,1,:,:],torch.add(torch.neg(torch.mul(x[:,0,:,:],0.39)),torch.neg(torch.mul(x[:,2,:,:],0.61))))
        flattened = torch.flatten(x)
        min = torch.min(x)
        max = torch.max(x)
        x = (x-min)/(max-min)
        #print(x)

        x=torch.unsqueeze(x,dim=1)
        normalized_threshold = torch.sigmoid(self.threshold)
        # normalized_threshold = torch.tensor(1).to(device=DEVICE)
        # print(normalized_threshold)
        ones = torch.ones(x.size()).to(device=DEVICE)
        zeros = torch.zeros(x.size()).to(device=DEVICE)
        out = torch.where(x>normalized_threshold,ones,zeros)
        #print(x)
        
        # out = torch.sigmoid(x - normalized_threshold) * (ones - zeros) + zeros
        
        # print(out)
        # print(out.size())
        return out
        

class TGIUNET(nn.Module):
    """[summary]

    Args:
        in_channels: default to RGB
        out_channels: default binary segmentation
        features: default channels through network
    """
    def __init__(
        self, 
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512]
        ):
        super(TGIUNET, self).__init__()
        self.threshold_layer = ThresholdLayer()
        self.encode = nn.ModuleList() # allow to do eval etc - look into it
        self.decode = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder 
        for feature in features:
            self.encode.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Decoder
        for feature in reversed(features):
            self.decode.append(
                nn.ConvTranspose2d(
                    feature*2, # *2 becuase adding the skip connection back in and concat across channels
                    feature,
                    kernel_size=2, stride=2 # double size
                )
            )
            self.decode.append(DoubleConv(feature*2, feature)) # after going up with transpose, now we go across
            
        self.bottle_neck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        
    def forward(self, x):
        skip_connections = [] # to store connections
        
        TGI_mask = self.threshold_layer(x)

        for e in self.encode:
            x = e(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decode), 2):
            x = self.decode[idx](x) # transpose
            skip_connection = skip_connections[idx//2]
            
            # if input image isn't divisible by 16 then ..
            # pooling will take the floor of / 2 so the (h,w) of x  
            # will be slightly different than the skip from decoding
            # and we won't be able to concatenate
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # dim = 1 for channels. (b, c, h, w)
            x = self.decode[idx+1](concat_skip)
        
        output_unet = self.final_conv(x)
        output_unet = torch.sigmoid(output_unet)   

        # output = torch.logical_and(TGI_mask,output_unet).int()
        output = torch.mul(output_unet,TGI_mask)
        
        return output
    

class DiceLoss(nn.Module):
    """[summary]
        Dice Score Loss Function
        Taken from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    Args:
        
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):   
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

def test_shape():
    x = torch.randn((3,1,256,256))
    model = UNET(in_channels=1, out_channels=1)
    ouput = model(x)
    print(ouput.shape)
    print(x.shape)
    assert ouput.shape == x.shape


if __name__ == "__main__":
    test_shape()