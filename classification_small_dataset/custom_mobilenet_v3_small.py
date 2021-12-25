import torch
import torch.nn as nn
class customMNSmall(nn.Module):
    def __init__(self, pretrained):
        super(customMNSmall, self).__init__()
        num_ftrs = pretrained.classifier[0].in_features
        self.pretrained = pretrained
        self.pretrained.classifier[0] = nn.Linear(num_ftrs, num_ftrs, bias=False)
        self.pretrained.classifier[3] = nn.Linear(num_ftrs, num_ftrs // 2)
        self.extension = nn.Sequential(nn.Linear(num_ftrs // 2, 64),
                                       nn.Linear(64, 1))
    def forward(self, x):
        x = self.pretrained(x)
        x = self.extension(x)
        return x