import sys
import torch
sys.path.append("../../")
from models import mobilenet_v3_small

model = mobilenet_v3_small.model

print(model)
