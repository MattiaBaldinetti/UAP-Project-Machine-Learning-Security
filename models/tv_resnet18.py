import torch.nn as nn
from torchvision.models import resnet18

def TV_ResNet18_CIFAR10(num_classes=10) -> nn.Module:
    m = resnet18(weights=None)  
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m