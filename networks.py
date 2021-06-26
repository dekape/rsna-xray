
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, requires_grad=False):
    """https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"""
    for param in model.parameters():
        param.requires_grad = requires_grad
    return None
            

# Example building model from scratch
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.train_count=0
        
        self.activation = nn.ReLU()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1)
        self.b1 = nn.BatchNorm2d(6)
        self.s2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=2)
        self.b3 = nn.BatchNorm2d(16)
        self.s4 = nn.MaxPool2d(kernel_size=2, padding=0, stride=5)
        self.f5 = nn.Linear(16*15*15, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)

    def forward(self, x):
        # print("0", x.shape, x.min(), x.max())
        x = self.activation(self.b1(self.c1(x)))
        # print("1", x.shape, x.min(), x.max())
        x = self.activation(self.s2(x))
        # print("2", x.shape, x.min(), x.max())
        x = self.activation(self.b3(self.c3(x)))
        # print("3", x.shape, x.min(), x.max())
        x = self.activation(self.s4(x))
        # print("4", x.shape, x.min(), x.max())
        x = self.activation(self.f5(x.view(x.shape[0], -1)))
        # print("5", x.shape, x.min(), x.max())
        x = self.activation(self.f6(x))
        # print("6", x.shape,x.min(), x.max())
        x = self.output(x)
        # print("7", x.shape, x.min(), x.max())    
        return x


# Example using torchvision models
class ResNet50Wrapper():
    def __init__(self):
        return None        
    def __new__(self, num_classes=10):
        resnet50 = models.resnet50(num_classes=num_classes)
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # make single channel
        return resnet50
    
    
class ResNet18Wrapper():
    def __init__(self):
        return None        
    def __new__(self, num_classes=10):
        resnet18 = models.resnet18(num_classes=num_classes)
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # make single channel
        return resnet18


# Example using pretrained torchvision models    
class ResNet18_TransferLearn():
    def __init__(self):
        return None        
    def __new__(self, num_classes=10):
        resnet18 = models.resnet18(pretrained=True)
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # make single channel
        resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        return resnet18
        
    
class ResNet18_FeatureExtract():
    def __init__(self):
        return None        
    def __new__(self, num_classes=10):
        resnet18 = models.resnet18(pretrained=True)
        
        # Turn off requires grad
        set_parameter_requires_grad(resnet18, requires_grad=False)
        
        # Replace first and last layers (those automatically have params with requires_grad=True)
        resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        return resnet18

class InceptionV3():
    def __init__(self):
        return None        
    def __new__(self, num_classes=10):
        inceptionv3 = models.inception_v3(init_weights=True)
        inceptionv3.fc = nn.Linear(inceptionv3.fc.in_features, num_classes)
        return inceptionv3
