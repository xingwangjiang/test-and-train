import os
import torch
from torchvision import models
from torchsummary import summary

model_AlexNet=torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3,out_channels=48,kernel_size=(11,11),stride=(4,4),padding=0),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm2d(num_features=48),
    torch.nn.MaxPool2d(kernel_size=(3,3),stride=(2,2)),

    torch.nn.Conv2d(in_channels=48,out_channels=128,kernel_size=(5,5),stride=(1,1),padding=2),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm2d(num_features=128),
    torch.nn.MaxPool2d(kernel_size=(3,3),stride=(2,2)),

    torch.nn.Conv2d(in_channels=128,out_channels=192,kernel_size=(3,3),stride=(1,1),padding=1),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm2d(num_features=192),

    torch.nn.Conv2d(in_channels=192,out_channels=192,kernel_size=(3,3),stride=(1,1),padding=1),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm2d(num_features=192),

    torch.nn.Conv2d(in_channels=192,out_channels=128,kernel_size=(3,3),stride=(1,1),padding=1),
    torch.nn.ReLU(inplace=True),
    torch.nn.BatchNorm2d(num_features=128),
    torch.nn.MaxPool2d(kernel_size=(3,3),stride=(2,2)),

    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5),

    torch.nn.Flatten(),

    torch.nn.Linear(in_features=128*6*6,out_features=2048),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.5),

    torch.nn.Linear(in_features=2048,out_features=1000),
    torch.nn.Softmax()
)
model=summary(model_AlexNet,(3,227,227))
