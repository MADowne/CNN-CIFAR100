# 导入模块
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import random
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2), 
                             nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                             nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                             nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.Flatten(), nn.Linear(256*5*5, 4096), nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(4096, 4096), nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(4096, 100))
    
  def forward(self, X):
    return self.net(X)

transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), # 此为训练集上的均值与方差
])
test_images = datasets.CIFAR100('./data/', train=False, download=True, transform=transform)
test_data = DataLoader(test_images, batch_size=512, shuffle=True)

temp_loss, temp_correct = 0, 0
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = Model().to(device)
net = torch.load('./model-alexnet')
net.eval()
with torch.no_grad():
    for X, y in test_data:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        loss = criterion(y_hat, y)

        label_hat = torch.argmax(y_hat, dim=1)
        temp_correct += (label_hat == y).sum()
        temp_loss += loss

print(f'test loss:{temp_loss/len(test_data):.3f}, test acc:{temp_correct/10000*100:.2f}%')

