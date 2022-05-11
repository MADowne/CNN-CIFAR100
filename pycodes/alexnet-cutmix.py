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

writer = SummaryWriter('alexnet-cutmix')


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2



# 下载以及转换cifar100
transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # 此为训练集上的均值与方差
                                ])
train_images = datasets.CIFAR100('./data/', train=True, download=True, transform=transform)
test_images = datasets.CIFAR100('./data/', train=False, download=True, transform=transform)

# batch size设置为512
train_data = DataLoader(train_images, batch_size=512, shuffle=True)
test_data = DataLoader(test_images, batch_size=512, shuffle=True)

# Alexnet
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

# 参数初始化
def initial(layer):
  if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
    nn.init.xavier_normal_(layer.weight.data)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = Model().to(device)
net.apply(initial)


lr = 0.01	# 学习率
beta = 1
cutmix_prob= 0.8
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

# 训练与测试
train_loss, test_loss, train_acc, test_acc = [], [], [], [] # 用来记录每个epoch的训练、测试误差以及准确率
test_acc1 = 0
i = 0
while 1:  # 训练
  
  net.train()
  temp_loss, temp_correct = 0, 0
  for X, y in train_data:
    input = X.to(device)
    target = y.to(device)
    r = np.random.rand(1)

    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        output = net(input)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    else:
        # compute output
        output = net(input)
        loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算每次loss与预测正确的个数
    label_hat = torch.argmax(output, dim=1)
    temp_correct += (label_hat == target).sum()
    temp_loss += loss

  i+=1
  print(f'epoch:{i}  train loss:{temp_loss/len(train_data):.3f}, train Aacc:{temp_correct/50000*100:.2f}%', end='\t')
  train_loss.append((temp_loss/len(train_data)).item())
  train_acc.append((temp_correct/50000).item())
  writer.add_scalar('Train/Loss', temp_loss/len(train_data) ,i)
  writer.add_scalar('Train/Acc',temp_correct/50000,i)


  # 测试
  temp_loss, temp_correct = 0, 0
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
    test_loss.append((temp_loss/len(test_data)).item())
    test_acc.append((temp_correct/len(test_data)).item())
    writer.add_scalar('Test/Loss', temp_loss/len(test_data),i)
    writer.add_scalar('Test/Acc',temp_correct/10000,i)
    if temp_correct/10000 < test_acc1:
      break
    test_acc1 = temp_correct/10000

torch.save(net, './model-alexnet-cutmix')

