import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from sklearn.model_selection import train_test_split



#シード値を固定
def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

fix_seed(seed=42)



#データセットの用意
trainval_data = datasets.CIFAR10('./data/cifar10', train=True, transform=transforms.ToTensor(), download=True)
train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data)-10000, 10000])

val_size = 3000
train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data)-val_size, val_size])



#前処理のクラス
class gcn():
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean)/(std + 10**(-6))  # 0除算を防ぐ


class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):  # 計算が重いのでGPUを用いる
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):  # 変換行列と平均をデータから計算
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # 各データについての平均を取る
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        self.E, self.V = torch.linalg.eigh(con_matrix)  # 固有値分解
        self.E = torch.max(self.E, torch.zeros_like(self.E)) # 誤差の影響で負になるのを防ぐ
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.E.squeeze()+self.epsilon)**(-0.5))), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x

GCN = gcn()
zca = ZCAWhitening()
zca.fit(trainval_data)



# 前処理を定義
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), 
                                      transforms.RandomRotation(degrees=(0, 20)),
                                      transforms.ToTensor(), 
                                      zca])

transform = transforms.Compose([transforms.ToTensor(), 
                                zca])



# データセットに前処理を設定
trainval_data.transform = transform_train

batch_size = 64

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)



#ニューラルネットを定義
rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Module):
    def __init__(self, dropout_ratio=0.2):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = torch.rand(*x.size()) > self.dropout_ratio
            return x * self.mask.to(x.device)
        else:
            return x * (1.0 - self.dropout_ratio)
        

class GlobalAvgpool2d(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
    
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class SENet_Attention(nn.Module):
    def __init__(self, input_dim, sq_ratio):
        super().__init__()
        self.input_dim = input_dim
        self.sq_dim = int(input_dim / sq_ratio)
        self.g_ave_pool = GlobalAvgpool2d()
        self.linear1 = nn.Linear(self.input_dim, self.sq_dim)
        self.linear2 = nn.Linear(self.sq_dim, self.input_dim)
        self.activ1 = nn.ReLU()
        self.activ2 = nn.Sigmoid()

    def forward(self, x):
        h = self.g_ave_pool(x).squeeze()
        h = self.linear1(h)
        h = self.activ1(h)
        h = self.linear2(h)
        h = self.activ2(h).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x_sq = x * h
        return x_sq
    

class Block_ResNet50(nn.Module):
    def __init__(self, input_dim, output_dim, stride1=(1, 1)):
        super().__init__()
        self.hid_dim = int(output_dim / 4)
        self.conv1 = nn.Conv2d(input_dim, self.hid_dim, 1, stride=stride1)
        self.conv2 = nn.Conv2d(self.hid_dim, self.hid_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hid_dim, output_dim, 1)
        self.activ1 = nn.ReLU()
        self.activ2 = nn.ReLU()
        self.activ3 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.hid_dim)
        self.norm2 = nn.BatchNorm2d(self.hid_dim)
        self.norm3 = nn.BatchNorm2d(output_dim)
        self.SENet = SENet_Attention(output_dim, 8)
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 1, stride=stride1),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activ1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activ2(h)
        h = self.conv3(h)
        h = self.norm3(h)
        h = self.SENet(h)
        x_short = self.shortcut(x)
        h = h + x_short
        h = self.activ3(h)
        return h


class Block_ResNet34(nn.Module):
    def __init__(self, input_dim, output_dim, stride1=(1, 1)):
        super().__init__()
        self.hid_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, self.hid_dim, 3, stride=stride1, padding=1)
        self.conv2 = nn.Conv2d(self.hid_dim, output_dim, 3, padding=1)
        self.activ1 = nn.ReLU()
        self.activ2 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.hid_dim)
        self.norm2 = nn.BatchNorm2d(output_dim)
        self.SENet = SENet_Attention(output_dim, 8)
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 1, stride=stride1),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activ1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.SENet(h)
        x_short = self.shortcut(x)
        h = h + x_short
        h = self.activ2(h)
        return h




class Layer(nn.Module):
    def __init__(self, block, input_dim, output_dim, dence, stride1=None):
        super().__init__()
        self.layers = []
        self.dence = dence
        if stride1 is not None:
            self.layers.append(block(input_dim, output_dim, stride1=stride1))
            self.dence -= 1
        else:
            self.layers.append(block(input_dim, output_dim))
            self.dence -= 1
        for _ in range(self.dence):
            self.layers.append(block(output_dim, output_dim))
        self.layer = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layer(x)


conv_net = nn.Sequential(
    nn.Conv2d(3, 64, 7, padding=(3, 3)),                    # 32x32x3 -> 32x32x64
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(3, stride=(2, 2), padding=1),              # 32x32x64 -> 16x16x64
    Layer(Block_ResNet34, 64, 64, 3),                     # 16x16x64 -> 16x16x64
    Layer(Block_ResNet34, 64, 128, 4, stride1=(2, 2)),    # 16x16x64 -> 8x8x128
    Layer(Block_ResNet34, 128, 256, 6, stride1=(2, 2)),  # 8x8x128 -> 4x4x256
    Layer(Block_ResNet34, 256, 512, 3, stride1=(2, 2)),  # 4x4x256 -> 2x2x512
    GlobalAvgpool2d(),                                      # 2x2x512 -> 1x1x512
    nn.Flatten(),
    nn.Linear(1*1*512, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10)
)



#初期化
def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)


conv_net.apply(init_weights)



#学習の設定
n_epochs = 100
lr = 0.005
device = 'cuda'

conv_net.to(device)
optimizer = optim.Adam(conv_net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()



#学習
for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []

    conv_net.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        n_train += t.size()[0]
        

        conv_net.zero_grad()

        x = x.to(device)
        t = t.to(device)

        y = conv_net.forward(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()

        pred = y.argmax(1)

        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())

    conv_net.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        n_val += t.size()[0]

        x = x.to(device)
        t = t.to(device)

        y = conv_net.forward(x)

        loss = loss_function(y, t)

        pred = y.argmax(1)
        
        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())

    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
        epoch,
        np.mean(losses_train),
        acc_train/n_train,
        np.mean(losses_valid),
        acc_val/n_val
    ))