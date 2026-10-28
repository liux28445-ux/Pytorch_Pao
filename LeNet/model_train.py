import copy
import time

from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
from model import LeNet

import torch
import torch.nn as nn
import pandas as pd

def train_val_data_process():
    train_data = FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor()
        ]),
        download=True
    )
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=8)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []

    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch{epoch}/{num_epochs - 1}")
        print('-' * 30)

        # 初始化参数值
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        # 验证集和训练集样本数量
        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 模型打开训练模式
            model.train()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)

            train_corrects += torch.sum(pre_lab == b_y)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)

            val_corrects += torch.sum(pre_lab == b_y)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        train_acc_all.append(train_corrects.dounle().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)


        print(f'{epoch} Train Loss:{train_loss_all[-1]} Train acc{train_acc_all[-1]}')
        print(f'{epoch} Val Loss:{val_loss_all[-1]} Val acc{val_acc_all[-1]}')

        if val_acc_all[-1] > best_acc:
            # 保存当前最好准确度
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练耗时
        time_use = time.time() - since

        print(f'训练和验证耗时：{time_use//60}m,{time_use % 60}s')

        model.load_stats_dict(best_model_wts)
        torch.save(model.load_stats_dict(best_model_wts),'./best_model.pth')

        train_process = pd.DataFrame(
            data={
                "epoch":range(num_epochs),
                "train_loss_all":train_loss_all,
                "val_loss_all":val_loss_all,
                "train_acc_all":train_acc_all
                "val_acc_all":val_acc_all
            }
        )
    return train_process

