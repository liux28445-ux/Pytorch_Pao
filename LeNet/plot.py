from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0
                               )

for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0 :
        break
    batch_x = b_x.squeeze().numpy()   # 将四维张量移除第一维，并转换成Numpy数组
    batch_y = b_y.numpy()             # 将张量转换成Numpy数组
    class_label = train_data.classes  # 训练集标签
    print(class_label)
