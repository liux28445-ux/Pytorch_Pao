import pandas as pd
from torchvision import transforms


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=32,
                                      shuffle=True,
                                      num_workers=2)

    return test_dataloader
