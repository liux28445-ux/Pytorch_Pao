
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 1. 加载数据集
train_data = FashionMNIST(
    root="./data",
    train=True,
    transform=transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor()
    ]),
    download=True
)

# 2. 创建数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

# 3. 取出一个批次的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    class_label = train_data.classes

    print("原始张量形状：", b_x.shape)
    print("处理后的形状：", batch_x.shape)
    print("标签形状：", batch_y.shape)

    # 4. 创建画布，8行8列
    fig, axes = plt.subplots(8, 8, figsize=(14, 14))

    # 5. 循环展示64张图片
    for i in range(64):
        ax = axes[i // 8, i % 8]

        # 显示第i张图片
        ax.imshow(batch_x[i], cmap="gray")

        # 显示图片对应的类别
        ax.set_title(class_label[batch_y[i]], fontsize=9)

        # 隐藏坐标轴
        ax.axis("off")

    plt.tight_layout()
    plt.show()
