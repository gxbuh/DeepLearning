"""
    CNN:图像分类
"""
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8 # 批次大小

def create_dataset():
    # 下载CIFAR10数据集  参数1:存放路径 参数2:是否为训练集 参数3:转换为tensor 参数4:是否下载
    train_data = CIFAR10(root='data/', train=True, transform=ToTensor(), download=True)
    test_data = CIFAR10(root='data/', train=False, transform=ToTensor(), download=True)

    return train_data, test_data


class ImageModel_CNN(nn.Module):
    def __init__(self):
        super(ImageModel_CNN, self).__init__() # 调用父类的构造函数
        # 卷积层(加权求和) + 激励层(激活函数) + 池化层(降采样)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(6 * 6 * 16, 120) # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 输出10类

        # 优化 dropout
        self.dropout = nn.Dropout(p=0.5) # 随机失活50%的神经元

    def forward(self, x):
        x = self.conv1(x) # 相当于 x = self.pool(self.relu(self.conv(x)))
        x = self.conv2(x)
        # 全连接层需要输入二维张量 (batch_size, features)
        # [8, 16, 6, 6] -> [8, 16*6*6] = [8, 576] ->8个样本，每个样本576个特征
        x = x.view(x.size(0), -1) # 参数1: 样本数(行数) 参数2: 列数(特征数) -1表示自动计算

        # 全连接层 + 激励层
        x = torch.relu(self.fc1(x))

        x = self.dropout(x) # dropout层
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) # dropout层

        x = self.fc3(x) # 最后一层不需要激活函数，交给损失函数处理CrossEntropyLoss

        return x


def train(train_data):
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel_CNN().to(device)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30

    start = time.time()
    for epoch in range(num_epochs):
        model.train() # 训练模式
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            running_loss += loss.item()
            if i % 1000 == 999: # 每1000个小批量输出一次loss
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0
        end_time = time.time()
        print(f'Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds')

    end = time.time()
    print(f'Training finished in {end - start:.2f} seconds')
    torch.save(model.state_dict(), 'model/image_cnn_model.pth')

def evaluate(test_data):
    dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel_CNN().to(device)
    model.load_state_dict(torch.load('model/image_cnn_model.pth'))
    model.eval() # 评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 评估时不需要计算梯度
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 由于训练中使用了CrossEntropyLoss损失函数，搭建神经网络时最后一层没有使用softmax激活函数, 要使用torch.max获取预测结果
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')


if __name__ == '__main__':


    # train_data, test_data = create_dataset()
    # print("训练集:", train_data.data.shape) # (50000, 32, 32, 3)
    # print("测试集:", test_data.data.shape)   # (10000, 32, 32, 3)
    # # 类别名称: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # print(f'类别数量: {len(train_data.classes)} 类别名称: {train_data.classes}')

    # plt.imshow(train_data.data[0]) # 显示第一张图片
    # plt.title(f'类别: {train_data.classes[train_data.targets[0]]}')
    # plt.show()
    # model = ImageModel_CNN().to(device)
    # summary(model, (3, 32, 32), batch_size=BATCH_SIZE, device=str(device)) # 输入图片尺寸为 (3, 32, 32)

    train_data, test_data = create_dataset()

    train(train_data)

    evaluate(test_data)