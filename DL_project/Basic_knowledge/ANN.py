import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary
from sklearn.preprocessing import StandardScaler


"""
    调优思路
    1. 增加隐藏层数量 和 每层神经元数量
    2. SGD -> Adam 优化器
    3. 降低学习率
    4. 增加训练轮数
    5. 标准化输入数据
"""


# 创建数据集
def create_dataset():
    data = pd.read_csv('data/phone_prices.csv')
    # print(data.head()) # 查看前五行数据

    # 特征和标签 [:, :-1]表示所有行，除了最后一列，-1表示最后一列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype(np.float32) # 转换为浮点型

    # 参数1：测试集比例，参数2：随机种子，参数3：分层抽样(参考标签y的分布)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88, stratify=y)

    # 优化点：标准化输入数据
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# 构建分类网络模型
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhonePriceModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128) # 输入层到隐藏层
        self.linear2 = nn.Linear(128, 512) # 隐藏层到隐藏层
        self.linear3 = nn.Linear(512, 128) # 优化点：增加隐藏层
        self.output = nn.Linear(128, output_dim) # 隐藏层到输出层
        self.relu = nn.ReLU()  # 定义 ReLU 激活函数

    def forward(self, x):
        x = self.relu(self.linear1(x)) # 加权求和 + 激活函数
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        # x = self.relu(self.linear4(x))
        # 正常写法 ： x = torch.softmax(se.lf.output(x), dim=-1)
        # 但后续会使用 CrossEntropyLoss 损失函数，它内部集成了 softmax 操作
        x = self.output(x)
        return x

def train(train_dataset, input_dim, output_dim):
    """
    步骤：
        1. 创建数据加载器
        2. 实例化模型
        3. 定义损失函数
        4. 定义优化器
    """

    # 参数1 ：数据集，参数2：批次大小，参数3：是否打乱数据(训练集一般打乱，测试集一般不打乱)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam优化器

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train() # 训练模式
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in train_loader:
            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f}s')

    # 保存模型
    # 参数·1：模型参数（权重矩阵，偏置向量），参数2：保存路径
    torch.save(model.state_dict(), 'model/phone_price_model.pth')


def evaluate(test_dataset, input_dim, output_dim):
    """
    步骤：
        1. 创建数据加载器
        2. 实例化模型
        3. 加载模型参数
        4. 评估模型
    """
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = PhonePriceModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load('model/phone_price_model.pth'))
    model.eval() # 评估模式

    correct = 0
    total = 0
    with torch.no_grad(): # 评估时不需要计算梯度
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果, dim=1表示按行取最大值,概率最大的类别即为预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    # model = PhonePriceModel(input_dim, output_dim).to(device)
    # summary(model, input_size=(input_dim,), batch_size=16, device=str(device))

    # 训练模型
    train(train_dataset, input_dim, output_dim)

    # 评估模型
    evaluate(test_dataset, input_dim, output_dim)
