import torch
from torch.utils.data import TensorDataset # 张量数据集
from torch.utils.data import DataLoader # 数据加载器
from torch import nn ## 神经网络模块 有平方差损失函数MSELoss和假设函数Linear等
from torch import optim # 优化器模块 有SGD等优化器
from sklearn.datasets import make_regression # 生成回归数据集
import matplotlib.pyplot as plt # 绘图库

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

"""
    线性回归模拟
    numpy对象->tensor张量->TensorDataset数据集对象->DataLoader数据加载器
"""

def create_dataset():
    # 生成回归数据集 X: 特征变量, y: 目标变量, coef: 真实回归系数 均为ndarray类型
    X, y, coef = make_regression(
        n_samples=100,  # 样本数量
        n_features=1,   # 特征数量
        noise=10.0,     # 噪声水平,噪声越大,数据越分散,拟合难度越大,噪声越小，数据越集中
        coef=True,      # 是否返回真实的回归系数
        bias=14.5,      # 偏置项
        random_state=42 # 随机种子
    )

    # 转换为张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1) # 调整y的形状为列向量 -1: 自动计算该维度大小

    # 创建TensorDataset对象
    dataset = TensorDataset(X_tensor, y_tensor)

    return dataset

def train_model(dataset):
    # 创建数据加载器  参数1: 数据集对象, 参数2: 批量大小, 参数3: 打乱数据，提高模型泛化能力 (训练集打乱，测试集不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 定义线性回归模型  参数1: 输入特征数量, 参数2: 输出特征数量
    model = nn.Linear(in_features=1, out_features=1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss() # 均方误差损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01) # 随机梯度下降优化器, 参数1: 模型参数, 参数2: 学习率

    # 训练模型 训练轮数,损失列表,总损失,总样本数
    num_epochs, loss_list, total_loss, total_samples = 100, [], 0.0, 0

    for epoch in range(num_epochs): # 0,1,2,...,99
        # 每轮分批次训练，故从数据加载器中获取每个批次的数据  共7批(16个样本/批次,最后1批4个样本)
        for inputs, targets in dataloader: # inputs: x特征, targets: y目标值
            # 前向传播
            outputs = model(inputs) # 计算模型输出即y预测值
            loss = criterion(outputs, targets) # 计算损失

            # 反向传播和优化
            optimizer.zero_grad() # 清零梯度
            loss.backward()       # 反向传播计算梯度
            optimizer.step()      # 更新模型参数

            # 累加损失和样本数
            #  nn.MSELoss() 的 reduction='mean'，返回的是当前批次上每个样本的平均损失, 后面再算每轮的平均损失
            total_loss += loss.item() * inputs.size(0) # loss.item(): 获取标量损失值
            total_samples += inputs.size(0) # inputs.size(0) = 批次大小 or 当前批次样本数


        # 计算并记录每轮的平均损失
        avg_loss = total_loss / total_samples
        loss_list.append(avg_loss)

        # 重置总损失和样本数
        total_loss, total_samples = 0.0, 0

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 绘制损失曲线
    plt.plot(range(1, num_epochs + 1), loss_list, label='训练损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.title('线性回归模型训练损失曲线')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制预测值与真实值的对比图
    X_all = dataset.tensors[0]  # 获取所有特征数据
    y_all = dataset.tensors[1]  # 获取所有目标数据
    y_pred = model(X_all).detach()  # 模型预测值, detach()分离计算图
    plt.scatter(X_all.numpy(), y_all.numpy(), label='真实值', color='blue')
    plt.scatter(X_all.numpy(), y_pred.numpy(), label='预测值', color='red')

    plt.xlabel('特征 X')
    plt.ylabel('目标 y')
    plt.title('线性回归模型预测值与真实值对比')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    dataset = create_dataset()
    train_model(dataset)

