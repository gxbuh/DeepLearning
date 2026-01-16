"""

损失函数介绍
    概述：
    在机器学习和深度学习中，损失函数（Loss Function）用于衡量模型预测值与真实值之间的差异。它是训练过程中优化的目标，通过最小化损失函数来提升模型的性能。
    也叫“代价函数”（Cost Function）或“目标函数”（Objective Function）或“误差函数”（Error Function）。

    分类：
        分类问题：
            多分类交叉熵损失函数（Categorical Cross-Entropy Loss） CrossEntropyLoss
            二分类交叉熵损失函数（Binary Cross-Entropy Loss） BCELoss
        回归问题：
            均方误差损失函数（Mean Squared Error Loss） MSELoss    MSE
            平均绝对误差损失函数（Mean Absolute Error Loss） L1Loss  MAE
            smooth L1损失函数（Smooth L1 Loss） SmoothL1Loss

"""
import torch

def test01():
    # 多分类交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()  # reduction='mean' 默认值

    # 假设有3个样本，类别数为4 ->  即公式中的f(x)
    outputs = torch.tensor([[2.0, 1.0, 0.1, 0.5],
                            [0.5, 2.5, 0.3, 0.2],
                            [1.0, 0.2, 0.3, 3.0]])  # 模型的输出，也即预测值
    # 即公式中的 y
    targets = torch.tensor([0, 1, 3])  # 真实标签
    # tagets = torch.tensor([[1, 0, 0, 0],
    #                         [0, 1, 0, 0],
    #                         [0, 0, 0, 1]])  # 真实标签  one-hot编码形式

    loss = criterion(outputs, targets) # 参数1: 模型输出(预测值) 参数2: 真实标签
    print(f'CrossEntropyLoss: {loss.item()}')

def test02():
    # 二分类交叉熵损失函数
    criterion = torch.nn.BCELoss()  # reduction='mean' 默认值
    # 假设有3个样本
    outputs = torch.tensor([0.9, 0.2, 0.8], dtype=torch.float)  # 模型的输出，也即预测值 (0~1之间)
    targets = torch.tensor([0, 1, 0], dtype=torch.float)  # 真实标签 (0或1)

    loss = criterion(outputs, targets) # 参数1: 模型输出(预测值) 参数2: 真实标签
    print(f'BCELoss: {loss.item()}')

def test03():
    # 平均绝对误差损失函数
    criterion = torch.nn.L1Loss()  # reduction='mean' 默认值
    # criterion = torch.nn.MSELoss()  # 均方误差损失函数
    # criterion = torch.nn.SmoothL1Loss()  # smooth L1损失函数

    # 假设有3个样本
    outputs = torch.tensor([2.5, 0.0, 2.0], dtype=torch.float)  # 模型的输出，也即预测值
    targets = torch.tensor([3.0, -0.5, 2.0], dtype=torch.float, requires_grad=True)  # 真实标签

    loss = criterion(outputs, targets) # 参数1: 模型输出(预测值) 参数2: 真实标签
    print(f'L1Loss: {loss.item()}')


if __name__ == "__main__":
    test03()