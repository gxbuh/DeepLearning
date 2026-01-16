"""
    自动微分模块
    多远线性公式
        y = wx + b
        w的行就对应样本特征的个数 列代表神经元个数
    权重更新公式
        w(新) = w(旧) - lr(学习率) * dw (权重的梯度,即损失函数对权重的偏导数)
        b(新) = b(旧) - lr(学习率) * db (偏置的梯度,即损失函数对偏置的偏导数)
        实际应用中,可以不考虑偏置b的更新,会将b视为常数或全0
    反向传播
        计算损失函数对各个参数的梯度
    Autograd 自动微分
        PyTorch 中的自动微分模块
        通过跟踪张量上的操作来自动计算梯度
        主要类: torch.autograd.Tensor
    计算图
        由张量和它们之间的操作组成的有向无环图(DAG)
        节点表示张量,边表示操作
        前向传播: 构建计算图并计算输出
        反向传播: 通过计算图计算梯度
    关键属性
        requires_grad: 指示是否需要计算梯度
        .grad: 存储张量的梯度
    关键函数
        backward(): 触发反向传播计算梯度
        no_grad(): 上下文管理器,在其作用域内禁止梯度计算

    detach(): 解决张量不能直接转换为 NumPy 数组的问题
        从计算图中分离张量,返回一个新的张量,共享同一内存,不需要梯度计算requires_grad=False，用于在不影响计算图的情况下进行张量操作
"""

"""
import torch

# 参数1: 初始值, 参数2: 是否需要计算梯度(自动微分), 参数3: 数据类型
w = torch.tensor(10.0, requires_grad=True, dtype=torch.float)  # 权重

loss = w ** 2 + 20 # 损失函数 = w²+20 -> 求导： 2w

# 反向传播计算梯度，梯度 = dloss/dw = 2w，即导数，计算完毕后会记录到 w.grad 中
loss.sum().backward() # 反向传播计算梯度 .sum()保证loss是标量

lr = 0.01  # 学习率

for i in range(100):
    #正向传播计算损失
    loss = w ** 2 + 20

    # 清零梯度
    if w.grad is not None: # 第一次迭代时 w.grad 为 None
        w.grad.zero_() # 每次迭代前清零梯度，否则梯度会累加

    loss.sum().backward()  # 反向传播计算梯度

    w.data -= lr * w.grad.data # 梯度更新, 使用 .data 访问张量数据，避免跟踪梯度

    print(f'Iteration {i+1}: w = {w.item():.4f}, loss = {loss.item():.4f}')
    # 第一次结果: w = 9.8000, loss = 120.0000
    # 第二次结果: w = 9.6040, loss = 116.0000
    # ...

"""

import torch

x = torch.ones(2, 5)  # 表示：特征(输入张量),2个样本,5个特征
y = torch.zeros(2, 3) # 表示：标签(目标张量,真实值),2个样本,3个类别
w = torch.randn(5, 3, requires_grad=True)  # 权重,5个输入特征,3个神经元(类别)
b = torch.randn(3, requires_grad=True)     # 偏置,3个神经元(类别)

z = torch.matmul(x, w) + b # z - x @ w + b, 前向传播计算输出预测值

criterion = torch.nn.MSELoss() # 均方误差损失函数 nn: neural network 神经网络
loss = criterion(z, y) # 计算损失

loss.backward()

print(f'Gradient w.r.t w:\n{w.grad}') # 打印权重的梯度
print(f'Gradient w.r.t b:\n{b.grad}') # 打印偏置