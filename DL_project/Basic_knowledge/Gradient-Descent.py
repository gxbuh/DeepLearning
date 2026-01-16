import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30
# 1. 实际平均温度
def test01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, temperature, color='darkred')
    plt.scatter(days, temperature, color='darkblue')
    plt.title('Temperature')
    plt.show()

# 2. 指数加权平均温度
def test02(beta):
    torch.manual_seed(0) # 固定随机数种子
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10 # 产生30天的随机温度
    exp_weight_avg = []
    for idx, temp in enumerate(temperature, 1):  # 从下标1开始
        # 第一个元素的的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
        new_temp = beta * exp_weight_avg[-1] + (1 - beta) * temp
        exp_weight_avg.append(new_temp)
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color='darkred')
    plt.scatter(days, temperature, color='darkblue')
    plt.title(f'Exponential Weighted Average Temperature (beta={beta})')
    plt.show()

def test03():
    w = torch.tensor([1.0], requires_grad=True)
    loss = ((w ** 2) / 2.0).sum()

    # momentum
    # optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)
    # Adagrad
    # optimizer = torch.optim.Adagrad([w], lr=0.01)
    # RMSprop
    # optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.9)
    # Adam
    optimizer = torch.optim.Adam([w], lr=0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'第一次梯度: {w.grad.item()}')
    print(f'更新后的权重: {w.item()}')

    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'第二次梯度: {w.grad.item()}')
    print(f'更新后的权重: {w.item()}')


if __name__ == '__main__':
    test01()
    test02(beta=0.5)
    test02(beta=0.9)
    # test03()