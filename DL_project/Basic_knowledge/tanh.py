import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建画布和坐标轴
_, axes = plt.subplots(1, 2)

# 函数图像
x = torch.linspace(-15, 15, 1000)
# 输入值x通过tanh函数转换成激活值y
y = torch.tanh(x)
axes[0].plot(x, y, color='darkblue')
axes[0].grid()
axes[0].set_title('tanh 函数图像')


# 导数图像
x = torch.linspace(-15, 15, 1000, requires_grad=True)
y = torch.tanh(x)
y.sum().backward()
# x.detach():输入值x的数值
# x.grad:计算梯度，求导
axes[1].plot(x.detach(), x.grad, color='darkblue')
axes[1].grid()
axes[1].set_title('tanh 导数图像')
plt.show()

