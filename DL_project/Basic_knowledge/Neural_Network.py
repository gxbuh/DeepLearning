"""
神经网络基础代码示例
"""

import torch
import torch.nn as nn
from torchsummary import summary

class Model(nn.Module): # 自定义类继承 nn.Module
    def __init__(self):
        super(Model, self).__init__() # 调用父类的构造函数

        self.linear1 = nn.Linear(3, 3) # 隐藏层
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2) # 输出层

        # 对隐藏层进行参数初始化 一般不需要手动初始化，PyTorch会自动初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # 分解版
        # x = self.linear1(x)  # 线性变换 加权求和
        # x = torch.sigmoid(x) # 激活函数

        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.output(x), dim=-1) # dim=-1 按行计算，一条一条样本的处理

        return x


# python
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)  # 把模型移动到同一设备
    data = torch.randn(size=(5, 3)).to(device)  # 列是3个特征，行可以是任意多个样本

    print("输入设备：", data.device, "模型参数设备：", next(model.parameters()).device)
    output = model(data)
    print("输出数据：", output)  # 自动调用 model.forward(data)
    print(f'output shape: {output.shape}')

    print("================计算模型参数和结构================")
    # summary 的 input_size 不包含 batch 维，传入 device 参数
    summary(model, input_size=(3,), device=str(device))

    for name, param in model.named_parameters():
        print(f'Layer: {name} | Device: {param.device} | Values : {param} \n')

if __name__ == '__main__':
    train()
