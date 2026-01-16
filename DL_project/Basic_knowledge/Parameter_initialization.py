"""
    参数初始化模块

参数初始化的目的
    1. 防止梯度消失或梯度爆炸
    2. 加快收敛速度
    3. 打破对称性

常用的参数初始化方法
    无法打破对称性的方法
        1. 全零初始化
        2. 全一初始化
        3. 固定值初始化
    能打破对称性的方法
        1. 随机初始化(均匀分布)
        2. 正态分布初始化
        3. kaiming初始化
        4. xavier初始化
关于初始化的选择上
    1. 对于浅层网络，可以选择简单的初始化方法，如随机初始化或正态分布初始化。
    2. 对于深层网络
        kaiming初始化 ：适用于ReLU及其变体激活函数的网络。
        xavier初始化 ：适用于Sigmoid和Tanh激活函数的网络。


"""


import torch.nn as nn
import torch


def dmo():
    linear = nn.Linear(5, 3)
    # 1. 随机初始化(均匀分布)
    # nn.init.uniform_(linear.weight)

    # 2. 正态分布初始化
    # nn.init.normal_(linear.weight, mean=0.0, std=1)

    # 3. kaiming正态分布初始化
    # nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')

    # 4. kaiming均匀分布初始化
    # nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')

    # 5. xavier正态分布初始化
    # nn.init.xavier_normal_(linear.weight)

    # 6. xavier均匀分布初始化
    # nn.init.xavier_uniform_(linear.weight)

    # 7. 全零初始化
    # nn.init.zeros_(linear.weight)

    # 8. 全一初始化
    # nn.init.ones_(linear.weight)

    # 9. 固定值初始化
    nn.init.constant_(linear.weight, 0.5)

    print(linear.weight)
    print(linear.bias)


if __name__ == "__main__":
    dmo()





