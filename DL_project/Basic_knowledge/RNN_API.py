import torch
import torch.nn as nn


# 参数1: 词向量的维度 参数2: 隐藏状态向量维度 参数3: 隐藏层数量
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)

# 参1: 每个句子的词的个数(长度) 参2: 句子的数量(批量大小) 参3: 词向量的维度
x = torch.randn(size=(5, 32, 128))

# 参1: 隐藏层的层数 = 隐藏层层数num_layers  参2: 句子的数量句子的数量(批量大小) 参3: 隐藏状态向量维度(hidden_size)
h0 = torch.randn(size=(1, 32, 256))


# 参1: x, 本次的输入  参2: h0, 上一次的隐藏状态
output, h1 = rnn(x, h0)
print(f'output: {output.shape}')    # [5, 32, 256] 每个时间步的输出，包含了所有时间步的隐藏状态
print(f'h1: {h1.shape}')           # [1, 32, 256] 最后1个时间步的隐藏状态

