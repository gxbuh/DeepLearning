import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from input_demo import *

# todo 生成下三角矩阵
def sub_mask(size):
    attn_shape = torch.ones((1, size, size), dtype=torch.long)
    mask = torch.triu(attn_shape, diagonal=1).type(torch.uint8)
    return mask == 0

# todo 注意力
def attention(query, key, value, mask=None, dropout=None):
    # 自注意力机制 query, key, value 输入的维度都是 [batch_size, seq_len, embedding_size]]->[2, 4, 512]
    # 编码器端padding_mask 解码器端sentence_mask

    d_k = query.size()[-1] # 词嵌入维度：512
    # attention公式：score = query * key.T / sqrt(d_k)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None: # mask = 0 -> score = -1e9
        score = score.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(score, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 返回 attention(Q,K,V)输出和注意力权重
    return torch.matmul(attention_weights, value), attention_weights

'''
    要求：module 必须是一个“干净的、未参与计算图的 Module”
    clones() 只能复制「未 forward 的 layer」
'''
def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

# todo 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert embedding_size % head == 0 # 确保词嵌入维度能被头数整除

        self.head = head # 头数
        self.embedding_size = embedding_size
        self.d_k = embedding_size // head # 词嵌入维度
        self.linear_layers = clones(nn.Linear(embedding_size, embedding_size), n = 4) # 4 个线性层

        self.attention_weights = None # 注意力权重
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch_size, seq_len, embedding_size]]->[2, 4, 512]
        if mask is not None:
            mask = mask.unsqueeze(0) # [8, 4, 4] -> [1, 8, 4, 4]

        batch_size = query.size(0)

        # [batch_size, seq_len, embedding_size]]->[2, 4, 512]->[2, 4, 8, 64]->[2, 8, 4, 64]
        # 让句子长度4和句子特征64靠在一起 更有利捕捉句子特征
        # (Linear1, query), (Linear2, key), (Linear3, value)
        query, key, value = [l(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # myoutptlist_data = []
        # for model, x in zip(self.linears, (query, key, value)):
        #     print('x--->', x.shape) # [2,4,512]
        #     myoutput = model(x)
        #     print('myoutput--->',  myoutput.shape)  # [2,4,512]
        #     # [2,4,512] --> [2,4,8,64] --> [2,8,4,64]
        #     tmpmyoutput = myoutput.view(batch_size, -1,  self.head, self.d_k).transpose(1, 2)
        #     myoutptlist_data.append( tmpmyoutput )
        # mylen = len(myoutptlist_data)   # mylen:3
        # query = myoutptlist_data[0]     # [2,8,4,64]
        # key = myoutptlist_data[1]       # [2,8,4,64]
        # value = myoutptlist_data[2]     # [2,8,4,64]

        # attention_weights -> [batch_size, head, seq_len, seq_len] -> [2, 8, 4, 4]
        output, self.attention_weights = attention(query, key, value, mask, self.dropout)

        # [batch_size, head, seq_len, seq_len] -> [2, 4, 8, 64] -> [2, 4, 512]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_size)

        output = self.linear_layers[-1](output)
        return output

# todo 前馈全连接层
# 作用: * 考虑注意力机制可能对复杂过程的拟合程度不够, 通过增加两层网络来增强模型的能力.
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1): # d_model: 词嵌入维度，d_ff: 前馈全连接层内部特征维度
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 来自于 编码器的多头注意力层
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

# todo 规范化层
class LayerNorm(nn.Module): # 针对同一样本，不同特征进行规范化. 每个句子长度不一
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # y = a_2 * (x - mean) / (std + eps) + b_2
        # 相当于 y = k * x + b 中的 k , b
        self.a_2 = nn.Parameter(torch.ones(features)) # Parameter 可学习参数
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps # 避免除零

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 平均值
        std = x.std(-1, keepdim=True) # 标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # * 是位置相乘，不是矩阵相乘

# todo 子层连接结构
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(features=size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # x shape: [batch_size, seq_len, embedding_size] -> [2, 4, 512]
        # sublayer(x) shape: [batch_size, seq_len, embedding_size] -> [2, 4, 512]
        return x + self.dropout(sublayer(self.norm(x)))

# todo 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # 多头注意力层
        self.feed_forward = feed_forward # 前馈全连接层
        self.sublayer = clones(SubLayerConnection(size, dropout), 2) # 创建两个子层连接层
        self.size = size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return self.dropout(x)

# todo 编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        # layer: 1 个 编码器层
        # N: 编码器层数
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(features=layer.size) # 规范化层

    def forward(self, x, mask):
        # 数据经过 N 个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # 返回规范化层后的结果

def test_encoder():
    # print(sub_mask(5))
    # MultiHeadAttention(head=8, embedding_size=512)
    # print(MultiHeadAttention)

    # 测试 MultiHeadAttention
    query = torch.randn(2, 4, 512)
    key = torch.randn(2, 4, 512)
    value = torch.randn(2, 4, 512)
    mask = sub_mask(4)
    attention_output, attention_weights = attention(query, key, value, mask)
    # print(attention_output.shape)
    # print(attention_weights.shape)
    multi_head_attention = MultiHeadAttention(head=8, embedding_size=512)
    # multi_head_attention_output = multi_head_attention(query, key, value, mask)
    # print(f'多头注意力层输出的维度为：{multi_head_attention_output.shape}')

    # 测试 FeedForward
    feed_forward = FeedForward(512, 2048)
    # feed_forward_output = feed_forward(multi_head_attention_output)
    # print(f'前馈全连接层输出的维度为：{feed_forward_output.shape}')

    # 测试 LayerNorm
    # layer_norm = LayerNorm(512)
    # layer_norm_output = layer_norm(feed_forward_output)
    # print(f'规范化层输出的维度为：{layer_norm_output.shape}')

    # 测试 SubLayerConnection
    sub_layer_connection = SubLayerConnection(512, 0.1)
    # sub_layer = lambda x: x

    # 测试编码器层
    encoder_layer = EncoderLayer(512, self_attn=multi_head_attention, feed_forward=feed_forward, dropout=0.1)
    # encoder_layer_output = encoder_layer(layer_norm_output, mask)
    # print(f'编码器层输出的维度为：{encoder_layer_output.shape}')

    # 测试编码器
    '''
    clone 问题
    “ Encoder 里用 copy.deepcopy 复制了一个“已经参与过前向计算、内部缓存了 Tensor 的模块”。”
    注意要注释掉前面的
    '''

    embedding = Embeddings(1000, 512)
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    embed_result = embedding(x)
    pe = PositionalEncoding(embedding_size=512, dropout_rate=0.1)
    x = pe(embed_result)
    encoder = Encoder(layer=encoder_layer, N=6)
    encoder_output = encoder(x, mask)
    print(f'编码器输出的维度为：{encoder_output.shape}')
    return encoder_output

if __name__ == '__main__':
    test_encoder()