import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from input_demo import *
from encoder_demo import *

# todo 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size # 词嵌入维度
        self.self_attn = self_attn # 自注意力机制 Q = K = V
        self.src_attn = src_attn # 一般注意力机制 Q != K == V
        self.feed_forward = feed_forward # 前馈全连接层
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory # 源句子
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return self.dropout(x)


# todo 解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(features=layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


def test_decoder():
    # 测试解码器
    embedding = Embeddings(1000, 512)
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    embed_result = embedding(x)
    pe = PositionalEncoding(embedding_size=512, dropout_rate=0.1)
    x = pe(embed_result)

    decoder_layer = DecoderLayer(size=512, self_attn=MultiHeadAttention(head=8, embedding_size=512),
                                 src_attn=MultiHeadAttention(head=8, embedding_size=512),
                                 feed_forward=FeedForward(512, 64), dropout=0.1)

    decoder = Decoder(layer=decoder_layer, N=6)
    target_mask = source_mask = torch.zeros(8, 4, 4)

    memory = test_encoder()

    decoder_output = decoder(x, memory, source_mask, target_mask)
    # print(f'解码器输出的维度为：{decoder_output.shape}')
    # print(decoder_output)
    return decoder_output


if __name__ == '__main__':
    test_decoder()

