import torch
import torch.nn as nn
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# todo 词嵌入
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len) -> [2, 4]
        # embed shape: (batch_size, seq_len, embedding_size) -> [2, 4, 512]
        embed = self.embedding(x)
        # 1.符合标准正态分布  2.增强embedding的影响
        return embed * math.sqrt(self.embedding_size)

# todo 位置编码器
# 加入位置编码器，将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中, 以弥补位置信息的缺失.
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout_rate, max_len=5000):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout_rate)

        # 位置编码
        pe = torch.zeros(max_len, embedding_size)
        # 位置列矩阵 eg:[max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 变化矩阵 eg: [1, 256]
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))

        # [max_len, 1] @ [1, 256] = [max_len, 256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [max_len, 256] -> [1, max_len, 256]

        # 把pe位置编码矩阵 注册成模型的持久缓冲区buffer; 模型保存再加载时，可以根模型参数一样，一同被加载
        # 什么是buffer: 对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不参与模型训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_size] -> [2, 4, 512]
        # pe shape: [1, max_len, embedding_size] -> [1, 5000, 512]
        x = x + self.pe[:, :x.size(1)] # [2, 4, 512] + [1, 4, 512] = [2, 4, 512]
        return self.dropout(x)


if __name__ == '__main__':
    embedding = Embeddings(10, 512)
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    embed_result = embedding(x)

    position_encode = PositionalEncoding(512, 0.1)
    position_result = position_encode(embed_result)
    print(position_result.shape)