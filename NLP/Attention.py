import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim1, value_dim2, output_dim):
        super(Attention, self).__init__()
        self.query_dim = query_dim # Q张量最后一维的维度
        self.key_dim = key_dim
        self.value_dim1 = value_dim1
        self.value_dim2 = value_dim2 # V张量最后一维的维度
        self.output_dim = output_dim

        # 第一个全连接层，计算注意力权重
        # 输入维度为 Q + K, Q->[1, 1, 32], K->[1, 1, 32], 拼接后->[1, 1, 64]
        # 输出维度为 32
        self.attention_fc = nn.Linear(self.query_dim + self.key_dim, value_dim1)

        # 第二个全连接层，计算最终输出
        # 输入维度为 V + 注意力权重, V->[1, 1, 64], 注意力权重->[1, 1, 32], 拼接后->[1, 1, 96]
        # 输出维度为 output_dim
        self.output_fc = nn.Linear(self.value_dim2 + value_dim1, output_dim)

    def forward(self, Q, K, V):
        # Q: 查询张量，形状为 (batch_size, 1, query_dim=32)
        # K: 键张量，形状为 (batch_size, 1, key_dim=32)
        # V: 值张量，形状为 (batch_size, seq_len=32, value_dim=64)

        # 1. 计算注意力分数
        # 将 Q 和 K 在最后一个维度拼接，然后通过全连接层
        # Q: [batch_size, 1, 32], K: [batch_size, 1, 32]
        # torch.cat() 修正：需要将张量放在元组或列表中
        combined = torch.cat((Q, K), dim=-1)  # [batch_size, 1, 64]

        # 2. 通过全连接层得到注意力分数
        # 假设 attention_fc 输出维度是 32（序列长度）
        scores = self.attention_fc(combined)  # [batch_size, 1, 32]

        # 3. 应用 softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, 32]
        print(f"注意力权重形状: {attn_weights.shape}")

        # 4. 应用注意力权重到 V
        # attn_weights: [batch_size, 1, 32]
        # V: [batch_size, 32, 64]
        attn_applied = torch.bmm(attn_weights, V)  # [batch_size, 1, 64]

        # 5. 拼接 Q 和注意力结果
        # Q: [batch_size, 1, 32]
        # attn_applied: [batch_size, 1, 64]
        output = torch.cat((Q, attn_applied), dim=-1)  # [batch_size, 1, 96]

        # 6. 通过输出全连接层
        output = self.output_fc(output)  # [batch_size, 1, output_dim]

        return output


if __name__ == '__main__':
    query_dim = 32
    key_dim = 32
    value_dim1 = 32
    value_dim2 = 64
    output_dim = 32
    attention = Attention(query_dim, key_dim, value_dim1, value_dim2, output_dim)

    print(attention)

    Q = torch.randn(1, 1, query_dim)  # Query张量，形状为 (1, 1, 32)
    K = torch.randn(1, 1, key_dim)    # Key张量，形状为 (1, 1, 32)
    V = torch.randn(1, 32, value_dim2)  # Value张量，形状为 (1, 32, 64)

    output = attention(Q, K, V)
    print(output.shape)  # 输出张量的形状
    print(output)  # 输出张量的内容
