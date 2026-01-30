from generator_demo import *


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder # 编码器
        self.decoder = decoder # 解码器
        self.source_embed = source_embed # 源句子嵌入
        self.target_embed = target_embed # 目标句子嵌入
        self.generator = generator # 生成器

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def test_transformer():
    """
    完整的 Transformer 模型测试函数
    """
    # 1. 创建编码器
    from encoder_demo import Encoder, EncoderLayer, MultiHeadAttention, FeedForward, LayerNorm
    from input_demo import Embeddings, PositionalEncoding

    # 创建多头注意力和前馈网络
    multi_head_attn = MultiHeadAttention(head=8, embedding_size=512)
    feed_forward = FeedForward(d_model=512, d_ff=2048)

    # 创建编码器层
    encoder_layer = EncoderLayer(
        size=512,
        self_attn=multi_head_attn,
        feed_forward=feed_forward,
        dropout=0.1
    )

    # 创建编码器（6层）
    encoder = Encoder(layer=encoder_layer, N=6)

    # 2. 创建解码器
    from decoder_demo import Decoder, DecoderLayer  # 假设有这些模块

    # 创建解码器层（需要自注意力、源注意力和前馈网络）
    self_attn = MultiHeadAttention(head=8, embedding_size=512)
    src_attn = MultiHeadAttention(head=8, embedding_size=512)
    feed_forward_dec = FeedForward(d_model=512, d_ff=2048)

    decoder_layer = DecoderLayer(
        size=512,
        self_attn=self_attn,
        src_attn=src_attn,
        feed_forward=feed_forward_dec,
        dropout=0.1
    )

    decoder = Decoder(layer=decoder_layer, N=6)

    # 3. 创建嵌入层
    src_embed = nn.Sequential(
        Embeddings(vocab_size=1000, embedding_size=512),
        PositionalEncoding(embedding_size=512, dropout_rate=0.1)
    )

    tgt_embed = nn.Sequential(
        Embeddings(vocab_size=1000, embedding_size=512),
        PositionalEncoding(embedding_size=512, dropout_rate=0.1)
    )

    # 4. 创建生成器
    generator = Generator(d_model=512, vocab=1000)

    # 5. 创建完整的 EncoderDecoder 模型
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        source_embed=src_embed,
        target_embed=tgt_embed,
        generator=generator
    )

    # 6. 准备测试数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    # 源序列和目标序列（词汇表索引）
    src = torch.randint(low=1, high=1000, size=(batch_size, src_seq_len))
    tgt = torch.randint(low=1, high=1000, size=(batch_size, tgt_seq_len))

    # 创建掩码
    src_mask = torch.ones(batch_size, 1, src_seq_len).type(torch.uint8)  # 源序列掩码
    tgt_mask = sub_mask(tgt_seq_len)  # 目标序列后续掩码

    # 7. 执行前向传播
    print("开始测试 Transformer 模型...")
    print(f"输入源序列形状: {src.shape}")
    print(f"输入目标序列形状: {tgt.shape}")
    print(f"源掩码形状: {src_mask.shape}")
    print(f"目标掩码形状: {tgt_mask.shape}")

    try:
        output = model(src, tgt, src_mask, tgt_mask)
        print(f"输出形状: {output.shape}")
        print("Transformer 模型测试成功!")
        return model, output
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        return None, None


# 运行测试
if __name__ == "__main__":
    print("=" * 50)
    print("运行完整版 Transformer 测试:")
    model_full, output_full = test_transformer()





