"""

    seq2seq模型实现英文到法文的翻译任务，包含编码器和解码器（带注意力机制）

"""

import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# todo 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大句子长度不能超过10个 (包含标点)
MAX_LENGTH = 10
# 数据文件路径
data_path = 'data/eng-fra-v2.txt'

# todo 字符串规范化函数
def normalizeString(s):
    s = s.lower().strip() # 小写并去掉首尾空格
    s = re.sub(r"([.!?])", r" \1", s) # 在标点前加空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # 非字母字符替换为空格
    return s

# 读取数据文件，构建词汇表
def get_data():
    """
    读取数据文件，构建词汇表并返回句子对列表
    :return: 英文词汇表, 英文索引表, 英文字典大小, 法文词汇表, 法文索引表, 法文字典大小, 句子对列表
    """

    print("正在读取数据...")
    # 读取数据文件并按行分割
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    en_fre_pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines] # 句子对列表
    # print(f"共读取 {len(en_fre_pairs)} 条句子对")
    # return en_fre_pairs

    English_word2index = {"SOS": SOS_token, "EOS": EOS_token}
    French_word2index = {"SOS": SOS_token, "EOS": EOS_token}

    # 获取英文和法文的词汇表
    for pair in en_fre_pairs:
        for word in pair[0].split(' '):
            if word not in English_word2index:
                English_word2index[word] = len(English_word2index)
        for word in pair[1].split(' '):
            if word not in French_word2index:
                French_word2index[word] = len(French_word2index)

    # print(len(English_word2index), len(French_word2index))

    English_index2word = {index: word for word, index in English_word2index.items()}
    French_index2word = {index: word for word, index in French_word2index.items()}
    # print(English_index2word, French_index2word)
    print("数据读取完毕。")
    return English_word2index, English_index2word, len(English_word2index), French_word2index, French_index2word, len(French_word2index), en_fre_pairs

English_word2index, English_index2word, English_word_len,\
French_word2index,French_index2word, French_word_len, en_fre_pairs = get_data()


# todo 定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, sentence_pairs, input_word2index, target_word2index, max_length=MAX_LENGTH):
        self.sentence_pairs = sentence_pairs
        self.input_word2index = input_word2index
        self.target_word2index = target_word2index
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):

        idx = min(max(idx, 0), len(self.sentence_pairs)) # 确保索引在范围内

        # input_sentence 是英文句子， target_sentence 是法文句子
        input_sentence, target_sentence = self.sentence_pairs[idx]
        input_indices = [self.input_word2index[word] for word in input_sentence.split(' ')] + [EOS_token]
        target_indices = [self.target_word2index[word] for word in target_sentence.split(' ')] + [EOS_token]

        # 填充或截断到最大长度
        # input_indices = input_indices[:self.max_length]
        # target_indices = target_indices[:self.max_length]

        input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)

        return input_tensor, target_tensor


def get_dataloader():
    # 数据多，测试时可以只取部分数据
    # en_fre_pairs_subset = en_fre_pairs[:1000]
    dataset = TranslationDataset(en_fre_pairs, English_word2index, French_word2index)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader


# todo 定义编码器类
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.input_size = input_size # 输入词典大小即英文单词数
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) # 词嵌入层

        # batch_first=True 表示输入输出张量的形状为 (batch_size, seq_len, feature_dim)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True) # GRU层

    def forward(self, input, hidden):
        # input: [batch_size, seq_len]
        embedded = self.embedding(input)  # [batch_size, seq_len, hidden_size]
        # output: [batch_size, seq_len, hidden_size], hidden: [1, batch_size, hidden_size]
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# todo 定义不带注意力机制的解码器类
class DecoderGRU(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size # 隐藏层大小 256
        self.output_size = output_size # 输出词典大小即法文单词数 4345

        self.embedding = nn.Embedding(output_size, hidden_size) # 词嵌入层

        # batch_first=True 表示输入输出张量的形状为 (batch_size, seq_len, feature_dim)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True) # GRU层

        self.out = nn.Linear(hidden_size, output_size) # 输出层 输入256 输出维度为法文字典大小4345

        self.softmax = nn.LogSoftmax(dim=-1) # LogSoftmax 层 数值归一化

    def forward(self, input, hidden):
        # input: [batch_size, seq_len=1]
        embedded = self.embedding(input)  # [batch_size, 1, hidden_size]
        embedded = F.relu(embedded) # 防止过拟合
        # output: [batch_size, 1, hidden_size], hidden: [1, batch_size, hidden_size]
        output, hidden = self.gru(embedded, hidden)

        # [1, 1, 256] -> [1, 256]
        output = self.out(output.squeeze(1))  # [batch_size, output_size] 去掉seq_len维度
        output = self.softmax(output)  # [batch_size, output_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def test_decoder():
    dataset = TranslationDataset(en_fre_pairs, English_word2index, French_word2index)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_size = English_word_len
    hidden_size = 256
    encoder = EncoderGRU(input_size, hidden_size)
    print(f'encoder {encoder}')

    decoder = DecoderGRU(French_word_len, hidden_size)
    print(f'encoder {decoder}')

    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        print(f"input_tensor shape: {input_tensor.shape}, target_tensor shape: {target_tensor.shape}")

        h0 = encoder.init_hidden()
        encoder_output, hn = encoder(input_tensor, h0)

        decoder_hidden = hn  # 使用编码器的最后隐藏状态作为解码器的初始隐藏状态

        for di in range(target_tensor.shape[1]):  # 遍历目标序列的每个时间步
            tmp = target_tensor[0][di].view(1, -1)  # 当前时间步的输入，形状为 (1, 1)
            decoder_output, decoder_hidden = decoder(tmp, decoder_hidden)
            print(f"Decoder output at time step {di}: shape {decoder_output.shape}")

        break

# todo 带注意力机制的解码器类
class AttentionDecoderGRU(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderGRU, self).__init__()
        self.hidden_size = hidden_size # 隐藏层大小 256
        self.output_size = output_size # 输出词典大小即法文单词数 4345
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size) # 词嵌入层(输入维度为 French_word_len 4345, 输出维度为 hidden_size 256)

        self.attention = nn.Linear(hidden_size * 2, max_length)  # 求Q注意力权重

        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size) # 拼接注意力结果和嵌入后的输入

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size) # 输出层 输入256 输出维度为法文字典大小4345

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size, seq_len=1]->[1, 1] -> Q 当前编码时，预测除的上一个词，初始为SOS
        # hidden: [1, batch_size, hidden_size]->[1, 1, 256] -> K 解码器上一层的隐藏层的输出结果，初始为编码器最后的隐藏层
        # encoder_outputs: [max_length, hidden_size]->[10 256] -> V 编码器所有时间步的输出结果

        embedded = self.embedding(input) # [1, 1] -> [1, 1, 256]
        embedded = F.relu(embedded) # 防止过拟合

        # 拼接注意力结果和嵌入后的输入 [1, 1, 256] + [1, 1, 256] -> [1, 1, 512] -> Linear[512, max_len] -> [1, 1, 10]
        attention_weights = F.softmax(self.attention(torch.cat((embedded, hidden), 2)), dim=2)

        attention_applied = torch.bmm(attention_weights,
                                      encoder_outputs.unsqueeze(0))  # [1, 1, 10] * [1, 10, 256] -> [1, 1, 256]

        embedded = torch.cat((embedded, attention_applied), dim=-1) # [1, 1, 512]

        attention_output = F.relu(self.attention_combine(embedded))  # [1, 1, 256]

        output, hidden = self.gru(attention_output, hidden)  # output: [1, 1, 256], hidden: [1, 1, 256]
        output = self.out(output.squeeze(1))
        output = self.softmax(output)

        return output, hidden, attention_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def test_attention_decoder():
    dataset = TranslationDataset(en_fre_pairs, English_word2index, French_word2index)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_size = English_word_len
    hidden_size = 256
    encoder = EncoderGRU(input_size, hidden_size)
    print(f'encoder {encoder}')

    attention_decoder = AttentionDecoderGRU(French_word_len, hidden_size)
    print(f'attention_decoder {attention_decoder}')

    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        print(f"input_tensor shape: {input_tensor.shape}, target_tensor shape: {target_tensor.shape}")

        h0 = encoder.init_hidden()
        output, hn = encoder(input_tensor, h0)

        decoder_hidden = hn  # 使用编码器的最后隐藏状态作为解码器的初始隐藏状态

        encoder_output = torch.zeros(MAX_LENGTH, hidden_size, device=device)
        for idx in range(output.shape[1]):
            encoder_output[idx] = output[0][idx] # 将编码器的输出按时间步存储起来

        for di in range(target_tensor.shape[1]):  # 遍历目标序列的每个时间步
            tmp = target_tensor[0][di].view(1, -1)  # 当前时间步的输入，形状为 (1, 1)
            decoder_output, decoder_hidden, attention_weights = attention_decoder(tmp, decoder_hidden,
                                                                                 encoder_output)
            print(f"Attention Decoder output at time step {di}: shape {decoder_output.shape}")
            print(f"Attention weights at time step {di}: shape {attention_weights.shape}")

        break

# 超参数设置
lr = 1e-4
epochs = 2
teacher_forcing_ratio = 0.5
print_interval_num = 1000
plot_interval_num = 100


def train():
    dataloader = get_dataloader()
    # 实例化编码器
    encoder = EncoderGRU(English_word_len, 256).to(device)
    # 实例化带注意力机制的解码器
    decoder = AttentionDecoderGRU(French_word_len, 256).to(device)
    # 优化器
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    # 损失函数
    criterion = nn.NLLLoss()

    plot_loss_list = [] # 用于绘制损失曲线

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}\n")
        print_loss_total, plot_loss_total = 0.0, 0.0
        start_time = time.time()
        for i, (input_tensor, target_tensor) in enumerate(tqdm(dataloader), start=1):
            loss = train_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            # print(loss)
            # print(input_tensor)
            # print(target_tensor)
            print_loss_total += loss
            plot_loss_total += loss
            if i % print_interval_num == 0: # 每1000次迭代打印一次平均损失
                print_loss_avg = print_loss_total / print_interval_num # 计算平均损失
                print_loss_total = 0.0
                elapsed_time = time.time() - start_time
                print(f"迭代次数: {i}, 平均损失: {print_loss_avg:.4f}, 用时: {elapsed_time:.2f}秒")
                start_time = time.time()
            if i % plot_interval_num == 0: # 每100次迭代记录一次平均损失用于绘图
                plot_loss_avg = plot_loss_total / plot_interval_num
                plot_loss_list.append(plot_loss_avg)
                plot_loss_total = 0.0
        # 保存模型
        torch.save(encoder.state_dict(), f'model/encoder_epoch{epoch}.pth') # 保存至 model 文件夹
        torch.save(decoder.state_dict(), f'model/decoder_epoch{epoch}.pth')

    # 绘制损失曲线
    plt.figure()
    plt.plot(plot_loss_list)
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.savefig('training_loss.png')
    plt.show()

# todo 内部训练函数
def train_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # 编码参数
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    # 解码参数
    encoder_outputs_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for idx in range(input_tensor.shape[1]):
        encoder_outputs_c[idx] = encoder_outputs[0][idx]
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device)  # 解码器初始输入为SOS_token

    loss = 0.0
    target_length = target_tensor.size(1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # 教师强制：将目标作为下一个输入
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs_c)
            loss += criterion(decoder_output, target_tensor[0][di].unsqueeze(0))  # 累加损失

            decoder_input = target_tensor[0][di].view(1, -1)  # 下一个输入是目标词 用真实的标签作为下一个输入
    else:
        # 非教师强制：使用自己的预测作为下一个输入
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs_c)
            loss += criterion(decoder_output, target_tensor[0][di].unsqueeze(0))  # 累加损失

            topv, topi = decoder_output.topk(1) # 选择概率最高的词作为预测结果
            decoder_input = topi.detach().view(1, -1)  # 下一个输入是预测词

            if decoder_input.item() == EOS_token:
                break

    # 梯度清零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(encoder, decoder, input_tensor):
    with torch.no_grad():
        # 与train类似
        # ------------------------------------------------
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        encoder_outputs_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for idx in range(input_tensor.shape[1]):
            encoder_outputs_c[idx] = encoder_outputs[0][idx]

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS_token]], device=device)  # 解码器初始输入为SOS_token
        # ------------------------------------------------

        # 自回归式解码
        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH) # 初始化注意力权重矩阵
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_outputs_c)
            decoder_attentions[di] = attention_weights

            topv, topi = decoder_output.topk(1)  # 选择概率最高的词作为预测结果
            if topi.item() == EOS_token:  # 遇到EOS_token则结束解码
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(French_index2word[topi.item()]) # 将预测结果加入解码结果列表

            decoder_input = topi.detach().view(1, -1)  # 下一个输入是预测词

        return decoded_words, decoder_attentions[:di + 1] # 可能不到MAX_LENGTH长度，因此需要截取



def use_evaluate():
    PATH1 = 'model/encoder_epoch2.pth'
    PATH2 = 'model/decoder_epoch2.pth'

    # 实例化编码器和解码器
    encoder = EncoderGRU(English_word_len, 256).to(device)
    decoder = AttentionDecoderGRU(French_word_len, 256).to(device)
    # 加载模型参数
    encoder.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage), False) # False表示不严格匹配
    decoder.load_state_dict(torch.load(PATH2, map_location=lambda storage, loc: storage), False)

    sample_pairs = [
      ['i m impressed with your french .', 'je suis impressionne par votre francais .'],
      ['i m more than a friend .', 'je suis plus qu une amie .'],
      ['she is beautiful like her mother .', 'elle est belle comme sa mere .']
    ]
    for idx, pair in enumerate(sample_pairs):
        print(f"样本 {idx+1}:")
        print(f"英文句子: {pair[0]}")
        print(f"真实法文句子: {pair[1]}")

        English_sentence = pair[0]
        English_sentence_index = [English_word2index[word] for word in English_sentence.split(' ')] + [EOS_token]
        English_sentence_index = torch.tensor(English_sentence_index, dtype=torch.long, device=device).view(1, -1)

        output_words, attentions = evaluate(encoder, decoder, English_sentence_index)
        output_sentence = ' '.join(output_words)

        print(f"翻译结果: {output_sentence}")
        print()




if __name__ == '__main__':
    # # en_fre_pairs = get_data()
    # # print(random.choice(en_fre_pairs))  # 随机打印一句话对进行检查
    #
    # # English_word2index, English_index2word, English_word_len, \
    # #     French_word2index, French_index2word, French_word_len, en_fre_pairs = get_data()
    # # dataset = TranslationDataset(en_fre_pairs, English_word2index, French_word2index)
    # # print(dataset.__getitem__(0))
    #
    # dataloader = get_dataloader()
    # # for i, (input_tensor, target_tensor) in enumerate(dataloader):
    # #     print(f"输入张量: {input_tensor}")
    # #     print(f"目标张量: {target_tensor}")
    # #     if i == 2:  # 只打印前3个样本
    # #         break
    #
    # encoder = EncoderGRU(English_word_len, 256).to(device)
    # for i, (input_tensor, target_tensor) in enumerate(dataloader):
    #     h0 = encoder.init_hidden()
    #     output, hn = encoder(input_tensor, h0)
    #     print(f"编码器输出形状: {output.shape}")
    #     print(f"编码器隐藏状态形状: {hn.shape}")
    #     if i == 2:  # 只打印前3个样本
    #         break
#     test_decoder()
#     attention_decoder = AttentionDecoderGRU(French_word_len, 256).to(device)
#     print(attention_decoder)

    # train()
    use_evaluate()

