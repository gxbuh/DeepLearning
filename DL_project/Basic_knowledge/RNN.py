"""
实现一个基于RNN的文本生成模型，用于生成歌词
"""


import torch
import re
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_vocab():
    unique_words, all_words = [], [] # unique_words存放不重复的词，all_words存放所有词
    for line in open('data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line) # 分词
        all_words.append(words) # 把当前行的词加入到所有词列表中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # print('unique_words: ', len(unique_words))
    word2idx = {word: idx for idx, word in enumerate(unique_words)}
    corpus_idx = []
    for words in all_words:
        tmp = [word2idx[word] for word in words]
        tmp.append(word2idx[' ']) # 添加空格作为每行的结束符
        corpus_idx.extend(tmp)
    return unique_words, word2idx, len(unique_words), corpus_idx



class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx # 整个语料库的索引列表
        self.num_chars = num_chars   # 每个句子中的词个数
        self.word_count = len(corpus_idx) # 词典大小
        self.num_words = self.word_count // self.num_chars # 句子数量


    def __len__(self):
        return self.num_words # 返回句子数量

    def __getitem__(self, idx):
        start = min(max(0, idx ), self.word_count - self.num_chars - 1) # 当前样本起始索引
        end = start + self.num_chars # 当前样本结束索引

        input_seq = torch.tensor(self.corpus_idx[start:end]) # 输入序列
        target_seq = torch.tensor(self.corpus_idx[start + 1:end + 1]) # 目标序列，向后移动一个词
        return input_seq, target_seq


class TextGeneratorRNN(nn.Module):
    def __init__(self, unique_word_count):
        super(TextGeneratorRNN, self).__init__()
        self.embedding = nn.Embedding(unique_word_count, 128) # 词嵌入层
        self.rnn = nn.RNN(128, 256, 1) # RNN层
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, unique_word_count) # 全连接层

    def forward(self, x, hidden):
        x = self.embedding(x) # 词嵌入 格式：(batch句子的数量，句子长度，每个词的维度)
        out, hidden = self.rnn(x.transpose(0, 1), hidden) # RNN计算 格式：(句子长度, batch句子的数量，隐藏状态维度)
        out = self.dropout(out)
        out = out.reshape(shape=(-1, out.shape[-1])) # 调整输出形状
        out = self.dropout(out)
        out = self.fc(out) # 全连接层
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 256) # 初始化隐藏状态


def train():
    unique_words, word2idx, vocab_size, corpus_idx = build_vocab()
    lyrics_dataset = LyricsDataset(corpus_idx, 32)
    dataloader = DataLoader(lyrics_dataset, batch_size=32, shuffle=True)
    model = TextGeneratorRNN(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()

        for inputs, targets in dataloader:
            hidden = model.init_hidden(inputs.size(0)).to(device)  # 初始化隐藏状态
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, hidden = model(inputs, hidden)
            targets = torch.transpose(targets, 0, 1).reshape(shape=(-1, )) # 调整目标形状
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}')


    torch.save(model.state_dict(), 'model/lyrics_rnn_model.pth')

def evaluate(start_word, sentence_len):
    unique_words, word2idx, vocab_size, corpus_idx = build_vocab()
    model = TextGeneratorRNN(vocab_size).to(device)
    model.load_state_dict(torch.load('model/lyrics_rnn_model.pth'))
    model.eval()

    hidden = model.init_hidden(1).to(device)
    word_idx = word2idx[start_word] # 获取起始词
    generate_sentence = [word_idx] # 产生词的索引
    for i in range(sentence_len):
        output, hidden = model(torch.tensor([[word_idx]]).to(device), hidden)
        word_idx = torch.argmax(output) # 获取概率最大的词的索引
        generate_sentence.append(word_idx) # 把预测结果添加到列表中

    for idx in generate_sentence:
        print(unique_words[idx], end=' ')



if __name__ == '__main__':
    # unique_words, word2idx, vocab_size, corpus_idx = build_vocab()
    # print(f'词典大小: {vocab_size}')
    # dataset = LyricsDataset(corpus_idx, num_chars=5)
    # print(f'<UNK>: {len(dataset)}')

    # train()

    evaluate(start_word='星星', sentence_len=50)