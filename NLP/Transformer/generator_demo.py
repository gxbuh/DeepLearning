import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from input_demo import *
from encoder_demo import *
from decoder_demo import *

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        """
        d_model: 词嵌入维度
        vocab: 词表大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def test_generator():
    decoder_output = test_decoder()
    generator = Generator(d_model=512, vocab=1000)
    generator_output = generator(decoder_output)
    print(f'生成器输出的维度为：{generator_output.shape}')
    print(generator_output)

if __name__ == '__main__':
    test_generator()