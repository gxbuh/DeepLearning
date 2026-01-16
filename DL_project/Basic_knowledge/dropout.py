import torch
import torch.nn as nn
from torch.nn.functional import dropout


def test():

    dropout = nn.Dropout(p=0.5)  # 定义 Dropout 层，丢弃概率为 0.5

    m = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)