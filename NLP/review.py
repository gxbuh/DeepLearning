import torch
import torch.nn as nn

embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[[1, 2, 5, 6, 4, 3, 2, 9]]])
print(embedding(input).shape)





