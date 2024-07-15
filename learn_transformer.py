import torch
import torch.nn as nn

embedding = nn.Embedding(10, 8)
transformer = nn.Transformer(d_model=8, batch_first=True).eval()

src = torch.LongTensor([[0, 1, 2, 3, 4]])
tgt = torch.LongTensor([[0, 2, 1, 2, 4]])

print(transformer(embedding(src), embedding(tgt[:, :1])))

print(transformer(embedding(src), embedding(tgt[:, :2]), tgt_mask=transformer.generate_square_subsequent_mask(2)))