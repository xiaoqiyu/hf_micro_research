# -*- coding: utf-8 -*-
# @time      : 2019/12/16 15:24
# @author    : rpyxqi@gmail.com
# @file      : dataloader_sample.py

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#

batch_size = 3
max_length = 3
hidden_size = 2
n_layers = 1
tensor_in = torch.FloatTensor([[1, 2, 3], [1, 1, 0], [2, 0, 0]]).resize_(3, 3, 1)
tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = [3, 2, 1]  # list of integers holding information about the batch size at each sequence step
# pack it
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
# initialize
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
# forward
out, hn = rnn(pack, h0)
# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
print(unpacked[0].size())
print(hn.size())


def get_features():
    train_x = [
        [[], [], []]]
#
# class SimpleCustomBatch:
#     def __init__(self, data):
#         transposed_data = list(zip(*data))
#         self.inp = torch.stack(transposed_data[0], 0)
#         self.tgt = torch.stack(transposed_data[1], 0)
#
#     # custom memory pinning method on custom type
#     def pin_memory(self):
#         self.inp = self.inp.pin_memory()
#         self.tgt = self.tgt.pin_memory()
#         return self
#
#
# def collate_wrapper(batch):
#     return SimpleCustomBatch(batch)
#
#
# inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
# tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
# dataset = TensorDataset(inps, tgts)
#
# loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
#                     pin_memory=True)
#
# for batch_ndx, sample in enumerate(loader):
#     print(sample.inp.is_pinned())
#     print(sample.tgt.is_pinned())
