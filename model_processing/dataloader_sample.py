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
import numpy as np
import pandas as pd
import os

from utils.helper import get_full_data_path
from utils.logger import Logger
from utils.helper import get_full_data_path
from utils.helper import get_full_model_path

logger = Logger().get_log()
#

# 定义常量
INPUT_SIZE = 49  # 定义输入的特征数
HIDDEN_SIZE = 32  # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = 32  # batch
EPOCH = 10  # 学习次数
LR = 0.001  # 学习率
TIME_STEP = 20  # 步长，一般用不上，写出来就是给自己看的
DROP_RATE = 0.2  # drop out概率
LAYERS = 2  # 有多少隐层，一个隐层一般放一个LSTM单元
MODEL = 'LSTM'  # 模型名字

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


def standadize(arr):
    arr = arr.replace(np.inf, 0.0)
    arr = arr.replace(-np.inf, 0.0)
    _max, _min = arr.max(), arr.min()
    return (arr - _min) / (_max - _min)


def _get_min(bartime_lst):
    ret = []
    for item in bartime_lst:
        _h, _m = item.split(':')
        ret.append((int(_h) - 9) * 60 + int(_m))
    _max, _min = max(ret), min(ret)
    return [(item - _min) / (_max - _min) for item in ret]


def load_features(all_features=False, security_id=None):
    ret = os.listdir(get_full_data_path())
    lst = []
    for item in ret:
        if item.endswith('csv') and 'corr' not in item and (not security_id or (item.startswith(security_id))):
            _df = pd.read_csv(get_full_data_path(item))
            # index, exchangeCD, ticker, dataDate
            # barTime: change to the offset minutes since the market start, and the relative time span in the day
            bar_time_lst = _df['barTime']
            # label5 = _df['label5']
            label = _df['label']
            _del_col = list(set(_df.columns).intersection(
                {'index', 'exchangeCD', 'ticker', 'dataDate', 'barTime', 'barTime.1', 'index.1', 'label5', 'label',
                 'ma20', 'maDiff20'}))
            _df.drop(
                _del_col,
                axis=1,
                inplace=True)
            _df = _df.apply(standadize, axis=0)
            bar_time_lst = _get_min(bar_time_lst)
            _df['barTime'] = bar_time_lst
            # _df['label5'] = label5
            _df['label'] = label
            if not all_features:
                return _df
            if _df['label'][0] == 1.0 or ('barTime' not in _df.columns) or (_df['barTime'][0] != _df['barTime'][0]):
                logger.debug('verify data')
            lst.append(_df)
    df = pd.concat(lst)
    all_feature_path = get_full_data_path('all_features_{0}.csv'.format(security_id))
    df.to_csv(all_feature_path)
    return df


def _get_ts_loader(val, targets):
    idx = 0
    img = []
    n_features = val.shape[0]
    # TODO  double check whether to handle by each day(barTime=1.0)
    while idx <= n_features - TIME_STEP:
        try:
            img.append(np.column_stack([val[idx:idx + TIME_STEP], targets[idx:idx + TIME_STEP].reshape(-1, 1)]))
        except Exception as ex:
            logger.debug("contact with error:{0}".format(ex))
        idx += 1
    train_tensor = torch.Tensor(np.array(img))
    train_img = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_img


def get_dataloader(train_ratio=0.5, val_ratio=0.3, df=None):
    df = load_features(all_features=False, security_id='002415.XSHE')

    print(df.shape)
    targets = np.array([1 if item >= 0 else 0 for item in df['label']])  # 2 classes
    # it seems the label here does not hv big influence to the results
    df.drop(['label'], axis=1, inplace=True)
    val = df.values
    img = []
    idx = 0
    n_features = df.shape[0]
    np.random.seed(42)
    perm_idx = list(range(n_features))
    np.random.shuffle(perm_idx)
    train_idx = int(n_features * train_ratio)
    val_idx = int(n_features * val_ratio)
    test_idx = int(n_features * (1 - train_ratio - val_ratio))
    train_val = val[perm_idx[:train_idx], :]
    val_val = val[perm_idx[train_idx:train_idx + val_idx], :]
    test_val = val[perm_idx[-test_idx:], :]
    train_target = targets[perm_idx[:train_idx]]
    val_target = targets[perm_idx[train_idx:train_idx + val_idx]]
    test_target = targets[perm_idx[-test_idx:]]

    train_loader = _get_ts_loader(train_val, train_target)
    val_loader = _get_ts_loader(val_val, val_target)
    test_loader = _get_ts_loader(test_val, test_target)
    return train_loader, val_loader, test_loader


# 定义LSTM的结构
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, 2)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output


def train_lstm():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = load_features(all_features=False, security_id='002415.XSHE')

    rnn = lstm().to(device)  # 使用GPU或CPU
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
    loss_func = nn.CrossEntropyLoss()  # 分类问题
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                               milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)
    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    for i in range(EPOCH):
        total_train_loss = []
        rnn.train()  # 进入训练模式
        step_cnt = 0
        train_loader, valid_loader, test_loader = get_dataloader(train_ratio=0.7, val_ratio=0.3, df=df)
        for step, item in enumerate(train_loader):
            # FIXME change this size control, change to shuffle and new size
            step_cnt += 1
            if step_cnt > 10:
                break
            # lr = set_lr(optimizer, i, EPOCH, LR)
            nx, ny, nz = item.shape
            blocks = torch.chunk(item, nz, dim=2)
            b_x = torch.cat(blocks[:-1], 2)
            b_y = blocks[-1]
            # b_x, b_y = item
            b_x = b_x.type(torch.FloatTensor).to(device)  #
            b_y = b_y.type(torch.long).to(device)  # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
            prediction = rnn(b_x)  # rnn output    # prediction (4, 72, 2)
            #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
            #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y[:, -1, :].view(b_y.size()[0]))  # 计算损失，target要转1-D
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))  # 存入平均交叉熵

        step_valid_loss = []
        rnn.eval()
        val_cnt = 0
        for step, item in enumerate(valid_loader):
            # FIXME remove hardcode
            val_cnt += 1
            if val_cnt > 3:
                break
            nx, ny, nz = item.shape
            blocks = torch.chunk(item, nz, dim=2)
            b_x = torch.cat(blocks[:-1], 2)
            b_y = blocks[-1]
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.long).to(device)
            with torch.no_grad():
                prediction = rnn(b_x)  # rnn output
            #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
            #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y[:, -1, :].view(b_y.size()[0]))  # calculate loss
            step_valid_loss.append(loss.item())
        valid_loss.append(np.mean(step_valid_loss))

        if valid_loss and valid_loss[-1] < min_valid_loss:
            logger.info("Save model in epoch:{0} with valid_loss:{1}".format(i, valid_loss))
            torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, './LSTM.model')  # 保存字典对象，里面'model'的value是模型
            torch.save(optimizer, './LSTM.optim')  # 保存优化器
            min_valid_loss = valid_loss[-1]
        logger.info('Epoch: {0}, loss is:{1}'.format(i, np.mean(total_train_loss)))
        logger.info('Epoch: {0}, Current learning rate: {1}'.format(i, mult_step_scheduler.get_lr()))
        mult_step_scheduler.step()  # 学习率更新


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


def data_loader_test():
    tensor_in = torch.FloatTensor([[1, 2, 3], [1, 1, 0], [2, 0, 0]]).resize_(3, 3, 1)
    # print(tensor_in)

    # x = np.random.random(200).reshape(100, 5, 4)
    x = np.array(range(200)).reshape(10, 5, 4)
    y = np.random.random(10).reshape(-1, 1)
    # x.append(y)
    tensor_in = torch.from_numpy(x)
    print(tensor_in.shape)
    v = Variable(tensor_in)
    data_loader = DataLoader(tensor_in, batch_size=2, shuffle=False)
    for bx, x in enumerate(data_loader):
        print(bx)
        print(x.shape)
        # print(x)

if __name__ == '__main__':
    # tensor_in = torch.FloatTensor([[1, 2, 3], [1, 1, 0], [2, 0, 0]]).resize_(3, 3, 1)
    # # print(tensor_in)
    #
    # # x = np.random.random(200).reshape(100, 5, 4)
    # x = np.array(range(200)).reshape(10, 5, 4)
    # y = np.random.random(10).reshape(-1, 1)
    # # x.append(y)
    # tensor_in = torch.from_numpy(x)
    #
    # print(tensor_in.shape)
    # v = Variable(tensor_in)
    # data_loader = DataLoader(tensor_in, batch_size=2, shuffle=False)
    # for bx, x in enumerate(data_loader):
    #     print(bx)
    #     print(x.shape)
    #     # print(x)

    # get_dataloader()
    train_lstm()
