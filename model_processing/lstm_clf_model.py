#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: lstm_clf_model.py
@time: 2020/3/30 13:47
@desc:
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.helper import get_full_data_path
from utils.logger import Logger
from utils.helper import get_full_data_path
from utils.helper import get_full_model_path
from data_processing.hf_features import cache_features

logger = Logger().get_log()
#

# 定义常量
INPUT_SIZE = 49  # 定义输入的特征数
HIDDEN_SIZE = 32  # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = 32  # batch
EPOCH = 20  # 学习次数
LR = 0.001  # 学习率
TIME_STEP = 20  # 步长，一般用不上，写出来就是给自己看的
DROP_RATE = 0.2  # drop out概率
LAYERS = 4  # 有多少隐层，一个隐层一般放一个LSTM单元
MODEL = 'LSTM'  # 模型名字
# the valid criterier could be cross_entropy_loss or accuracy, this only applies for valid, not for training
VALID_CRITERIER = 'cross_entropy_loss'
NUM_LABEL = 3


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


def _feature_preporcessing(df):
    bartime_dummy = df['barTime']
    # _df['label5'] = label5
    # labels = df['label']
    # targets = np.array([0 if item >= 0 else 1 for item in labels])
    df.drop(['barTime'], axis=1, inplace=True)
    df = df.apply(standadize, axis=0)
    df['barTime'] = bartime_dummy
    # df['targets'] = targets
    return df


def load_features(all_features=False, security_id=None):
    ret = os.listdir(get_full_data_path())
    lst = []
    for item in ret:
        if item.endswith('csv') and 'corr' not in item and (not security_id or (item.startswith(security_id))):
            _df = pd.read_csv(get_full_data_path(item))
            # index, exchangeCD, ticker, dataDate
            # barTime: change to the offset minutes since the market start, and the relative time span in the day
            bar_time_lst = _df['barTime']
            _del_col = list(set(_df.columns).intersection(
                {'index', 'exchangeCD', 'ticker', 'barTime', 'barTime.1', 'index.1', 'label5',
                 'ma20', 'maDiff20'}))
            _df.drop(
                _del_col,
                axis=1,
                inplace=True)
            bar_time_lst = _get_min(bar_time_lst)
            _df['barTime'] = bar_time_lst
            if not all_features:
                return _df
            if _df['label'][0] == 1.0 or ('barTime' not in _df.columns) or (_df['barTime'][0] != _df['barTime'][0]):
                logger.debug('verify data')
            lst.append(_df)
    df = pd.concat(lst)
    all_feature_path = get_full_data_path('all_features_{0}.csv'.format(security_id))
    df.to_csv(all_feature_path)
    return df


def _get_ts_loader(val=None, targets=None):
    idx = 0
    img = []
    n_features = val.shape[0]
    # TODO  double check whether to handle by each day(barTime=1.0)
    while idx <= n_features - TIME_STEP:
        try:
            if isinstance(targets, np.ndarray):
                img.append(np.column_stack([val[idx:idx + TIME_STEP], targets[idx:idx + TIME_STEP].reshape(-1, 1)]))
            else:
                img.append(val[idx:idx + TIME_STEP])
        except Exception as ex:
            logger.debug("contact with error:{0}".format(ex))
        idx += 1
    train_tensor = torch.Tensor(np.array(img))
    train_img = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_img


def get_dataloader(train_ratio=0.7, df=None):
    logger.debug('feature shape:{0}'.format(df.shape))
    # targets = np.array([1 if item >= 0 else 0 for item in df['label']])  # 2 classes
    targets = np.array([0 if item < 0 else 1 if item == 0 else 2 for item in df['label']])  # 3 classes
    df = df.drop(['label', 'dataDate'], axis=1)
    n_features = df.shape[0]
    train_num = int(n_features * train_ratio)
    val_num = n_features - train_num
    df_train = df.head(train_num)
    df_val = df.tail(val_num)
    df_train = _feature_preporcessing(df_train)
    df_val = _feature_preporcessing(df_val)

    train_val = df_train.values
    val_val = df_val.values
    train_target = targets[:train_num]
    val_target = targets[-val_num:]

    train_loader = _get_ts_loader(train_val, train_target)
    val_loader = _get_ts_loader(val_val, val_target)
    return train_loader, val_loader


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
        self.hidden_out = nn.Linear(HIDDEN_SIZE, NUM_LABEL)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output

    def predict(self, x):
        _distribution = F.softmax(self.forward(x)[:, -1, :])
        return torch.argmax(_distribution, dim=1)


def train_lstm(test_date='2019-12-02', security_id='002415.XSHE'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = load_features(all_features=True, security_id=security_id)
    train_val_df = df[df.dataDate < test_date]
    test_df = df[df.dataDate == test_date]

    rnn = lstm().to(device)  # 使用GPU或CPU
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
    loss_func = nn.CrossEntropyLoss()  # 分类问题
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                               milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)
    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    best_accuracy = 0.0
    for i in range(EPOCH):
        total_train_loss = []
        rnn.train()  # 进入训练模式
        train_loader, valid_loader = get_dataloader(train_ratio=0.7, df=train_val_df)
        for step, item in enumerate(train_loader):
            # lr = set_lr(optimizer, i, EPOCH, LR)
            nx, ny, nz = item.shape
            blocks = torch.chunk(item, nz, dim=2)
            b_x = torch.cat(blocks[:-1], 2)
            b_y = blocks[-1]
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
        total_num = 0
        correct_num = 0
        # best_score = 0.0

        rnn.eval()
        for step, item in enumerate(valid_loader):
            nx, ny, nz = item.shape
            blocks = torch.chunk(item, nz, dim=2)
            b_x = torch.cat(blocks[:-1], 2)
            b_y = blocks[-1]
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.long).to(device)

            if VALID_CRITERIER == 'cross_entropy_loss':
                with torch.no_grad():
                    prediction = rnn(b_x)  # rnn output
                #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
                #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
                loss = loss_func(prediction[:, -1, :], b_y[:, -1, :].view(b_y.size()[0]))  # calculate loss
                step_valid_loss.append(loss.item())

            elif VALID_CRITERIER == 'accuracy':
                targets = rnn.predict(b_x)
                # correct_tensor = (targets == b_y[:, -1, :].view(b_y.size()[0])).tolist()
                correct_num += sum((targets == b_y[:, -1, :].view(b_y.size()[0])).tolist())
                total_num += targets.shape[0]
            else:
                logger.warn("Invalid valid_criterier: {0}".format(VALID_CRITERIER))
        if VALID_CRITERIER == 'cross_entropy_loss':
            logger.info('Epoch:{0}, mean train loss:{1},std train loss:{2}, valid loss is:{3}'.format(i, np.mean(
                total_train_loss), np.std(total_train_loss), valid_loss))
            valid_loss.append(np.mean(step_valid_loss))
            if valid_loss and valid_loss[-1] < min_valid_loss:
                logger.info("Save model in epoch:{0} with valid_loss:{1}".format(i, valid_loss))
                torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                            'valid_loss': valid_loss},
                           get_full_model_path('LSTM_{0}.model'.format(security_id)))  # 保存字典对象，里面'model'的value是模型
                torch.save(optimizer, get_full_model_path('LSTM_{0}.optim'.format(security_id)))  # 保存优化器
                min_valid_loss = valid_loss[-1]
        elif VALID_CRITERIER == 'accuracy':
            _accuracy = float(correct_num / total_num)
            logger.info('Epoch:{0}, mean train loss:{1},std train loss:{2}, accucury is:{3}'.format(i, np.mean(
                total_train_loss), np.std(total_train_loss), _accuracy))
            if _accuracy > best_accuracy:
                best_accuracy = _accuracy
                logger.info(
                    "Save model in epoch:{0} with valid_accuracy:{1}, and best accuracy:{2}".format(i, _accuracy,
                                                                                                    best_accuracy))
                torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                            'valid_accuracy': _accuracy},
                           get_full_model_path('LSTM_{0}.model'.format(security_id)))  # 保存字典对象，里面'model'的value是模型
                torch.save(optimizer, get_full_model_path('LSTM_{0}.optim'.format(security_id)))  # 保存优化器

        else:
            logger.warn("Invalid valid_criterier: {0}".format(VALID_CRITERIER))

        logger.info('Epoch: {0}, Current learning rate: {1}'.format(i, mult_step_scheduler.get_lr()))
        mult_step_scheduler.step()  # 学习率更新
    plt.plot(train_loss, color='r')
    plt.plot(valid_loss, color='b')
    plt.legend(['train_loss', 'valid_loss'])
    train_loss_track_path = get_full_model_path('train_loss_{0}.jpg'.format(security_id))
    # plt.show()
    plt.savefig(train_loss_track_path)


def predict_with_lstm(date='2019-12-02', inputs=None, predict_sample={'399005.XSHE': ['002415.XSHE']}):
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
        rnn_dict = torch.load(get_full_model_path('LSTM.model'))
        return rnn_dict.get('model').predict(inputs)
    test_sample = predict_sample or {'399005.XSHE': ['002415.XSHE']}
    from collections import defaultdict
    ret_labels = defaultdict(dict)
    ret_accuracy = {}
    for mkt, sec_ids in test_sample.items():
        for sec_id in sec_ids:
            # test for one day
            _df = cache_features(start_date=date, end_date=date, test_sample={mkt: [sec_id]}, saved=False)
            _bar_time_lst = _df['barTime'].iloc[:, 0]
            _del_col = list(set(_df.columns).intersection(
                {'index', 'exchangeCD', 'ticker', 'barTime', 'barTime.1', 'index.1', 'label5',
                 'ma20', 'maDiff20'}))
            _df.drop(
                _del_col,
                axis=1,
                inplace=True)
            bar_time_lst = _get_min(_bar_time_lst)
            _df['barTime'] = bar_time_lst
            targets = np.array([1 if item >= 0 else 0 for item in _df['label']])  # 2 classes
            targets = np.array([0 if item < 0 else 1 if item == 0 else 2 for item in _df['label']])  # 3 classes
            _df = _df.drop(['label', 'dataDate'], axis=1)
            _df = _feature_preporcessing(_df)
            val = _df.values
            _data_loader = _get_ts_loader(val=val, targets=None)
            x = torch.from_numpy(val).float()
            _full_model_path = get_full_model_path('LSTM_{0}.model'.format(sec_id))
            rnn_dict = torch.load(_full_model_path)
            ret_predicts = rnn_dict.get('model').predict(_data_loader.dataset.float()).numpy()
            n_predict = ret_predicts.shape[0]
            correct_num = [1 if item == ret_predicts[idx] else 0 for idx, item in enumerate(targets[-n_predict:])]
            ret_labels.update({sec_id: dict(zip(_bar_time_lst[-n_predict:], ret_predicts))})
            ret_accuracy.update({sec_id: float(sum(correct_num) / n_predict)})
    return ret_labels, ret_accuracy


if __name__ == '__main__':
    # load_features(all_features=True, security_id='002415.XSHE')
    # get_dataloader()
    # train_lstm()
    # x = np.random.random(60 * 49).reshape(3, 20, 49)
    # print(predict_with_lstm(x))
    import pprint

    predicts, accuaracy = predict_with_lstm(date='2019-12-03')
    pprint.pprint(predicts)
    pprint.pprint(accuaracy)
