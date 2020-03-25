#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: lstm_cls_models.py
@time: 2020/3/22 17:32
@desc:
'''

import torch
import torch.nn as nn
import numpy as np
# import torchvision
from torch.utils.data import DataLoader
from datetime import datetime  # 用于计算时间
from data_processing.hf_features import load_features

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

# 设置GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(0)


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
        self.hidden_out = nn.Linear(HIDDEN_SIZE, 10)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output


def get_dataloader():
    df = load_features(all_features=False, security_id='002415.XSHE')
    print(df.shape)
    targets = [1 if item >= 0 else 0 for item in df['label']]  # 2 classes
    # it seems the label here does not hv big influence to the results
    df.drop(['label'], axis=1, inplace=True)
    val = df.values
    img = []
    labels = []
    idx = 0
    n_features = df.shape[0]
    # TODO  double check whether to handle by each day(barTime=1.0)
    while idx <= n_features - TIME_STEP:
        img.append((val[idx: idx + TIME_STEP], targets[idx: idx + TIME_STEP][-1]))
        labels.append(targets[idx: idx + TIME_STEP][-1])
        idx += 1
    # train_tensor = torch.Tensor(val.reshape(-1, TIME_STEP, INPUT_SIZE))

    train_tensor = torch.Tensor(np.array(img).reshape(-1, TIME_STEP, INPUT_SIZE))
    train_img = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=False)
    train_target = torch.Tensor(np.array(labels).reshape(-1, 1))

    for idx, x in enumerate(train_img):
        print(idx, x.shape)
    return train_img, train_target


#
# def get_features():
#     # 进行归一化，分割数据集
#     # 获取mnist的数据
#     train_data = torchvision.datasets.MNIST(
#         root='./mnist/',
#         train=True,
#         transform=torchvision.transforms.ToTensor(),
#         download=False  # 需要下载数据集，就设置为True
#     )
#     test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
#
#     print('train_data.train_data.size():', train_data.train_data.size())  # 打印训练集特征的size
#     print('train_data.train_labels.size():', train_data.train_labels.size())  # 打印训练集标签的size
#
#     # 先归一化，在分割数据，为节约时间，只取了部分
#     data_x = train_data.train_data.type(torch.FloatTensor) / 255.
#     data_y = train_data.train_labels.type(torch.Tensor)
#
#     train_x = data_x[:10000]
#     train_y = data_y[:10000]
#     valid_x = data_x[10000:12000]
#     valid_y = data_y[10000:12000]
#     test_x = data_x[12000:14000]
#     test_y = data_y[12000:14000]
#
#     data_train = list(
#         train_x.numpy().reshape(1, -1, TIME_STEP, INPUT_SIZE))  # 使用list只会把最外层变为list，内层还是ndarray，和.tolist()方法不同
#     data_valid = list(valid_x.numpy().reshape(1, -1, TIME_STEP, INPUT_SIZE))
#     data_test = list(test_x.numpy().reshape(1, -1, TIME_STEP, INPUT_SIZE))
#     data_train.append(list(train_y.numpy().reshape(-1, 1)))
#     data_valid.append(list(valid_y.numpy().reshape(-1, 1)))
#     data_test.append(list(test_y.numpy().reshape(-1, 1)))
#
#     data_train = list(zip(*data_train))  # 最外层是list，次外层是tuple，内层都是ndarray
#     data_valid = list(zip(*data_valid))
#     data_test = list(zip(*data_test))
#
#     train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)
#     valid_loader = DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)
#     test_loader = DataLoader(data_train, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)
#     return train_loader, valid_loader, test_loader


def train_models(train_loader, valid_loader):
    rnn = lstm().to(device)  # 使用GPU或CPU
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
    loss_func = nn.CrossEntropyLoss()  # 分类问题
    # 定义学习率衰减点，训练到50%和75%时学习率缩小为原来的1/10
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                               milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)

    # 训练+验证
    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    for i in range(EPOCH):
        total_train_loss = []
        rnn.train()  # 进入训练模式
        for step, (b_x, b_y) in enumerate(train_loader):
            # lr = set_lr(optimizer, i, EPOCH, LR)
            b_x = b_x.type(torch.FloatTensor).to(device)  #
            b_y = b_y.type(torch.long).to(device)  # CrossEntropy的target是longtensor，且要是1-D，不是one hot编码形式
            prediction = rnn(b_x)  # rnn output    # prediction (4, 72, 2)
            #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
            #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))  # 计算损失，target要转1-D，注意b_y不是one hot编码形式
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))  # 存入平均交叉熵

        total_valid_loss = []
        rnn.eval()
        for step, (b_x, b_y) in enumerate(valid_loader):
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.type(torch.long).to(device)
            with torch.no_grad():
                prediction = rnn(b_x)  # rnn output
            #         h_s = h_s.data        # repack the hidden state, break the connection from last iteration
            #         h_c = h_c.data        # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))  # calculate loss
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))

        if (total_valid_loss < min_valid_loss):
            torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, './LSTM.model')  # 保存字典对象，里面'model'的value是模型
            #         torch.save(optimizer, './LSTM.optim')     # 保存优化器
            min_valid_loss = total_valid_loss

        # 编写日志
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      min_valid_loss,
                                                                      optimizer.param_groups[0]['lr'])
        mult_step_scheduler.step()  # 学习率更新
        # 服务器一般用的世界时，需要加8个小时，可以视情况把加8小时去掉
        print(str(datetime.datetime.now() + datetime.timedelta(hours=8)) + ': ')
        print(log_string)  # 打印日志
        # log('./LSTM.log', log_string)  # 保存日志


# def test_models(test_loader):
#     # 测试
#     best_model = torch.load('./LSTM.model').get('model').cuda()
#     best_model.eval()
#     final_predict = []
#     ground_truth = []
#
#     for step, (b_x, b_y) in enumerate(test_loader):
#         b_x = b_x.type(torch.FloatTensor).to(device)
#         b_y = b_y.type(torch.long).to(device)
#         with torch.no_grad():
#             prediction = best_model(b_x)  # rnn output
#         #     h_s = h_s.data        # repack the hidden state, break the connection from last iteration
#         #     h_c = h_c.data        # repack the hidden state, break the connection from last iteration
#
#         loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))  # calculate loss
#
#         ground_truth = ground_truth + b_y.view(b_y.size()[0]).cpu().numpy().tolist()
#         final_predict = final_predict + torch.max(prediction[:, -1, :], 1)[1].cpu().data.numpy().tolist()
#
#     ground_truth = np.asarray(ground_truth)
#     final_predict = np.asarray(final_predict)
#
#     accuracy = float((ground_truth == final_predict).astype(int).sum()) / float(final_predict.size)
#     print(accuracy)


if __name__ == '__main__':
    get_dataloader()
