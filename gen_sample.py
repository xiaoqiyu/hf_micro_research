#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: gen_sample.py
@time: 20-2-22 下午3:10
@desc:
'''

import os
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uqer
from uqer import DataAPI
import seaborn as sns

sns.set()

uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")


def get_samples(mode=0, *arge, **kwargs):
    ''' mode 对应的测试集
        0. 每种规模分别取前N，默认000300-沪深300，000001-上证综指，000905-中证500，399005-深证中小板指,399006-创业板指
        1. 每个行业龙头，主要市值排行
        2. 流动性看：成交量活跃度，按换手率，和成交量日间日内波动
        3.  异动股，日间（短期）、日内
        4.  风格： 动量、反转
        5. 对每种指数，300，500
    '''
    total_num = kwargs.get('total_num') or 100
    start_date = "20191201"
    end_date = "20191231"
    if mode == 0:
        # tickers = ['000300','000001','000905','399005','399006','399001']
        tickers = ['399005', '399006', '399001']
        ticker_to_exchangecd = {'000300': 'XSHG', '000001': 'XSHG', '000905': 'XSHG', '399005': 'XSHE',
                                '399006': 'XSHE',
                                '399001': 'XSHE'}
        ret = {}
        _num = int(total_num / len(tickers))
        for _ticker in tickers:
            df = DataAPI.mIdxCloseWeightGet(secID=u"", ticker=_ticker, beginDate=start_date, endDate=end_date,
                                            field=["consID", "weight"], pandas="1").sort_values(by='weight',
                                                                                                ascending=False)
            # TODO choose the top weights coons, then the correlations with the becnchmark is high, change to sample num
            # randomly
            sec_ids = df['consID'][:_num]
            ret.update({'{0}.{1}'.format(_ticker, ticker_to_exchangecd.get(_ticker)): sec_ids})
        return ret


if __name__ == "__main__":
    ret = get_samples(0)
    lst = []
    for k, v in ret.items():
        lst.extend(v)
    print(len(lst), len(set(lst)))
    pprint.pprint(ret)
