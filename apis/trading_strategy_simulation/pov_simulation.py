#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: pov_simulation.py
@time: 2020/6/4 13:52
@desc:
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
import uqer
from uqer import DataAPI
from utils.logger import Logger
import math
from data_processing.hf_features import get_features_by_date
from model_processing.lstm_clf_model import predict_with_lstm
from data_processing.gen_sample import get_samples
from model_processing.lstm_clf_model import lstm
from account_info.get_account_info import get_account_info

logger = Logger(log_level='DEBUG').get_log()

uqer_client = uqer.Client(token=get_account_info().get('uqer_token'))


def pov_simulation(security_id=u"002180.XSHE", trade_date='20191205', start_time='13:00:00', end_time='13:30:30',
                   direction=0,
                   must_be_filled=True, participant_rate=20,
                   frequency=3, trackerror=0.5, limit_price=28.88, price_type='b1', price_bc=1, price_fp=0.1):
    df_tick = get_features_by_date(security_id=security_id, date=trade_date, min_unit="1", tick=True)
    # rpr = participant_rate / (1 - participant_rate)
    _filter_df = df_tick[df_tick.dataTime >= start_time]
    vwap_daily = df_tick['value'].sum() / df_tick['volume'].sum()

    _before_df = df_tick[df_tick.dataTime < start_time]
    # curr_time = datetime.datetime.now()
    curr_time = datetime.datetime.strptime('{0} {1}'.format(trade_date, start_time), '%Y%m%d %H:%M:%S')
    curr_time_str = start_time
    # acc_mkt_vol = _before_df['volume'].sum()
    acc_mkt_vol = 0.0
    accu_vol = 0
    pred_vol = _before_df.tail()['volume'].mean()
    daily_vol_mkt = _before_df['volume'].sum()
    daily_val_mkt = _before_df['value'].sum()
    print(df_tick.shape)
    real_end_time_str = "14:45:00" if must_be_filled else end_time
    _filter_df = _filter_df[_filter_df.dataTime < real_end_time_str]
    real_start_time = datetime.datetime.strptime('{0} {1}'.format(trade_date, start_time), '%Y%m%d %H:%M:%S')
    real_end_time = datetime.datetime.strptime('{0} {1}'.format(trade_date, real_end_time_str), '%Y%m%d %H:%M:%S')
    set_end_time = datetime.datetime.strptime('{0} {1}'.format(trade_date, end_time), '%Y%m%d %H:%M:%S')
    acc_val = 0.0
    vwap_period = _filter_df['value'].sum() / _filter_df['volume'].sum()
    while curr_time >= real_start_time and curr_time < real_end_time:
        if acc_mkt_vol:
            print(accu_vol / acc_mkt_vol, float(participant_rate / 100) - float(trackerror / 100),
                  accu_vol / acc_mkt_vol < float(participant_rate / 100) - float(trackerror / 100))
        if curr_time >= set_end_time and must_be_filled and (
                not acc_mkt_vol or (accu_vol / acc_mkt_vol >= abs(
            float(participant_rate / 100) - float(trackerror / 100)))):
            break
        # vol = rpr * acc_mkt_vol + rpr * pred_vol - accu_vol / (1 - participant_rate)
        vol = int((float(participant_rate / 100) * (acc_mkt_vol + pred_vol) - accu_vol) / 100) * 100

        print(curr_time.strftime('%H:%M:%S'), vol / 10000, accu_vol / 10000)
        try:
            _curr_row = _filter_df[_filter_df.dataTime == curr_time.strftime('%H:%M:%S')]
            _curr_vol = list(_curr_row['volume'])[0]
            _curr_val = list(_curr_row['value'])[0]
            _last_price = list(_curr_row['lastPrice'])[0]
            _curr_price = _curr_val / _curr_vol
            daily_vol_mkt += _curr_vol
            daily_val_mkt += _curr_val
            print('expected rate: {0}, real rate:{1}'.format(vol / pred_vol, vol / _curr_vol))
            acc_mkt_vol += _curr_vol
            pred_vol = (pred_vol + _curr_vol) / 2
            # handle price strategy
            if accu_vol and acc_val / accu_vol >= _curr_price * (
                    1 - price_fp / 100) and acc_val / accu_vol <= _curr_price(1 + price_fp / 100):
                vol = 0
            acc_val += _curr_price * vol
            accu_vol += vol
        except Exception as ex:
            print(ex)
        curr_time = curr_time + datetime.timedelta(seconds=frequency)
        print(
            'updated time:{0}, updated acc_mkt_vol:{1}, pred_vol:{2},acc_vol:{3}'.format(curr_time, acc_mkt_vol / 10000,
                                                                                         pred_vol / 10000,
                                                                                         accu_vol / 10000))
    print(acc_mkt_vol, accu_vol, accu_vol / acc_mkt_vol)
    print(acc_val / accu_vol, vwap_daily, vwap_period)


if __name__ == "__main__":
    pov_simulation(security_id=u"002180.XSHE", trade_date='20191205', start_time='13:00:00', end_time='13:30:30',
                   direction=0,
                   must_be_filled=True, participant_rate=20,
                   frequency=3, trackerror=0.5, limit_price=28.88, price_type='b1', price_bc=1, price_fp=0.1)
