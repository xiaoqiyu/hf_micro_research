#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: get_vix_mkt.py
@time: 2020/3/8 18:08
@desc:
'''
import pandas as pd
from WindPy import w

w.start()


def get_vix_mkt(sec_ids=["VIXY.P", "TVIX.O", "UVXY.P", "VXX.BAT", "SPX.GI"], fields="close,pct_chg,turn,volume"):
    close_lst = []
    dates=[]
    for sec_id in sec_ids:
        ret = w.wsd(sec_id, fields, "2015-01-01", "2020-03-07", "unit=1;TradingCalendar=AMEX")
        close_lst.append(ret.Data[0])
        dates = ret.index()
        print(dates)
    df = pd.DataFrame(close_lst)
    df = df.T
    df.columns = sec_ids
    df.to_csv('close.csv')



get_vix_mkt()
