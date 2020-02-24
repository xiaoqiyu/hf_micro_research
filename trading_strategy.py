# -*- coding: utf-8 -*-
# @time      : 2019/10/16 11:29
# @author    : rpyxqi@gmail.com
# @file      : trading_strategy.py

from WindPy import w
import talib as ta
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

w.start()


def get_trade_schedule(participant_rate=0.15, target_ratio=0.01, sec_code="002299.SZ", start_date="",
                       end_date="2019-10-15", period=100, target_vol=2150000, target_period=20, price_ratio=0.97):
    _t_days = w.tdays("2017-01-14", end_date)
    t_days = [item.strftime('%Y-%m-%d') for item in _t_days.Data[0]]

    ret = w.wsd(sec_code, "close,volume,turn,open,chg,pct_chg,total_shares", t_days[-period], end_date, "")
    # print(ret.Data)
    vols = ret.Data[1]
    turns = ret.Data[2]
    total_shares = ret.Data[-1]
    price_ma5 = np.array(ret.Data[0][-5:]).mean() * price_ratio
    price_prev = ret.Data[0][-1] * price_ratio
    # print("price line is :{0}".format(ma5))
    ma_vols = list(ta.SMA(np.array(list(vols), dtype=float), timeperiod=5))
    # ma_turns = list(ta.SMA(np.array(list(turns), dtype=float), timeperiod=5))

    idx = -1
    total_vol = 0.0
    ret_vols = []

    # compute target vol
    target_vol = target_vol or target_ratio * total_shares[-1]
    # participant rate testing
    # p_rates_up = []
    #     # p_rates_down = []
    #     # for target_period in [10, 20, 30, 40, 50, 60]:
    #     #     p_rates_up.append(100 * (target_vol / (sum(ma_vols[-target_period:]) * 1.05)))
    #     #     p_rates_down.append(100 * (target_vol / (sum(ma_vols[-target_period:]) * .95)))
    #     # return p_rates_up, p_rates_down
    # update the participant_rate according to the target complete period
    if target_period:
        participant_rate = target_vol / sum(ma_vols[-target_period:])
        print('updated participant rate is:{0}'.format(participant_rate))

    while total_vol < target_vol:
        if target_vol - total_vol <= 100:
            ret_vols.append(100)
            break
        try:
            _vol = int(ma_vols[idx] * participant_rate / 100) * 100
        except:
            break
        ret_vols.append(_vol)
        total_vol += _vol
        idx -= 1
    return ret_vols, [price_prev, price_ma5]


def get_schedule(**kwargs):
    sec_code = kwargs.get('sec_code')
    end_date = kwargs.get('end_date')
    if not sec_code or not end_date:
        raise ValueError('sec_code and end_date should not be empty')
    participant_rate = kwargs.get('participant_rate') or 0.15
    target_ratio = kwargs.get('target_ratio') or 0.01
    period = kwargs.get('period') or 100
    target_vol = kwargs.get('target_vol') or 8000000
    target_period = kwargs.get('target_period')
    price_ratio = kwargs.get('price_ratio') or 0.95
    update = kwargs.get('update') or False
    ret_vol, ret_price = get_trade_schedule(participant_rate=participant_rate, target_ratio=target_ratio,
                                            sec_code=sec_code, end_date=end_date, period=period, target_vol=target_vol,
                                            target_period=target_period, price_ratio=price_ratio)
    # FIXME fix the border
    print(sum(ret_vol), target_vol)
    next_date = w.tdaysoffset(1, end_date).Data[0][0].strftime('%Y-%m-%d')
    sec_code = sec_code
    vol = ret_vol[0]
    price = min(ret_price)
    df = pd.DataFrame([[sec_code, next_date, vol, price]], columns=['sec_code', 'trade_date', 'vol', 'price'])
    curr_df = pd.read_csv('data/trade_strategy.csv')
    curr_df = curr_df.append(df)
    if update:
        curr_df.to_csv('data/trade_strategy.csv', index=False)
    return df


def nasida():
    # 高位：10-21-11.53;10-16:10.69; 中位：6-3：9.36；5-22：9.98；低位：1-31：23.12%；11-09：18.28;2-13;08-09:
    # 中：6-4-7.45；10-21：11.53；低：1-4：
    ret_vols5, ret_prices = get_trade_schedule(participant_rate=0.1, target_ratio=0.01, sec_code="002180.SZ",
                                               end_date="2019-10-11", period=100, target_vol=None, target_period=10,
                                               price_ratio=0.98)
    ret_vols10, ret_prices = get_trade_schedule(participant_rate=0.1, target_ratio=0.01, sec_code="002180.SZ",
                                                end_date="2018-06-04", period=100, target_vol=None, target_period=22,
                                                price_ratio=0.98)
    ret_vols15, ret_prices = get_trade_schedule(participant_rate=0.08, target_ratio=0.01, sec_code="002180.SZ",
                                                end_date="2019-01-04", period=100, target_vol=None, target_period=None,
                                                price_ratio=0.98)
    ret_vols20, ret_prices = get_trade_schedule(participant_rate=0.05, target_ratio=0.01, sec_code="002180.SZ",
                                                end_date="2019-01-04", period=100, target_vol=None, target_period=22,
                                                price_ratio=0.98)
    # 8-9
    print(ret_vols5)
    print(len(ret_vols15))
    plt.plot([item / 100 for item in ret_vols5])
    plt.plot([item / 100 for item in ret_vols10])
    plt.plot([item / 100 for item in ret_vols15])
    plt.plot([item / 100 for item in ret_vols20])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(["情形1", "情形2", "情形3-a", "情形3-b"])
    plt.xlabel("第i个交易日")
    plt.ylabel("每日交易量（手）")
    plt.title("纳思达减持1%股份交易路径模拟")
    plt.show()


def trading_schedules(*args, **kwargs):
    sec_code, end_date = args
    high = kwargs.get('high') or 30
    low = kwargs.get('low') or 23
    target_price = kwargs.get('target_price') or 30
    target_period = kwargs.get('period') or 20
    cache_period = kwargs.get('cache_period') or 500
    high_ratio = kwargs.get('high_ratio') or 0.1
    mid_ratio = kwargs.get('mid_ratio') or 0.075
    low_ratio = kwargs.get('low_ratio') or 0.005
    target_ratio = kwargs.get('target_ratio') or 0.01
    target_vol = kwargs.get('target_vol')
    vol_adj = kwargs.get('vol_adj') or 5
    mock_period = kwargs.get('mock_period') or 50
    _t_days = w.tdays("2010-01-14", end_date)
    t_days = [item.strftime('%Y-%m-%d') for item in _t_days.Data[0]]
    ret = w.wsd(sec_code, "close,volume,turn,open,chg,pct_chg,total_shares", t_days[-cache_period], end_date, "")
    total_shares = ret.Data[-1][-1]
    vols = ret.Data[1]
    turns = ret.Data[2]
    closes = ret.Data[0]
    pct_chgs = ret.Data[-2]
    std = np.array([item / 100 for item in pct_chgs[-120:]]).std()
    sigma = std * math.sqrt(250)
    print(std, sigma)
    curr_price = closes[-1]

    _price_intercept = (target_price - curr_price) / target_period
    mock_prices = []
    r = _price_intercept * math.sqrt(250) / math.sqrt(target_period)
    T = 0.04
    I = 10000
    target_vol = int(target_vol or total_shares * target_ratio)
    for idx in range(10):
        _path = []
        S0 = curr_price
        for _i in range(mock_period):
            ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.standard_normal(1))
            _path.append(ST1[0])
            S0 = ST1[0]
        mock_prices.append(_path)
    _mean_prices = list(np.array(mock_prices).mean(axis=0))
    _mean_prices.insert(0, curr_price)
    # plt.plot(_mean_prices)
    # plt.show()
    _mock_vols = vols[-5:]
    total_vol = 0.0
    for idx in range(mock_period):
        vol_ma5 = sum(_mock_vols[idx: idx + 5])
        close = _mean_prices[idx]
        _participant_ratio = low_ratio if close < low else high_ratio if close > high else mid_ratio
        _vol = int(vol_ma5 / 5 * _participant_ratio / 100) * 100
        print(_vol)
        if target_vol - total_vol <= 100:
            _mock_vols.append(target_vol - total_vol)
            total_vol = target_vol
            print(target_vol, total_shares)
            print('branch 1', target_vol - total_vol)
        elif target_vol > total_vol:
            _real_vol = int(min(_vol, target_vol - total_vol))
            # FIXME add 90 days total vol
            _mock_vols.append(_real_vol)
            total_vol += _real_vol
            print('branch 2', _real_vol)
        else:
            _mock_vols.append(0)
            print('branch 3', 0)
    import pprint
    print(sum(_mock_vols))
    print(target_vol)
    # pprint.pprint(sum(_mock_vols), target_vol)


def get_strategy_snfz(end_date='', sec_code='002299.SZ', target_share=0.03):
    _t_days = w.tdays("2017-01-14", end_date)
    t_days = [item.strftime('%Y-%m-%d') for item in _t_days.Data[0]]
    ret = w.wsd(sec_code, "close,volume,turn,open,chg,pct_chg,total_shares", t_days[-20], end_date, "")
    vols = ret.Data[1]
    turns = ret.Data[2]
    total_shares = ret.Data[-1]
    avg_turn = sum(turns) / len(turns)
    participant_rate_low = 0.1 / avg_turn
    print('low participant rate%', participant_rate_low * 100)
    # print('low trading vol', total_shares[-1] * 0.001)
    avg_vol = sum(vols[-10:]) / len(vols[-10:])
    print('24-24.5 target vol:', min(avg_vol, total_shares[-1] * 0.001) / 10000, total_shares[-1] * 0.001 / 10000,
          avg_vol / 10000)
    print('>24.5 target vol: ')


if __name__ == '__main__':
    # trading_schedules('002180.SZ', '20191122')
    # x = np.random.standard_normal(100)
    # print(x.std(), x.mean(), len(x))
    # plt.plot(x)
    # plt.show()
    # get_strategy_snfz(end_date='20200102', sec_code='002299.SZ', target_share=0.03)
    nasida()