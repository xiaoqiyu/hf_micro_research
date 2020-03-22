#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: trade_simulation.py
@time: 2020/3/8 17:28
@desc:
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import uqer
from uqer import DataAPI
import seaborn as sns
from utils.logger import Logger
import math
from data_processing.hf_features import get_features_by_date

sns.set()
logger = Logger(log_level='INFO').get_log()

uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")


def get_market_impacts():
    '''
    return tmp impact and perm impact
    '''
    return 0.001, 0.002


def _sample_intraday_trend(labels=[], presicion_score=0.7):
    _len = len(labels)
    if not _len:
        return None
    total_correct_num = int(_len * presicion_score)
    _random_sample = np.random.random(1000) * _len
    correct_set = set()
    idx = 0
    while len(correct_set) <= total_correct_num:
        logger.debug("idx:{0}, set num:{1}".format(idx, len(correct_set)))
        correct_set.add(int(_random_sample[idx]))
        idx += 1
    ret = [item if idx in correct_set else abs(item - 1) for idx, item in enumerate(labels)]
    return ret


# TODO 日内TWAP-> VWAP, 边界处理，增加冲击成本,限价条件
def get_features(security_id='', start_date='', end_date='', min_unit="5"):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date.replace('-', ''),
                             endDate=end_date.replace('-', ''), isOpen=u"1",
                             field=u"calendarDate", pandas="1")
    t_dates = list(df['calendarDate'])
    df_min = DataAPI.MktBarHistDateRangeGet(securityID=security_id, startDate=start_date.replace('-', ''),
                                            endDate=end_date.replace('-', ''),
                                            unit=min_unit, field=u"", pandas="1")

    df_min['vwap'] = df_min['totalValue'] / df_min['totalVolume']
    # TODO normally it should be closePrice
    df_min['ret'] = df_min[['vwap']].rolling(2).apply(lambda x: x[-1] / x[0] - 1)
    df_min['label'] = [1 if item >= 0.0 else 0 for item in (([0.0] + list(df_min['ret']))[1:])]
    # df_min['dealPrice'] = (df_min['openPrice'] + df_min['vwap'])/2
    df_min['dealPrice'] = df_min['vwap']
    return df_min


def get_sim_results(participant_ratio=0.15, target_vol=10000000, trade_date='2019-12-02', mode=2,
                    intraday_precision=0.7,
                    intraday_adj=0.1, df_min=None):
    df_agg = df_min.groupby('dataDate').agg(
        {
            'totalVolume': ['sum'],
        })
    flatten_columns = ['{0}_{1}'.format(item[0], item[1]) for item in df_agg.columns]
    df_agg.columns = flatten_columns
    df_agg = df_agg.reset_index()
    df_agg['vol_mean'] = df_agg[['totalVolume_sum']].rolling(5).apply(lambda x: x.mean() * participant_ratio)
    df_agg.sort_values(by='dataDate', inplace=True)
    vol_dict = dict(zip(df_agg['dataDate'][1:], df_agg['vol_mean'][:-1]))
    total_vol = 0
    total_value = 0.0
    daily_vol_lst = []
    daily_value_lst = []
    daily_vwap_lst = []
    daily_price_lst = []
    all_dates = []
    for date, vol in vol_dict.items():
        if date < trade_date:
            continue
        if total_vol >= target_vol:
            logger.info('complete trading on date:{0} with total_vol:{1}'.format(date, total_vol))
            break
        all_dates.append(date)
        _df = df_min[df_min.dataDate == date]
        daily_vol = min(vol, target_vol - total_vol)
        _vol = [daily_vol / 48] * _df.shape[0]
        daily_vol = min(vol, target_vol - total_vol)
        if mode == 1:
            total_vol += sum(_vol)
            if total_vol > target_vol:
                _vol[-1] -= total_vol - target_vol
                total_vol = target_vol
            elif target_vol - total_vol < 100:
                _vol[-1] += target_vol - total_vol
                total_vol = target_vol
            # total_value += sum(_df['vwap'] * _vol)
            daily_vol_lst.append(sum(_vol))
            _value = sum(_df['dealPrice'] * _vol)
            daily_value_lst.append(_value)
            daily_vwap_lst.append(_value / sum(_vol))
        elif mode == 2:
            _predictions = _sample_intraday_trend(list(_df['label']), intraday_precision)
            adj_vol = [_vol[0]]
            for idx, item in enumerate(_predictions[:-1]):
                adj_vol.append(_vol[idx] * (1 + intraday_adj) if item > 0.0 else _vol[idx] * (1 - intraday_adj))
            total_vol += sum(adj_vol)
            if total_vol > target_vol:
                adj_vol[-1] -= total_vol - target_vol
                total_vol = target_vol
            elif target_vol - total_vol < 100:
                adj_vol[-1] += target_vol - total_vol
                total_vol = target_vol
            # total_value += sum(_df['vwap'] * adj_vol)
            daily_vol_lst.append(sum(adj_vol))
            _value = sum(_df['dealPrice'] * adj_vol)
            daily_value_lst.append(sum(_df['dealPrice'] * adj_vol))
            daily_vwap_lst.append(_value / sum(adj_vol))
            daily_price_lst.append(_df['dealPrice'].mean())
            # if _value / sum(adj_vol) < sum(_df['dealPrice'] * _vol) / sum(_vol):
            # print(date, _value / sum(adj_vol), _df['dealPrice'].mean())
    # print(intraday_adj, sum(daily_value_lst) / sum(daily_vol_lst))
    return daily_vol_lst, daily_value_lst, daily_vwap_lst, daily_price_lst, all_dates


def get_transaction():
    df = DataAPI.MktBarHistDateRangeGet(securityID=u"002180.XSHE", startDate=u"20191205", endDate=u"20191231", unit="5",
                                        field=u"", pandas="1")
    df_agg = df.groupby('dataDate').agg(
        {
            'totalVolume': ['max', 'min', 'mean'],
        })
    flatten_columns = ['{0}_{1}'.format(item[0], item[1]) for item in df_agg.columns]
    df_agg.columns = flatten_columns
    df_agg = df_agg.reset_index()
    df_agg['vol_diff'] = (df_agg['totalVolume_max'] - df_agg['totalVolume_min']) / df_agg['totalVolume_mean']
    df_agg.sort_values(by='vol_diff', ascending=False, inplace=True)
    print(df_agg)


def use_case_sim(security_id='002180.XSHE', participant_ratio=0.2, start_end='20191101', end_date='20191231',
                 trade_date='2019-12-05', passive_ratio=0.5, mode=1):
    '''
    mode = 1: all execute by passive(twap/vwap); mode = 2: all execute by liquididy; mode=3: ratio of passive execute is
    passive_ratio, others by liquidity
    '''
    prev_trade_date = '2019-12-04'

    df = DataAPI.MktEqudGet(secID=security_id, tradeDate=u"", beginDate=start_end, endDate=end_date, isOpen="",
                            field=["tradeDate", "chgPct", "closePrice", "turnoverVol"], pandas="1")
    df['ret_std'] = df[['chgPct']].rolling(20).apply(lambda x: x.std())
    df['vol_sum'] = df[['turnoverVol']].rolling(5).apply(lambda x: x.sum())

    trade_row = df[df.tradeDate == trade_date]
    prev_row = df[df.tradeDate == prev_trade_date]
    daily_vol = list(trade_row['ret_std'])[0]
    total_vol = list(prev_row['vol_sum'])[0]
    passive_ratio = 1.0 if mode == 1 else (0.0 if mode == 2 else passive_ratio)
    passive_vol = total_vol * passive_ratio
    liquid_vol = total_vol - passive_vol

    df = DataAPI.MktBarHistDateRangeGet(securityID=security_id, startDate=trade_date.replace('-', ''),
                                        endDate=trade_date.replace('-', ''), unit="1",
                                        field=u"", pandas="1")
    # df.sort_values(by='totalVolume', ascending=True, inplace=True)
    df['vwap'] = df['totalValue'] / df['totalVolume']
    s = passive_vol * participant_ratio / 240
    adv = total_vol
    sigma = daily_vol * math.sqrt(245)

    df_tick = get_features_by_date(security_id=u"002180.XSHE", date='20191205', min_unit="1", tick=True)
    ask_vol_sum = {}
    bid_vol_sum = {}
    df_tick = df_tick[df_tick.barTime <= "14:57"]
    df_tick = df_tick[df_tick.barTime > '09:30']
    for row in (df_tick[['barTime', 'totalAskVolume', 'totalBidVolume']].values):
        _ask_tmp = ask_vol_sum.get(row[0]) or 0.0
        _bid_tmp = bid_vol_sum.get(row[0]) or 0.0
        ask_vol_sum.update({row[0]: _ask_tmp + row[1]})
        bid_vol_sum.update({row[0]: _bid_tmp + row[2]})
    q_dict = {}
    for k, v in bid_vol_sum.items():
        q_dict.update({k: abs(v - ask_vol_sum.get(k))})

    df = df[df.barTime <= "14:57"]
    df = df[df.barTime > '09:30']
    mkt_vols = list(df['totalVolume'])
    _tmp = [s / item for item in mkt_vols]
    participant_min = dict(zip(df['barTime'], _tmp))
    vt_dict = dict(zip(df['barTime'], df['totalVolume']))
    total_value = 0.0
    total_vol = 0
    # TODO get the TWAP/VWAP simulate results (with and without market impact), check market impact results
    if passive_ratio > 0:
        istar, tmp_dict, perm_dict = get_market_impact(s, adv, sigma, vt_dict, q_dict)
        ret_rows = []
        for bar_time, item in df[['barTime', 'vwap']].values:
            total_vol += s
            # _tmp_impact = tmp_dict.get(bar_time)
            _tmp_impact = istar
            total_value += item * s * (1 - _tmp_impact / 10000)
            ret_rows.append([bar_time, istar, tmp_dict.get(bar_time), perm_dict.get(bar_time)])
        mi_df = pd.DataFrame(ret_rows, columns=['barTime', 'istar', 'tmp', 'perm'])
        mi_df.to_csv('mi_report_mode_{0}_{1}.csv'.format(mode, passive_ratio))

    if passive_ratio < 1.0:
        print('liquid target_vol is: ', liquid_vol * participant_ratio)
        liquid_trade_value, liquid_trade_vol, finish_time = get_liquid_results(df_tick, liquid_vol * participant_ratio,
                                                                               ratio=0.13,
                                                                               level=1, price_rule=1)
        print(liquid_trade_value, liquid_trade_vol, liquid_trade_value / liquid_trade_vol, finish_time)
        total_value += liquid_trade_value
        total_vol += liquid_trade_vol
        # liquid_value, liquid_vol, finish_time = get_liquid_results(df_tick, liquid_vol * participant_ratio, ratio=0.20,
        #                                                            level=2, price_rule=1)
        # print(liquid_value, liquid_value, liquid_value / liquid_vol, finish_time)

    print(total_value, total_vol, total_value / total_vol)


def get_market_impact(s, adv, sigma, vt_dict, q_dict):
    '''
    return instance impact, tmporary impact, and permenant impact, all in bp
    '''
    a1, a2, a3, a4, b1 = 2431.9, 0.52, 0.92, 1.00, 0.84
    # Asia emerging market parameters
    a1, a2, a3, a4, b1 = 1225.76, 0.75, 0.70, 0.50, 0.86
    istar = a1 * math.pow(s / adv, a2) * math.pow(sigma, a3)
    perm_impact_dict = {}
    tmp_impact_dict = {}

    participant_ret = []

    for k, v in q_dict.items():
        if vt_dict.get(k):
            try:
                tmp_impact_dict.update({k: b1 * istar * math.pow(v / vt_dict.get(k), a4)})
                perm_impact_dict.update({k: (1 - b1) * istar})
                participant_ret.append([k, s / vt_dict.get(k)])
            except Exception as ex:
                logger.warn('error for time:{0} with msg:{1}'.format(k, ex))
    participant_df = pd.DataFrame(participant_ret, columns=['barTime', 'participant_ratio'])
    # participant_df = participant_df.T
    participant_df.to_csv('participant_ratio.csv')
    return istar, tmp_impact_dict, perm_impact_dict


def get_liquid_results(df_tick, target_vol, ratio, level, price_rule):
    df_tick.sort_values(by='dataTime', ascending=True, inplace=True)
    n_row = df_tick.shape[0]
    total_vol = 0
    total_values = 0.0
    idx = 0
    rows = df_tick.values
    columns = list(df_tick.columns)
    finish_time = ''
    cnt = 0
    logger.debug("target vol for liquidity:{0}".format(target_vol))
    while idx < n_row and total_vol < target_vol:
        row = rows[idx]
        cnt += 1
        avg_price = 0.0 if total_vol == 0 else total_values / total_vol
        mkt_avg_price = row[columns.index('avgPrice')]
        mkt_mid_price = row[columns.index('midPrice')]
        mkt_last_price = row[columns.index('lastPrice')]
        if price_rule and not (mkt_last_price > avg_price and mkt_mid_price > mkt_avg_price):
            idx += 3
            continue
        finish_time = row[columns.index('dataTime')]
        bid_vol1 = row[columns.index('bidVolume1')]
        bid_vol2 = row[columns.index('bidVolume2')] if level == 2 else 0
        bid_price1 = row[columns.index('bidPrice1')]
        bid_price2 = row[columns.index('bidPrice2')]
        _vol = int((bid_vol1 + bid_vol2) * ratio / 100) * 100
        total_vol += _vol
        if target_vol - total_vol < 100:
            _vol += (target_vol - total_vol)
            total_vol = target_vol
        # print(idx, n_row, total_vol, target_vol, _vol)
        _value1 = bid_price1 * (bid_vol1 if bid_vol1 <= _vol else _vol)
        _value2 = bid_price2 * max(_vol - bid_vol1, 0)
        total_values += (_value1 + _value2)
        idx += 3
    logger.debug("total execution count:{0}".format(cnt))
    return total_values, total_vol, finish_time


def strategy3_report():
    # case 1&2 report simulation
    features = get_features(security_id='002180.XSHE', start_date='20191125', end_date='20191231', min_unit="5")
    sum_alpha = []
    mean_alpha = []
    for participant_ratio in [0.1, 0.15, 0.2]:
        for adj_ratio in [0.15, 0.2, 0.25, 0.3]:
            vol2, value2, vwap2, avg_prices, dates = get_sim_results(participant_ratio=0.15, target_vol=10000000,
                                                                     trade_date='2019-12-02', mode=2,
                                                                     intraday_precision=0.75, intraday_adj=adj_ratio,
                                                                     df_min=features)
            alpha = [(item - avg_prices[idx]) / avg_prices[idx] * 10000 for idx, item in enumerate(vwap2)]
            df = pd.DataFrame({'tradeDate': dates, 'strategy_vwap': vwap2, 'mkt_vwap': avg_prices, 'alpha(bp)': alpha})
            df.to_csv("case2_report_{0}_{1}.csv".format(adj_ratio, participant_ratio))
            print(participant_ratio, adj_ratio, sum(alpha), sum(alpha) / len(alpha))
            sum_alpha.append(sum(alpha))
            mean_alpha.append(sum(alpha) / len(alpha))
    print(max(sum_alpha), min(sum_alpha), sum(sum_alpha) / len(sum_alpha))
    print(sum_alpha)


def strategy4_report():
    # use case simulation
    # use_case_sim(security_id='002180.XSHE', participant_ratio=0.15, start_end='20191101', end_date='20191231',
    #              trade_date='2019-12-05', passive_ratio=0.7, mode=1)
    # use_case_sim(security_id='002180.XSHE', participant_ratio=0.15, start_end='20191101', end_date='20191231',
    #              trade_date='2019-12-05', passive_ratio=0.7, mode=2)
    # use_case_sim(security_id='002180.XSHE', participant_ratio=0.15, start_end='20191101', end_date='20191231',
    #              trade_date='2019-12-05', passive_ratio=0.5, mode=3)
    use_case_sim(security_id='002180.XSHE', participant_ratio=0.15, start_end='20191101', end_date='20191231',
                 trade_date='2019-12-05', passive_ratio=0.7, mode=3)


if __name__ == '__main__':
    # strategy3_report()
    # strategy4_report()
    # df = DataAPI.MktTickRTIntraDayGet(securityID=u"300694.XSHE", startTime=u"09:30", endTime=u"14:57", field=u"", pandas="1")
    # df['spread'] = df['bidPrice1'] - df['askPrice1']
    # df['volPerDeal'] = df['volume']/df['deal']
    # print(df.shape)

    df = \
        DataAPI.MktBarHistDateRangeGet(securityID=u"300694.XSHE", startDate=u"20200203", endDate=u"20200311", unit="60",
                                       field=u"", pandas="1")[['barTime', 'totalVolume']]
    print(df.columns)
